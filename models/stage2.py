from typing import List, Any, Dict, Union, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel
from x_transformers import Decoder
from torch import Tensor
from x_transformers.x_transformers import TokenEmbedding, AbsolutePositionalEmbedding
# from tokenizer import VQTokenizer

class VQ_Decoder(nn.Module):
    def __init__(self,
                dim: int = 512,
                depth: int = 12,
                heads: int = 8,
                use_alibi_positional_bias: bool = False,
                 **kwargs):
        super(VQ_Decoder, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.use_alibi_positional_bias = use_alibi_positional_bias

        if use_alibi_positional_bias:
            self.model = Decoder(
                    dim = self.dim,
                    depth = self.depth,
                    heads = self.heads,
                    attn_flash = True,
                    alibi_pos_bias = True, # turns on ALiBi positional embedding
                    alibi_num_heads = self.heads // 2    # only use ALiBi for 4 out of the 8 heads, so other 4 heads can still attend far distances
                )
        else:
            self.model = Decoder(
                    dim = self.dim,
                    depth = self.depth,
                    heads = self.heads,
                    attn_flash = True,
                )

    def forward(self, x: Tensor, **kwargs) -> Union[Tensor, dict]:
        batch_size, context_length, embedding_size = x.shape
        return self.model.forward(x)


class VQ_SVG_Stage2(nn.Module):
    def __init__(self,
                tokenizer = None,  # must be of type VQTokenizer but I cannot import it here because of circular imports
                max_seq_len: int = 512,
                dim: int = 512,
                depth: int = 12,
                heads: int = 8,
                text_encoder_str: str = "bert-base-uncased",
                use_alibi_positional_bias = True,
                device = "cpu",
                freeze_text_encoder = True,
                 **kwargs):
        super(VQ_SVG_Stage2, self).__init__()

        self.text_encoder_str : str = tokenizer.text_encoder_str
        self.vq_vocab_size : int = tokenizer.num_tokens
        self.special_token_mapping : dict = tokenizer.special_token_mapping
        self.patch_idx_range : Tuple[int, int] = tokenizer._get_patch_idx_range()
        self.pos_idx_range : Tuple[int, int] = tokenizer._get_pos_idx_range()
        self.device = device

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.use_alibi_positional_bias = use_alibi_positional_bias

        if not self.use_alibi_positional_bias:
            self.pos_emb = AbsolutePositionalEmbedding(self.dim, max_seq_len)
        self.vq_embedding = TokenEmbedding(dim, self.vq_vocab_size)
        self.text_embedder: BertModel = BertModel.from_pretrained(text_encoder_str).to(device)
        
        if freeze_text_encoder:
            print("[INFO] Freezing the text encoder (BERT) weights.")
            for param in self.text_embedder.parameters():
                param.requires_grad = False

        if self.text_embedder.config.hidden_size != self.dim:
            self.mapping_layer = nn.Linear(self.text_embedder.config.hidden_size, self.dim)
        else:
            self.mapping_layer = nn.Identity()

        self.transformer = VQ_Decoder(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            use_alibi_positional_bias=self.use_alibi_positional_bias
        )

        self.final_linear = nn.Linear(self.dim, self.vq_vocab_size)

    def loss_function(self, targets: Tensor, pred_logits: Tensor, **kwargs) -> dict:
        loss = F.cross_entropy(pred_logits,targets)
        return {'loss': loss}

    def _combine_text_and_vq(self, text_tokens: Tensor,text_attn_mask:Tensor, vq_tokens: Tensor) -> Tensor:
        """
        This is the function that assembles the text and vq tokens together with an <SOS> token as pre-fix
        returns an embedded version of [<SOS>, <CLS>, text, <SEP>, <T_PAD>*, <BOS>, vq, <EOS>, <V_PAD>*]

        requires text_attn_mask for the BERT encoder
        """
        bs = text_tokens.shape[0]
        device = text_tokens.device
        with torch.no_grad():
            text_embedding = self.text_embedder.forward(text_tokens, attention_mask=text_attn_mask)
            text_embedding = text_embedding.last_hidden_state

        text_embedding = self.mapping_layer.forward(text_embedding)  # (bs, max_text_len, dim)
        text_embedding[~(text_attn_mask.bool())] = 0.0  # remove impact of padding tokens
        vq_embeddings = self.vq_embedding.forward(vq_tokens)  # (bs, max_vq_len, dim)
        sos_embedding = self.vq_embedding.forward(torch.ones(bs, 1, dtype=torch.long, device=device) * self.special_token_mapping['<SOS>'])  # (bs, 1, dim)
        
        stacked_embeddings = torch.cat([sos_embedding, text_embedding, vq_embeddings], dim=1)
        if stacked_embeddings.shape[1] > self.max_seq_len:
            print(f"[WARN] Input sequence length ({stacked_embeddings.shape[1]}) exceeds maximum sequence length ({self.max_seq_len}). Truncating input sequence.")
            stacked_embeddings = stacked_embeddings[:, :self.max_seq_len, :]
        if not self.use_alibi_positional_bias:
            stacked_embeddings = stacked_embeddings + self.pos_emb.forward(stacked_embeddings)

        return stacked_embeddings

    def forward(self, text_tokens: Tensor, text_attn_mask:Tensor, vq_tokens: Tensor, **kwargs) -> Tuple[Tensor, dict]:
        stacked_embeddings = self._combine_text_and_vq(text_tokens,text_attn_mask, vq_tokens)
        # TODO the attention mask should here also be used to zero out attention of the decoder to the text padding tokens
        out = self.transformer.forward(stacked_embeddings)
        out = self.final_linear.forward(out)
        out = out[:, text_tokens.shape[-1] + 1:]  # remove the predictions for the text token range +1 (<SOS> token that was added during embedding)
        return out, {}
    
    def generate(self,
                 text_tokens: Tensor,
                 attention_mask: Tensor,
                 vq_tokens: Tensor,
                 temperature:float = 0.0) -> Union[Tensor, str]:
        """
        Returns the generated sequence of VQ tokens and the reason for stopping the generation.
        """
        
        assert self.pos_idx_range[0] >= self.patch_idx_range[1], "pos_idx_range must start after patch_idx_range ends"
        assert vq_tokens.ndim == 2 and vq_tokens.size(0) == 1, "VQ_Tokens must be of shape (1, sequence_length) and contain at least the <BOS> token"

        if (vq_tokens[:, -1] >= self.pos_idx_range[0]).all() and (vq_tokens[:, -1] <= self.pos_idx_range[1]).all():
            required_token = "patch"
        elif (vq_tokens[:, -1] >= self.patch_idx_range[0]).all() and (vq_tokens[:, -1] <= self.patch_idx_range[1]).all():
            required_token = "pos"
        elif (vq_tokens[:, -1] < self.patch_idx_range[0]).all():  # e.g. only <BOS> tokens in input
            required_token = "patch"
        else:
            raise ValueError("Last tokens in Input must be of the same type (special, patch, or pos).")

        with torch.no_grad():
            while vq_tokens.shape[1] < self.max_seq_len:
                predictions, _ = self.forward(text_tokens, attention_mask ,vq_tokens)
                predictions[:, -1, self.special_token_mapping["<PAD>"]] = -torch.inf  # mask the padding token
                if required_token == "patch":
                    predictions[:, -1, self.pos_idx_range[0]:self.pos_idx_range[1]] = -torch.inf
                    required_token = "pos"
                elif required_token == "pos":
                    predictions[:, -1, self.patch_idx_range[0]:self.patch_idx_range[1]] = -torch.inf
                    predictions[:, -1, self.special_token_mapping["<EOS>"]] = -torch.inf  # cannot end on a patch token
                    required_token = "patch"
                
                # get the last predicted token
                if temperature > 0:
                    pass
                else:
                    last_token = predictions[:, -1:, :].argmax(dim=-1)
                # check if the last token is the <EOS> token
                # append the last token to the input tokens
                vq_tokens = torch.cat([vq_tokens, last_token], dim=1)
                if last_token.item() == self.special_token_mapping["<EOS>"]:
                    reason = "EOS token reached"
                    break
                elif text_tokens.shape[1] + vq_tokens.shape[1] + 2 >= self.max_seq_len:
                    reason = "Max sequence length reached"
                    break
        return vq_tokens, reason
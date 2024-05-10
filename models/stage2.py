from typing import List, Any, Dict, Union, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel
from x_transformers import Decoder
from torch import Tensor
from x_transformers.x_transformers import TokenEmbedding, AbsolutePositionalEmbedding
# from tokenizer import VQTokenizer
import math

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
        self.tokenizer = tokenizer

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
    
    def _top_p(self, logits, thres = 0.9, **kwargs):
        if kwargs:
            print("Unused kwargs in top-p sampling:", kwargs)
        # credit: lucidrains
        sorted_logits, sorted_indices = torch.sort(logits, descending = True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

        sorted_indices_to_remove = cum_probs > thres
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

        sorted_logits[sorted_indices_to_remove] = float('-inf')
        return sorted_logits.scatter(1, sorted_indices, sorted_logits)

    def _top_k(self, logits, frac_num_tokens = 0.1, k = None, **kwargs):
        if kwargs:
            print("Unused kwargs in top-k sampling:", kwargs)
        # credit: lucidrains
        num_tokens = logits.shape[-1]

        k = k if k is not None else math.ceil(frac_num_tokens * num_tokens)
        k = min(k, num_tokens)

        val, ind = torch.topk(logits, k)
        probs = torch.full_like(logits, float('-inf'))
        probs.scatter_(1, ind, val)
        return probs
    
    def generate(self,
                 text_tokens: Tensor,
                 attention_mask: Tensor,
                 vq_tokens: Tensor,
                 temperature:float = 0.0,
                 sampling_method:str = None,
                 sampling_kwargs:dict = {}) -> Union[Tensor, str]:
        """
        Returns the generated sequence of VQ tokens and the reason for stopping the generation.

        Args:
            - text_tokens (Tensor): The input text tokens
            - attention_mask (Tensor): The attention mask for the input text tokens
            - vq_tokens (Tensor): The input VQ tokens
            - temperature (float, optional): The temperature for the sampling. Defaults to 0.0.
            - sampling_method (str, optional): The sampling method to use. Defaults to None. Must be one of `top_p` or `top_k`.
            - sampling_kwargs (dict, optional): The sampling kwargs to use. Defaults to {}. `top_p` expects a `thres` key and `top_k` expects a `k` key (or `frac_num_tokens`).
        """
        
        assert self.pos_idx_range[0] >= self.patch_idx_range[1], "pos_idx_range must start after patch_idx_range ends"
        # assert vq_tokens.ndim == 2 and vq_tokens.size(0) == 1, "VQ_Tokens must be of shape (1, sequence_length) and contain at least the <BOS> token"

        # I'm so sorry for this code but basically this checks if all last tokens are in patch or position range. 
        # As this op is batched, it could also happen that last tokens are in patch range and some are already finished with EOS, thats why the long conditions
        if torch.logical_or((vq_tokens[:, -1] >= self.pos_idx_range[0]), (vq_tokens[:, -1] < self.patch_idx_range[0])).all() and torch.logical_or((vq_tokens[:, -1] <= self.pos_idx_range[1]), (vq_tokens[:, -1] < self.patch_idx_range[0])).all():
            required_token = "patch"
        elif torch.logical_or((vq_tokens[:, -1] >= self.patch_idx_range[0]), (vq_tokens[:, -1] < self.patch_idx_range[0])).all() and torch.logical_or((vq_tokens[:, -1] <= self.patch_idx_range[1]), (vq_tokens[:, -1] < self.patch_idx_range[0])).all():
            required_token = "pos"
        elif (vq_tokens[:, -1] < self.patch_idx_range[0]).all():  # e.g. only <BOS> tokens in input
            required_token = "patch"
        else:
            raise ValueError(f"Check if you're mixing patch and pos tokens at last position {vq_tokens[:, -1]}")

        with torch.no_grad():
            reached_end_mask = torch.logical_or(vq_tokens[:, -1:] == self.special_token_mapping["<EOS>"],
                                                vq_tokens[:, -1:] == self.special_token_mapping["<PAD>"])
            while vq_tokens.shape[1] < self.max_seq_len:
                predictions, _ = self.forward(text_tokens, attention_mask ,vq_tokens)
                logits = predictions[:, -1]

                logits[:, self.special_token_mapping["<PAD>"]] = -torch.inf  # mask the padding token
                if required_token == "patch":
                    logits[:, self.pos_idx_range[0]:self.pos_idx_range[1]] = -torch.inf
                    required_token = "pos"
                elif required_token == "pos":
                    logits[:, self.patch_idx_range[0]:self.patch_idx_range[1]] = -torch.inf
                    logits[:, self.special_token_mapping["<EOS>"]] = -torch.inf  # cannot end on a patch token
                    required_token = "patch"
                
                # sampling
                if temperature > 0:
                    if sampling_method == "top_p":
                        filtered_logits = self._top_p(logits, **sampling_kwargs)
                    elif sampling_method == "top_k":
                        filtered_logits = self._top_k(logits, **sampling_kwargs)
                    else:
                        filtered_logits = logits
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    sample = torch.multinomial(probs, 1)
                else:
                    sample = logits.argmax(dim=-1, keepdim=True)

                sample[reached_end_mask] = self.special_token_mapping["<PAD>"]
                reached_end_mask = torch.logical_or(reached_end_mask, sample == self.special_token_mapping["<EOS>"])
                vq_tokens = torch.cat([vq_tokens, sample], dim=1)
                if reached_end_mask.all():
                    reason = "EOS token reached"
                    break
                elif vq_tokens.shape[1] + 1 >= self.max_seq_len - text_tokens.shape[1]:
                    reason = "Max sequence length reached"
                    vq_tokens[~reached_end_mask.squeeze(1),-1] = self.special_token_mapping["<EOS>"]
                    break

        return vq_tokens, reason
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
                dim=self.dim,
                depth=self.depth,
                heads=self.heads,
                attn_flash=True,
                alibi_pos_bias=True,  # turns on ALiBi positional embedding
                alibi_num_heads=self.heads // 2,
                # only use ALiBi for 4 out of the 8 heads, so other 4 heads can still attend far distances
                **kwargs
            )
        else:
            self.model = Decoder(
                dim=self.dim,
                depth=self.depth,
                heads=self.heads,
                attn_flash=True,
                **kwargs
            )

    def forward(self, x: Tensor, **kwargs) -> Union[Tensor, dict]:
        batch_size, context_length, embedding_size = x.shape
        return self.model.forward(x)


class VQ_SVG_Stage2(nn.Module):
    def __init__(self,
                 tokenizer=None,  # must be of type VQTokenizer but I cannot import it here because of circular imports
                 max_seq_len: int = 512,
                 dim: int = 512,
                 depth: int = 12,
                 heads: int = 8,
                 text_encoder_str: str = "bert-base-uncased",
                 use_alibi_positional_bias=True,
                 device="cpu",
                 freeze_text_encoder=True,
                 **kwargs):
        super(VQ_SVG_Stage2, self).__init__()

        self.text_encoder_str: str = tokenizer.text_encoder_str
        self.vq_vocab_size: int = tokenizer.num_tokens
        self.special_token_mapping: dict = tokenizer.special_token_mapping
        self.patch_idx_range: Tuple[int, int] = tokenizer._get_patch_idx_range()
        self.pos_idx_range: Tuple[int, int] = tokenizer._get_pos_idx_range()
        self.device = device
        self.tokenizer = tokenizer
        self.ncode = self.tokenizer.tokens_per_patch

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.use_alibi_positional_bias = use_alibi_positional_bias

        if not self.use_alibi_positional_bias:
            self.pos_emb = AbsolutePositionalEmbedding(self.dim, max_seq_len).to(device)
        self.vq_embedding = TokenEmbedding(dim, self.vq_vocab_size).to(device)
        self.text_embedder: BertModel = BertModel.from_pretrained(text_encoder_str).to(device)

        if freeze_text_encoder:
            print("[INFO] Freezing the text encoder (BERT) weights.")
            for param in self.text_embedder.parameters():
                param.requires_grad = False

        if self.text_embedder.config.hidden_size != self.dim:
            self.mapping_layer = nn.Linear(self.text_embedder.config.hidden_size, self.dim).to(device)
        else:
            self.mapping_layer = nn.Identity().to(device)

        self.transformer = VQ_Decoder(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            use_alibi_positional_bias=self.use_alibi_positional_bias,
            **kwargs
        ).to(device)

        self.final_linear = nn.Linear(self.dim, self.vq_vocab_size).to(device)

    def loss_function(self, targets: Tensor, pred_logits: Tensor, **kwargs) -> dict:
        loss = F.cross_entropy(pred_logits, targets)
        return {'loss': loss}

    def _combine_text_and_vq(self, text_tokens: Tensor, text_attn_mask: Tensor, vq_tokens: Tensor) -> Tensor:
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
        sos_embedding = self.vq_embedding.forward(
            torch.ones(bs, 1, dtype=torch.long, device=device) * self.special_token_mapping['<SOS>']
        )  # (bs, 1, dim)

        stacked_embeddings = torch.cat([sos_embedding, text_embedding, vq_embeddings], dim=1)
        if stacked_embeddings.shape[1] > self.max_seq_len:
            print(
                f"[WARN] Input sequence length ({stacked_embeddings.shape[1]}) exceeds maximum sequence length ({self.max_seq_len}). Truncating input sequence.")
            stacked_embeddings = stacked_embeddings[:, :self.max_seq_len, :]
        if not self.use_alibi_positional_bias:
            stacked_embeddings = stacked_embeddings + self.pos_emb.forward(stacked_embeddings)

        return stacked_embeddings

    def forward(self, text_tokens: Tensor, text_attn_mask: Tensor, vq_tokens: Tensor, **kwargs) -> Tuple[Tensor, dict]:
        stacked_embeddings = self._combine_text_and_vq(text_tokens, text_attn_mask, vq_tokens)
        # TODO the attention mask should here also be used to zero out attention of the decoder to the text padding tokens
        out = self.transformer.forward(stacked_embeddings)
        out = self.final_linear.forward(out)
        out = out[:, text_tokens.shape[
                         -1] + 1:]  # remove the predictions for the text token range +1 (<SOS> token that was added during embedding)
        return out, {}

    def _top_p(self, logits, thres=0.9, **kwargs):
        if kwargs:
            print("Unused kwargs in top-p sampling:", kwargs)
        # credit: lucidrains
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cum_probs > thres
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value=False)

        sorted_logits[sorted_indices_to_remove] = float('-inf')
        return sorted_logits.scatter(1, sorted_indices, sorted_logits)

    def _top_k(self, logits, frac_num_tokens=0.1, k=None, **kwargs):
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

    def _generate_from_text(self,
                            text: str,
                            temperature: float = 0.1,
                            sampling_method: str = None,
                            sampling_kwargs: dict = {},
                            return_drawing: bool = False,
                            return_tensor: bool = False,
                            **kwargs) -> Tensor:
        assert not (return_drawing and return_tensor), "Only one of return_drawing or return_tensor can be True"
        text_tokens, attention_mask = self.tokenizer.tokenize_text(text, add_padding=True, return_attention_mask=True)
        # attention_mask = torch.ones_like(text_tokens)
        vq_tokens = torch.tensor([1], device=self.device, dtype=torch.int64)

        text_tokens = text_tokens.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        vq_tokens = vq_tokens.unsqueeze(0).to(self.device)

        generation, reason = self.generate(text_tokens, attention_mask, vq_tokens, temperature=temperature,
                                           sampling_method=sampling_method, sampling_kwargs=sampling_kwargs)
        if return_drawing:
            return self.tokenizer._tokens_to_svg_drawing(generation.to(self.device), **kwargs)
        elif return_tensor:
            return self.tokenizer._tokens_to_image_tensor(generation.to(self.device), **kwargs)
        else:
            return generation

    def generate(self,
                 text_tokens: Tensor,
                 attention_mask: Tensor,
                 vq_tokens: Tensor,
                 temperature: float = 0.1,
                 sampling_method: str = None,
                 sampling_kwargs: dict = {}) -> Union[Tensor, str]:
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

        text_tokens = text_tokens.clone().to(self.device)
        attention_mask = attention_mask.clone().to(self.device)
        vq_tokens = vq_tokens.clone().to(self.device)

        # assert vq_tokens.ndim == 2 and vq_tokens.size(0) == 1, "VQ_Tokens must be of shape (1, sequence_length) and contain at least the <BOS> token"
        required_token = None
        first_pass = True
        # I'm so sorry for this code but basically this checks if all last tokens are in patch or position range.
        # As this op is batched, it could also happen that last tokens are in patch range and some are already finished with EOS, thats why the long conditions

        # when all tokens are position tokens or special start tokens, we require a patch
        if torch.logical_or(torch.logical_and((vq_tokens[:, -1] >= self.pos_idx_range[0]),
                                              (vq_tokens[:, -1] < self.pos_idx_range[1])),
                            vq_tokens[:, -1] < self.patch_idx_range[0]).all() or (
                vq_tokens[:, -1] < self.patch_idx_range[0]).all():
            required_token = "patch"

        reached_end_mask = torch.logical_or(vq_tokens[:, -1:] == self.special_token_mapping["<EOS>"],
                                            vq_tokens[:, -1:] == self.special_token_mapping["<PAD>"])

        # when the last ncode tokens are patch tokens, we require a pos token
        if torch.logical_or(torch.logical_and((vq_tokens[:, -self.ncode:] >= self.patch_idx_range[0]),
                                              (vq_tokens[:, -self.ncode:] < self.patch_idx_range[1])),
                            reached_end_mask).all():
            required_token = "pos"

        # for now I just assume that this covers everything lol
        if required_token is None:
            required_token = "patch"
            # raise ValueError(f"Check if you're mixing patch and pos tokens {vq_tokens[:, -1]}")

        with torch.no_grad():
            reached_end_mask = torch.logical_or(vq_tokens[:, -1:] == self.special_token_mapping["<EOS>"],
                                                vq_tokens[:, -1:] == self.special_token_mapping["<PAD>"])

            if reached_end_mask.all():
                if first_pass:
                    return [], "Input was already complete"
                reason = "EOS token reached"
                return vq_tokens, reason
            elif vq_tokens.shape[1] + 1 >= self.max_seq_len - text_tokens.shape[1]:
                if first_pass:
                    return [], "Input was already complete"
                reason = "Max sequence length reached"
                # vq_tokens[~reached_end_mask.squeeze(1),-1] = self.special_token_mapping["<EOS>"]
                return vq_tokens, reason
            # make sure that if we need a patch token, we have at most ncode-1 patch tokens in a row
            patch_token_in_a_row_counter = 0
            if required_token == "patch":
                for i in range(1, self.ncode + 1):
                    if i > len(vq_tokens[0]):
                        break
                    if torch.logical_or(torch.logical_and(vq_tokens[:, -i:] >= self.patch_idx_range[0],
                                                          vq_tokens[:, -i:] < self.patch_idx_range[1]),
                                        torch.logical_or((vq_tokens[:, -i:] == self.special_token_mapping["<EOS>"]), (
                                                vq_tokens[:, -i:] == self.special_token_mapping["<PAD>"])).any(dim=1,
                                                                                                               keepdim=True)).all():
                        patch_token_in_a_row_counter = i
                assert patch_token_in_a_row_counter < self.ncode, "More than ncode patch tokens in a row"
            while vq_tokens.shape[1] < self.max_seq_len:
                if first_pass:
                    first_pass = False
                predictions, _ = self.forward(text_tokens, attention_mask, vq_tokens)
                logits = predictions[:, -1]

                logits[:, self.special_token_mapping["<PAD>"]] = -torch.inf  # mask the padding token
                if required_token == "patch":
                    logits[:, self.pos_idx_range[0]:self.pos_idx_range[1]] = -torch.inf
                    patch_token_in_a_row_counter += 1
                    if patch_token_in_a_row_counter == self.ncode:
                        required_token = "pos"
                        patch_token_in_a_row_counter = 0
                    else:
                        required_token = "patch"
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
                    vq_tokens[~reached_end_mask.squeeze(1), -1] = self.special_token_mapping["<EOS>"]
                    break
            if first_pass:
                return [], "Input already exceeded maximum sequence length"
        return vq_tokens, reason

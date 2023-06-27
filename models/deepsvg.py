import torch
from torch import nn

class _DefaultConfig:
    """
    Model config.
    """
    def __init__(self):
        self.args_dim = 256              # Coordinate numericalization, default: 256 (8-bit)
        self.n_args = 11                 # Tensor nb of arguments, default: 11 (rx,ry,phi,fA,fS,qx1,qy1,qx2,qy2,x1,x2)
        # self.n_commands = len(SVGTensor.COMMANDS_SIMPLIFIED)  # m, l, c, a, EOS, SOS, z

        self.dropout = 0.1                # Dropout rate used in basic layers and Transformers

        self.model_type = "transformer"  # "transformer" ("lstm" implementation is work in progress)

        self.encode_stages = 1           # One-stage or two-stage: 1 | 2
        self.decode_stages = 1           # One-stage or two-stage: 1 | 2

        self.use_resnet = True           # Use extra fully-connected residual blocks after Encoder

        self.use_vae = True              # Sample latent vector (with reparametrization trick) or use encodings directly

        self.pred_mode = "one_shot"      # Feed-forward (one-shot) or autogressive: "one_shot" | "autoregressive"
        # self.rel_targets = False         # Predict coordinates in relative or absolute format

        self.label_condition = False     # Make all blocks conditional on the label
        self.n_labels = 100              # Number of labels (when used)
        self.dim_label = 64              # Label embedding dimensionality

        # self.self_match = False          # Use Hungarian (self-match) or Ordered assignment

        # self.n_layers = 4                # Number of Encoder blocks
        self.n_layers_decode = 4         # Number of Decoder blocks
        self.n_heads = 8                 # Transformer config: number of heads
        self.dim_feedforward = 512       # Transformer config: FF dimensionality
        self.d_model = 256               # Transformer config: model dimensionality

        # should be done by CFG
        # self.dim_z = 256                 # Latent vector dimensionality

        # self.max_num_groups = 8          # Number of paths (N_P)
        # self.max_seq_len = 30            # Number of commands (N_C)
        # self.max_total_len = self.max_num_groups * self.max_seq_len  # Concatenated sequence length for baselines

        # self.num_groups_proposal = self.max_num_groups  # Number of predicted paths, default: N_P

    # def get_model_args(self):
    #     model_args = []

    #     model_args += ["commands_grouped", "args_grouped"] if self.encode_stages <= 1 else ["commands", "args"]

    #     if self.rel_targets:
    #         model_args += ["commands_grouped", "args_rel_grouped"] if self.decode_stages == 1 else ["commands", "args_rel"]
    #     else:
    #         model_args += ["commands_grouped", "args_grouped"] if self.decode_stages == 1 else ["commands", "args"]

    #     if self.label_condition:
    #         model_args.append("label")

    #     return model_args

class LabelEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.label_embedding = nn.Embedding(cfg.n_labels, cfg.dim_label)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.label_embedding.weight, mode="fan_in")

    def forward(self, label):
        src = self.label_embedding(label)
        return src
    
class Decoder(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(Decoder, self).__init__()

        self.cfg = cfg

        if cfg.label_condition:
            self.label_embedding = LabelEmbedding(cfg)
        dim_label = cfg.dim_label if cfg.label_condition else None

        # "one_shot"
        seq_len = cfg.max_seq_len+1 if cfg.decode_stages == 2 else cfg.max_total_len+1
        self.embedding = ConstEmbedding(cfg, seq_len)

        if cfg.model_type == "transformer":
            decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
            decoder_norm = LayerNorm(cfg.d_model)
            self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)
        else:  # "lstm"
            self.fc_hc = nn.Linear(cfg.dim_z, 2 * cfg.d_model)
            self.decoder = nn.LSTM(cfg.d_model, cfg.d_model, dropout=cfg.dropout)

        args_dim = 2 * cfg.args_dim if cfg.rel_targets else cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def _get_initial_state(self, z):
        hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.cfg.d_model, dim=2)
        hidden_cell = hidden.contiguous(), cell.contiguous()
        return hidden_cell

    def forward(self, z, commands, args, label=None, hierarch_logits=None, return_hierarch=False):
        N = z.size(2)
        l = self.label_embedding(label).unsqueeze(0) if self.cfg.label_condition else None
        if hierarch_logits is None:
            z = _pack_group_batch(z)

        if self.cfg.decode_stages == 2:
            if hierarch_logits is None:
                src = self.hierarchical_embedding(z)
                out = self.hierarchical_decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None, memory2=l)
                hierarch_logits, z = self.hierarchical_fcn(out)

            if self.cfg.label_condition: l = l.unsqueeze(0).repeat(1, z.size(1), 1, 1)

            hierarch_logits, z, l = _pack_group_batch(hierarch_logits, z, l)

            if return_hierarch:
                return _unpack_group_batch(N, hierarch_logits, z)

        if self.cfg.pred_mode == "autoregressive":
            S = commands.size(0)
            commands, args = _pack_group_batch(commands, args)

            group_mask = _get_group_mask(commands, seq_dim=0)

            src = self.embedding(commands, args, group_mask)

            if self.cfg.model_type == "transformer":
                key_padding_mask = _get_key_padding_mask(commands, seq_dim=0)
                out = self.decoder(src, z, tgt_mask=self.square_subsequent_mask[:S, :S], tgt_key_padding_mask=key_padding_mask, memory2=l)
            else:  # "lstm"
                hidden_cell = self._get_initial_state(z)  # TODO: reinject intermediate state
                out, _ = self.decoder(src, hidden_cell)

        else:  # "one_shot"
            src = self.embedding(z)
            out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None, memory2=l)

        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits) + ((hierarch_logits,) if self.cfg.decode_stages == 2 else ())

        return _unpack_group_batch(N, *out_logits)
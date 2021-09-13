import torch
from torch import nn
from copy import deepcopy
from typing import Optional 
import torch.nn.functional as F
from torch import nn, Tensor

class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        
        self.cfg = cfg
        input_dim = cfg.input_dim
        d_model = cfg.d_model
        nhead = cfg.nhead
        nhid = cfg.dim_feedforward
        nlayers = cfg.num_layers
        dropout = cfg.dropout
        self.use_position_encoder = cfg.use_position_encoder
        self.use_transformer_encoder = cfg.use_transformer_encoder
        self.use_transformer_decoder = cfg.use_transformer_decoder

        self.feature_encoder = Encoder(input_dim, d_model, cfg.feature_encoder_layers)
        if self.use_position_encoder:
            self.pos_encoder = Encoder(input_dim, d_model, cfg.position_encoder_layers)

        if self.use_transformer_encoder:
            encoder_layer = TransformerEncoderLayer(cfg)
            self.transformer_encoder = TransformerEnocder(encoder_layer, cfg)
            self._reset_parameters(self.transformer_encoder)

        if self.use_transformer_decoder:
            self.query_embed = nn.Embedding(cfg.num_points_per_trajectory, d_model)
            decoder_layer = TransformerDecoderLayer(cfg)
            self.transformer_decoder = TransformerDecoder(decoder_layer, cfg)
            self._reset_parameters(self.transformer_decoder)
        
        self.proj = nn.Conv1d(cfg.d_model, 2, kernel_size=1, bias=True)


        if cfg.get('loss_type', 'mse') == 'smooth_l1':
            self.loss_func = nn.SmoothL1Loss().cuda()
        else:
            self.loss_func = nn.MSELoss().cuda()
    
    def _reset_parameters(self, model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, data_dict):
        
        src = data_dict['waypoints_feature']
        bs = src.shape[0]
        src = src.permute(0, 2, 1) # [bs, desc_dim, num_points]
        src = self.feature_encoder(src)
        src = src.permute(2, 0, 1) # [num_points, bs, d_model]
        
        pos_embed = None
        if self.use_position_encoder:
            pos = data_dict['pos_feature']
            pos = pos.permute(0, 2, 1) # [bs, desc_dim, num_points]
            pos_embed = self.pos_encoder(pos)
            pos_embed = pos_embed.permute(2, 0, 1) # [num_points, bs, d_model]

        if self.use_transformer_encoder:
            src = self.transformer_encoder(src, pos=pos_embed)
        
        if self.use_transformer_decoder:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            tgt = torch.zeros_like(query_embed)
            hs = self.transformer_decoder(tgt, src, pos=pos_embed, query_pos=query_embed)
            hs = hs.permute(1, 2, 0) # [num_points, bs, d_model] -> [bs, d_model, num_points]
        else:
            hs = src.permute(1, 2, 0) # [num_points, bs, d_model] -> [bs, d_model, num_points]

        traj_pred = self.proj(hs)
        data_dict['waypoints_pred'] = traj_pred.permute(0, 2, 1) # [bs, num_points, 2]

        if self.training:
            self.pred = data_dict['waypoints_pred']
            self.target = data_dict['traj_ins_pixel_norm']

        return data_dict

    def get_loss(self):
        return self.loss_func(self.pred, self.target)

def MLP(channels: list, do_bn=True, drop_out=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
            if drop_out:
                layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)

class Encoder(nn.Module):
    """ Encoding waypoint features using MLPs"""
    def __init__(self, input_dim, feature_dim, layers=[256, 128]):
        super(Encoder, self).__init__()
        self.encoder = MLP([input_dim] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, pts):
        return self.encoder(pts)

class TransformerEnocder(nn.Module):

    def __init__(self, encoder_layer, cfg):
        super().__init__()
        self.layers = _get_clones(encoder_layer, cfg.num_layers)
        self.num_layers = cfg.num_layers

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, cfg):
        super().__init__()
        self.layers = _get_clones(decoder_layer, cfg.num_layers)
        self.num_layers = cfg.num_layers
        self.norm = None
        self.return_intermediate = cfg.get('return_intermediate', False)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        dim_feedforward = cfg.dim_feedforward
        nhead = cfg.nhead
        dropout = cfg.dropout
        activation = cfg.activation
        normalize_before = cfg.get('normalize_before', False)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        d_model = cfg.d_model
        nhead = cfg.nhead
        dropout = cfg.dropout
        activation = cfg.activation
        normalize_before = cfg.get('normalize_before', False)
        dim_feedforward = cfg.dim_feedforward

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

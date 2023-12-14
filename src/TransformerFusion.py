import torch
import math
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from torch import nn, Tensor


class TransNonlinear(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MultiheadAttention(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64,
                 extra_nonlinear=True):
        super(MultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        self.extra_nonlinear = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit(feature_dim, key_feature_dim))
            if extra_nonlinear:
                self.extra_nonlinear.append(TransNonlinear(feature_dim, key_feature_dim))
            else:
                self.extra_nonlinear = None

    def forward(self, query=None, key=None, value=None,
                ):
        """
        query : #pixel x batch x dim

        """
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):
                concat = self.head[N](query, key, value)
                if self.extra_nonlinear:
                    concat = self.extra_nonlinear[N](concat)
                isFirst = False
            else:
                tmp = self.head[N](query, key, value)
                if self.extra_nonlinear:
                    tmp = self.extra_nonlinear[N](tmp)
                concat = torch.cat((concat, tmp), -1)

        output = concat
        return output


class RelationUnit(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.temp = 1
        self.WK = nn.Linear(feature_dim, key_feature_dim, bias=False)
        self.WQ = nn.Linear(feature_dim, key_feature_dim, bias=False)
        self.WV = nn.Linear(feature_dim, feature_dim, bias=False)
        self.after_norm = nn.BatchNorm1d(feature_dim)
        self.trans_conv = nn.Linear(feature_dim, feature_dim, bias=False)

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None, mask=None):
        w_k = self.WK(key)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1, 2, 0) # Batch, Dim, Len_1

        w_q = self.WQ(query)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1, 0, 2) # Batch, Len_2, Dim

        dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1
        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == 0, -1e9)
        affinity = F.softmax(dot_prod * self.temp, dim=-1) 
        affinity = affinity / (1e-9 + affinity.sum(dim=1, keepdim=True))

        w_v = self.WV(value)
        w_v = w_v.permute(1,0,2) # Batch, Len_1, Dim
        output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        output = output.permute(1,0,2)

        output = self.trans_conv(query - output)

        return F.relu(output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        self.norm = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed

        self.dropout = nn.Dropout(0.1)

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, query_pos=None):
        # BxNxC -> BxCxN -> NxBxC
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        query = key = value = self.with_pos_embed(src, query_pos_embed)

        # self-attention
        # NxBxC
        src2 = self.self_attn(query=query, key=key, value=value)
        src = src + src2

        # NxBxC -> BxCxN -> NxBxC
        src = self.norm(src.permute(1, 2, 0)).permute(2, 0, 1)
        return F.relu(src)
        # return src


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN,
                 d_model=512,
                 num_encoder_layers=6,
                 activation="relu",
                 self_posembed=None):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            multihead_attn, FFN, d_model, self_posembed=self_posembed)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, query_pos=None):
        num_imgs, batch, dim = src.shape
        output = src

        for layer in self.layers:
            output = layer(output, query_pos=query_pos)

        # import pdb; pdb.set_trace()
        # [L,B,D] -> [B,D,L]
        # output_feat = output.reshape(num_imgs, batch, dim)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, key_feature_dim, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(
            feature_dim=d_model,
            n_head=1, key_feature_dim=key_feature_dim)

        self.FFN = FFN
        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, tgt, memory, query_pos=None):
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        # NxBxC

        # self-attention
        query = key = value = self.with_pos_embed(tgt, query_pos_embed)

        tgt2 = self.self_attn(query=query, key=key, value=value)
        # tgt2 = self.dropout1(tgt2)
        tgt = tgt + tgt2
        # tgt = F.relu(tgt)
        # tgt = self.instance_norm(tgt, input_shape)
        # NxBxC
        # tgt = self.norm(tgt)
        tgt = self.norm1(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = F.relu(tgt)

        mask = self.cross_attn(
            query=tgt, key=memory, value=memory)
        # mask = self.dropout2(mask)
        tgt2 = tgt + mask
        tgt2 = self.norm2(tgt2.permute(1, 2, 0)).permute(2, 0, 1)

        tgt2 = F.relu(tgt2)
        return tgt2


class TransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN,
                 d_model=512,
                 num_decoder_layers=6,
                 key_feature_dim=64,
                 self_posembed=None):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            multihead_attn, FFN, d_model, key_feature_dim, self_posembed=self_posembed)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

    def forward(self, tgt, memory, query_pos=None):
        assert tgt.dim() == 3, 'Expect 3 dimensional inputs'
        tgt_shape = tgt.shape
        num_imgs, batch, dim = tgt.shape

        output = tgt
        for layer in self.layers:
            output = layer(output, memory, query_pos=query_pos)
        return output


def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        # xyz : BxNx3
        xyz = xyz.transpose(1, 2).contiguous()
        # Bx3xN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class TransformerFusion(nn.Module):
    def __init__(self,
                 use_xyz=True,
                 input_size=2048,
                 d_model=32,
                 num_layers=1,
                 key_feature_dim=128,
                 with_pos_embed=True,
                 encoder_pos_embed_input_dim=3,
                 decoder_pos_embed_input_dim=3,
                 ):
        super(TransformerFusion, self).__init__()

        self.use_xyz = use_xyz
        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.encoder_pos_embed_input_dim = encoder_pos_embed_input_dim
        self.decoder_pos_embed_input_dim = decoder_pos_embed_input_dim
        assert encoder_pos_embed_input_dim in (3, 6)
        self.with_pos_embed = with_pos_embed

        multihead_attn = MultiheadAttention(
            feature_dim=d_model, n_head=1, key_feature_dim=key_feature_dim)

        if self.with_pos_embed:
            encoder_pos_embed = PositionEmbeddingLearned(encoder_pos_embed_input_dim, d_model)
            decoder_pos_embed = PositionEmbeddingLearned(decoder_pos_embed_input_dim, d_model)
        else:
            encoder_pos_embed = None
            decoder_pos_embed = None

        self.encoder = TransformerEncoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_encoder_layers=num_layers,
            self_posembed=encoder_pos_embed)
        self.decoder = TransformerDecoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_decoder_layers=num_layers,
            key_feature_dim=key_feature_dim,
            self_posembed=decoder_pos_embed)

    def forward(self, search_feature, search_coord,
                       template_feature, template_coord):
        """Use transformer to fuse feature.

        template_feature : (B, N, C)
        template_coord : (B, N, 3) or (B, N, 6)
        """
        # BxNxC -> NxBxC
        search_feature = search_feature.permute(1, 0, 2)
        template_feature = template_feature.permute(1, 0, 2)

        ## encoder
        encoded_memory = self.encoder(template_feature,
                                      query_pos=template_coord if self.with_pos_embed else None)

        encoded_feat = self.decoder(search_feature,
                                    memory=encoded_memory,
                                    query_pos=search_coord if self.with_pos_embed else None)  # NxBxC

        # NxBxC -> BxNxC
        encoded_feat = encoded_feat.permute(1, 0, 2)

        return encoded_feat


if __name__ == "__main__":
    net = TransformerFusion(with_pos_embed=False)
    global_f = torch.randn(6, 2048, 32)
    local_f = torch.randn(6, 2048, 32)
    pc1 = torch.randn(6, 2048, 3)
    pc2 = torch.randn(6, 2048, 3)
    
    global_f = torch.randn(6, 2048, 32)
    local_f = torch.randn(6, 38, 32)
    pc1 = torch.randn(6, 2048, 3)
    pc2 = torch.randn(6, 38, 3)

    out = net(global_f, pc1, local_f, pc2)
    print(out.size())
    
    
    
    x = torch.randn(6, 100000, 3)
    random_indice = np.random.randint(0, 100000, size=(6, 2048))
    print(random_indice.shape)
    x = x[np.arange(6)[:, None], random_indice]
    print(x.size())

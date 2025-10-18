import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from torch.nn import TransformerDecoderLayer, LayerNorm, TransformerDecoder

# from nn.positional_encoding import PositionalEncoding1D, PositionalEncodingPermute2D


import math
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        _, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, :self.channels] = emb_x

        return emb[None, :, :orig_ch]


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        _, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch]


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 3))
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        _, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        return emb[None, :, :, :, :orig_ch]


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256, w=16, h=16, n_ctx=9):
        super().__init__()
        channels = int(np.ceil(num_pos_feats / 3))
        if channels % 2:
            channels += 1
        self.channels = channels
        self.n_ctx = n_ctx
        self.row_embed = nn.Embedding(w, channels)
        self.col_embed = nn.Embedding(h, channels)
        self.hei_embed = nn.Embedding(n_ctx, channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.hei_embed.weight)

    def forward(self, tensor):
        if len(tensor.shape) == 4:
            _tensor = tensor.reshape(-1, self.n_ctx, *tensor.shape[1:])
            _, z, orig_ch, x, y = _tensor.shape
        else:
            _, z, orig_ch, x, y = tensor.shape
        i = torch.arange(x, device=tensor.device)
        j = torch.arange(y, device=tensor.device)
        h = torch.arange(z, device=tensor.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        z_emb = self.hei_embed(h)
        pos = torch.cat([
            x_emb[None, None, :, :].repeat(z, x, 1, 1),
            y_emb[None, :, None, :].repeat(z, 1, y, 1),
            z_emb[:, None, None, :].repeat(1, x, y, 1)
        ], dim=-1)[:, :, :, 0:orig_ch].permute(0, 3, 1, 2).unsqueeze(0).repeat(_, 1, 1, 1, 1)
        return pos


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

















####################################################################################################
class GlobalZ(nn.Module):
    def __init__(self, in_channel=512):
        super().__init__()
        self.query = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.query_bn = nn.GroupNorm(1, in_channel // 2)
        self.key = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.key_bn = nn.GroupNorm(1, in_channel // 2)
        self.value = nn.Conv2d(in_channel, in_channel, (1, 1))
        self.value_bn = nn.GroupNorm(1, in_channel)
        self.pos = PositionalEncoding1D(in_channel // 2)

    def forward(self, query, key, value):
        b, n, c, h, w = query.shape
        query = query.reshape(b * n, c, h, w)
        key = key.reshape(b * n, c, h, w)
        value = value.reshape(b * n, c, h, w)

        query = self.query(query)
        query = self.query_bn(query)
        key = self.key(key)
        key = self.key_bn(key)
        value = self.value(value)
        value = self.value_bn(value)

        query_pool_g = F.adaptive_avg_pool2d(query, (1, 1))
        query_pool_g = query_pool_g.view(b, n, c // 2)
        key_pool_g = F.adaptive_avg_pool2d(key, (1, 1))
        key_pool_g = key_pool_g.view(b, n, c // 2)

        pos = self.pos(key_pool_g)

        sim_slice = torch.einsum('bmd,bnd->bmn', query_pool_g + pos, key_pool_g + pos)
        sim_slice = sim_slice / (c // 2) ** 0.5
        sim_slice = torch.softmax(sim_slice, dim=-1)
        context_pool_slice = torch.einsum('bmn,bnchw->bmchw', sim_slice, value.reshape(b, n, c, h, w))

        return context_pool_slice


class GlobalS(nn.Module):
    def __init__(self, in_channel=512):
        super().__init__()
        self.query = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.query_bn = nn.GroupNorm(1, in_channel // 2)
        self.key = nn.Conv2d(in_channel, in_channel // 2, (1, 1))
        self.key_bn = nn.GroupNorm(1, in_channel // 2)
        self.value = nn.Conv2d(in_channel, in_channel, (1, 1))
        self.value_bn = nn.GroupNorm(1, in_channel)

        self.pos_emb = PositionalEncodingPermute2D(in_channel // 2)

    def forward(self, query, key, value):
        b, n, c, h, w = query.shape
        query = query.reshape(b * n, c, h, w)
        key = key.reshape(b * n, c, h, w)
        value = value.reshape(b * n, c, h, w)

        pos = self.pos_emb(query)

        query = self.query(query)
        query = self.query_bn(query) + pos
        key = self.key(key)
        key = self.key_bn(key) + pos
        value = self.value(value)
        value = self.value_bn(value)

        query_pool_g = query.reshape(b, n, c // 2, h, w)
        key_pool_g = key.reshape(b, n, c // 2, h, w)
        # value = value.reshape(b, n, c, h, w)

        query_pool_g = torch.mean(query_pool_g, 1).reshape(b, c // 2, h * w)
        key_pool_g = torch.mean(key_pool_g, 1).reshape(b, c // 2, h * w)

        sim_slice = torch.einsum('bci,bcj->bij', query_pool_g, key_pool_g)
        sim_slice = sim_slice / (c // 2) ** 0.5
        sim_slice = torch.softmax(sim_slice, dim=-1)
        context_pool_s = torch.einsum('bij,bncj->bnci', sim_slice, value.reshape(b, n, c, h * w))
        context_pool_s = context_pool_s.reshape(b, n, c, h, w)

        return context_pool_s

class Decoder(nn.Module):
    def __init__(self, num_classes, high_level_inplanes, low_level_inplanes, BatchNorm):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.Mish(inplace=True)
        self.last_conv = nn.Sequential(
            nn.Conv2d(high_level_inplanes + 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.Mish(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.Mish(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=3,
                      padding=1)
        )
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TokenLearnerModuleV2(nn.Module):
    def __init__(self, in_channel=512, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.ln = nn.GroupNorm(1, in_channel)
        self.global_z = GlobalZ(in_channel)
        self.global_s = GlobalS(in_channel)
        self.conv_3x3 = nn.Conv3d(in_channel, in_channel, (3, 3, 3), groups=8, bias=False, padding=(1, 1, 1))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, in_channel, (1, 1), bias=False),
                                   nn.GroupNorm(1, in_channel),
                                   nn.Mish(),
                                   nn.Conv2d(in_channel, in_channel, (1, 1), bias=False),
                                   nn.GroupNorm(1, in_channel),
                                   nn.Mish(),
                                   nn.Conv2d(in_channel, in_channel, (1, 1), bias=False),
                                   nn.GroupNorm(1, in_channel),
                                   nn.Mish(),
                                   nn.Conv2d(in_channel, num_tokens, (1, 1), bias=False),
                                   )
        self.conv_feat = nn.Conv2d(in_channel, in_channel, (1, 1), groups=8, bias=False)

    def forward(self, x, x_aux):
        b, n, c, h, w = x.shape
        global_z = self.global_z(x, x_aux, x_aux)
        global_s = self.global_s(x, x_aux, x_aux)
        local = self.conv_3x3(x_aux.permute(0, 2, 1, 3, 4))
        local = torch.sigmoid(local * x.permute(0, 2, 1, 3, 4)) * local
        local = local.permute(0, 2, 1, 3, 4)

        selected = x + global_z + global_s + local
        selected = self.ln(selected.reshape(b * n, c, h, w))
        feat = selected
        selected = self.conv2(selected)
        selected = torch.softmax(selected.reshape(-1, self.num_tokens, h * w), -1)

        feat = self.conv_feat(feat)
        feat = feat.reshape(-1, c, h * w)
        feat = torch.einsum("bts, bcs->btc", selected, feat)
        return feat


class TokenFuser(nn.Module):
    def __init__(self, in_channel=512, num_tokens=8):
        super().__init__()
        self.ln1 = nn.GroupNorm(1, in_channel)
        self.ln2 = nn.GroupNorm(1, in_channel)
        self.ln3 = nn.GroupNorm(1, in_channel)
        self.conv1 = nn.Linear(num_tokens, num_tokens)
        self.mix = nn.Conv2d(in_channel, num_tokens, (1, 1), groups=8, bias=False)

    def forward(self, tokens, origin):
        tokens = self.ln1(tokens)
        tokens = self.conv1(tokens)
        tokens = self.ln2(tokens)

        origin = self.ln3(origin)
        origin = self.mix(origin)
        origin = torch.sigmoid(origin)
        mix = torch.einsum("bct,bthw->bchw", tokens, origin)
        return mix


class TransformerBlock(nn.Module):
    def __init__(self, in_channel=512,
                 n_ctx=9,
                 num_tokens=8):
        super().__init__()
        self.n_ctx = n_ctx
        self.num_tokens = num_tokens
        self.in_channel = in_channel
        decoder_layer = TransformerDecoderLayer(in_channel, nhead=8, dim_feedforward=1024, dropout=0.1,
                                                batch_first=True)
        decoder_norm = LayerNorm(in_channel)
        self.TokenLearner_p = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_a = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_v = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.TokenLearner_d = TokenLearnerModuleV2(in_channel, num_tokens=num_tokens)
        self.decoder_p = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.decoder_a = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.decoder_v = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.decoder_d = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.TokenFuser = TokenFuser(in_channel, num_tokens=num_tokens)

    def forward(self, x):
        x_p, x_a, x_v, x_d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        b, n, c, h, w = x_a.shape
        token_p = self.TokenLearner_p(x_p, x_v)
        token_a = self.TokenLearner_a(x_a, x_v)
        token_v = self.TokenLearner_v(x_v, x_a)
        token_d = self.TokenLearner_d(x_d, x_a)

        token_p = token_p.reshape(b, n * self.num_tokens, c)
        token_a = token_a.reshape(b, n * self.num_tokens, c)
        token_v = token_v.reshape(b, n * self.num_tokens, c)
        token_d = token_d.reshape(b, n * self.num_tokens, c)

        token_p_tmp = self.decoder_p(token_p, torch.cat([token_p, token_a, token_v, token_d], 1))
        token_a_tmp = self.decoder_a(token_a, torch.cat([token_p, token_a, token_v, token_d], 1))
        token_v_tmp = self.decoder_v(token_v, torch.cat([token_p, token_a, token_v, token_d], 1))
        token_d_tmp = self.decoder_d(token_d, torch.cat([token_p, token_a, token_v, token_d], 1))

        token_p_tmp = token_p_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)
        token_a_tmp = token_a_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)
        token_v_tmp = token_v_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)
        token_d_tmp = token_d_tmp.reshape(-1, self.num_tokens, self.in_channel).permute(0, 2, 1)

        x_p = x_p.reshape(b * n, c, h, w)
        x_a = x_a.reshape(b * n, c, h, w)
        x_v = x_v.reshape(b * n, c, h, w)
        x_d = x_d.reshape(b * n, c, h, w)
        x_p = self.TokenFuser(token_p_tmp, x_p) + x_p
        x_a = self.TokenFuser(token_a_tmp, x_a) + x_a
        x_v = self.TokenFuser(token_v_tmp, x_v) + x_v
        x_d = self.TokenFuser(token_d_tmp, x_d) + x_d
        out = torch.stack([x_p, x_a, x_v, x_d], 1).reshape(b, -1, n, c, h, w)
        return out


class MULLET(nn.Module):
    def __init__(self, encoder_name="resnet34",
                 in_channels=3,
                 encoder_depth=5,
                 num_tokens=24,
                 cls=4,
                 n_context=3,
                 bn=nn.BatchNorm2d
                 ):
        super().__init__()
        self.n_context = n_context
        self.num_tokens = num_tokens
        self.num_classes = cls
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights="imagenet",
            output_stride=16,
        )
        self.decoder = Decoder(cls, 512, 64, bn)

        self.TransformerBlock = TransformerBlock(512, n_ctx=n_context, num_tokens=num_tokens)

    def forward(self, x):
        b, s, n, c, h, w = x.shape
        x = x.reshape(b * s * n, c, h, w)
        x_feat = self.encoder(x)
        h_feat, low_feat = x_feat[-1], x_feat[2]
        h_feat = h_feat.reshape(b, s, n, *h_feat.shape[1:])
        low_feat = low_feat.reshape(b, s, n, *low_feat.shape[1:])
        h_feat = self.TransformerBlock(h_feat)

        feat_p = h_feat[:, 0].reshape(-1, *h_feat.shape[3:])
        feat_a = h_feat[:, 1].reshape(-1, *h_feat.shape[3:])
        feat_v = h_feat[:, 2].reshape(-1, *h_feat.shape[3:])
        feat_d = h_feat[:, 3].reshape(-1, *h_feat.shape[3:])

        x_p_masks = self.decoder(feat_p, low_feat[:, 0].reshape(-1, *low_feat.shape[3:]))
        x_a_masks = self.decoder(feat_a, low_feat[:, 1].reshape(-1, *low_feat.shape[3:]))
        x_v_masks = self.decoder(feat_v, low_feat[:, 2].reshape(-1, *low_feat.shape[3:]))
        x_d_masks = self.decoder(feat_d, low_feat[:, 3].reshape(-1, *low_feat.shape[3:]))

        x_p_masks = F.interpolate(x_p_masks, size=(h, w), mode='bilinear', align_corners=True)
        x_a_masks = F.interpolate(x_a_masks, size=(h, w), mode='bilinear', align_corners=True)
        x_v_masks = F.interpolate(x_v_masks, size=(h, w), mode='bilinear', align_corners=True)
        x_d_masks = F.interpolate(x_d_masks, size=(h, w), mode='bilinear', align_corners=True)

        x_p_masks = x_p_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        x_a_masks = x_a_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        x_v_masks = x_v_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        x_d_masks = x_d_masks.reshape(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
        return x_p_masks, x_a_masks, x_v_masks, x_d_masks  # (b, c, z, h, w)


import torch
from thop import profile
from thop import clever_format

# if __name__ == "__main__":
#     # 创建模型
#     model = MULLET(
#         encoder_name="resnet34",  # 可以换成 resnet50、densenet121 等
#         in_channels=3,
#         encoder_depth=5,
#         num_tokens=24,
#         cls=4,
#         n_context=3,
#         bn=torch.nn.BatchNorm2d
#     ).cuda()
#
#     # 设置为评估模式
#     model.eval()
#
#     # 构造一个虚拟输入 (batch=1, s=1, n=4期, c=3, h=224, w=224)
#     # x = torch.randn(1, 1, 4, 3, 224, 224).cuda()
#     x = torch.randn(1, 4, 1, 3, 224, 224).cuda()
#     # shape = (b=1, s=4, n=1, c=3, h=224, w=224)
#
#     # 计算 FLOPs 和 Params
#     flops, params = profile(model, inputs=(x,))
#     flops, params = clever_format([flops, params], "%.3f")
#
#     print(f"模型参数量: {params}")
#     print(f"模型GFLOPs: {flops}")
#
#     # 前向测试输出
#     with torch.no_grad():
#         out = model(x)
#         print("输出结果 shapes:")
#         for i, o in enumerate(out):
#             print(f"Phase {i}:", o.shape)
import torch
from thop import profile, clever_format
###################################################################################
#V2
# if __name__ == "__main__":
#     model = MULLET(
#         encoder_name="resnet34",
#         in_channels=3,
#         encoder_depth=5,
#         num_tokens=24,
#         cls=4,
#         n_context=3,
#         bn=torch.nn.BatchNorm2d
#     ).cuda()
#
#     model.eval()
#
#     # 注意这里：s=4, n=1
#     x = torch.randn(1, 4, 1, 3, 224, 224).cuda()
#
#     flops, params = profile(model, inputs=(x,))
#     flops, params = clever_format([flops, params], "%.3f")
#
#     print(f"模型参数量: {params}")
#     print(f"模型GFLOPs: {flops}")
#
#     with torch.no_grad():
#         out = model(x)
#         print("输出结果 shapes:")
#         for i, o in enumerate(out):
#             print(f"Phase {i}:", o.shape)
###################################################################################

import torch
from thop import profile, clever_format
# from your_model_file import MULLET   # 确认你模型导入路径

# import torch
# import torch.nn as nn
# from thop import profile, clever_format
# # from your_model_file import MULLET   # 这里改成你模型的导入路径
#
# # ----------- 解决 LayerNorm 报错的 Hook -----------
# def layernorm_hook(m, x, y):
#     # 不统计 FLOPs，只给个 0
#     m.total_ops = torch.zeros(1).to(y.device)
#
# custom_ops = {nn.LayerNorm: layernorm_hook}
# # ------------------------------------------------
#
# if __name__ == "__main__":
#     model = MULLET(
#         encoder_name="resnet34",
#         in_channels=3,
#         encoder_depth=5,
#         num_tokens=24,
#         cls=4,
#         n_context=3,
#         bn=torch.nn.BatchNorm2d
#     ).cuda()
#
#     model.eval()
#
#     # 输入 shape: batch=1, phase=4, n=1, c=3, h=224, w=224
#     x = torch.randn(1, 4, 1, 3, 224, 224).cuda()
#     print("输入 shape:", x.shape)
#
#     # ----------- 计算参数量和 FLOPs -----------
#     flops, params = profile(model, inputs=(x,), custom_ops=custom_ops)
#     flops, params = clever_format([flops, params], "%.3f")
#
#     print(f"模型参数量: {params}")
#     print(f"模型GFLOPs: {flops}")
#     # ----------------------------------------
#
#     # ----------- 前向推理并打印输出 shape -----------
#     with torch.no_grad():
#         out = model(x)
#         if isinstance(out, (list, tuple)):
#             print("输出结果 shapes:")
#             for i, o in enumerate(out):
#                 print(f"Output[{i}] shape:", o.shape)
#         else:
#             print("输出结果 shape:", out.shape)
#####################################################################
import torch
import torch.nn as nn
# 从你的模型文件导入MULLET
# from your_model_file import MULLET

if __name__ == "__main__":
    # 初始化模型
    model = MULLET(
        encoder_name="resnet34",
        in_channels=3,
        encoder_depth=5,
        num_tokens=24,
        cls=4,
        n_context=3,
        bn=torch.nn.BatchNorm2d
    ).cuda()

    model.eval()

    # 准备输入数据
    x = torch.randn(1, 4, 1, 3, 224, 224).cuda()
    print("输入 shape:", x.shape)

    # 前向推理并打印输出形状
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (list, tuple)):
            print("输出结果 shapes:")
            for i, o in enumerate(out):
                print(f"Output[{i}] shape:", o.shape)
        else:
            print("输出结果 shape:", out.shape)

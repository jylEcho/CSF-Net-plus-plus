import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from natten.functional import na2d_qk, na2d_av
from timm.models.layers import DropPath
from torch import nn

import selective_scan_cuda_oflex


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class FFN(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            dropout=0,
    ):
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = nn.GELU()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class RoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen):
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1)  # (h w d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # (h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # (w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1)  # (h w d1)

        return sin, cos


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class MultiEpisodeFusion(nn.Module):
    def __init__(self, dim, ssm_ratio=1.0, inner_kernel_size=3, num_heads=8, use_rpb=False):
        super().__init__()
        assert inner_kernel_size % 2 == 1, "inner_kernel_size must be odd."

        self.dim = dim
        self.inner_kernel_size = inner_kernel_size
        self.num_heads = num_heads
        self.scale = (dim // self.num_heads) ** -0.5

        # local enhance mamba parameters
        inner_dim = int(dim * ssm_ratio)
        self.dwc = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GroupNorm(dim, dim),

            nn.Conv2d(dim, inner_dim, kernel_size=1),
            LayerNorm2d(inner_dim),
        )

        self.norm = LayerNorm2d(inner_dim)
        self.out_proj = nn.Conv2d(inner_dim, dim, kernel_size=1)

        self.local_en = nn.Sequential(
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GroupNorm(inner_dim, inner_dim),

            nn.Conv2d(inner_dim, inner_dim, kernel_size=5, padding=2, groups=inner_dim),
            nn.GroupNorm(inner_dim, inner_dim),
            nn.GELU(),

            nn.Conv2d(inner_dim, inner_dim, kernel_size=1),
            nn.GroupNorm(inner_dim, inner_dim),
        )

        self.mixer = SS2D(d_model=dim, expansion_ratio=ssm_ratio, k_groups=6)

        # neighbor attention parameters
        # dim of natten's query must be multiple of 4, otherwise it will raise error
        inner_dim = (dim + 3) // 4 * 4
        self.q = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, inner_dim * 2, kernel_size=1)

        self.lepe = nn.Conv2d(inner_dim, inner_dim, kernel_size=5, padding=2, groups=inner_dim)

        self.out_proj2 = nn.Conv2d(inner_dim, dim, kernel_size=1)
        self.rpb = nn.Parameter(
            torch.randn(self.num_heads, 2 * self.inner_kernel_size - 1, 2 * self.inner_kernel_size - 1)
        ) if use_rpb else None

    # def _gl_fusion(self, q, k, pos_enc):
    #     q = rearrange(self.q(q), 'b (g c) h w -> b g h w c', g=self.num_heads)
    #     kv = rearrange(self.kv(k), 'b (kv g c) h w -> kv b g h w c', kv=2, g=self.num_heads)
    #     k, v = kv[0], kv[1]
    #     del kv
    #     sin, cos = pos_enc
    #     attn = na2d_qk(theta_shift(q, sin, cos) * self.scale, theta_shift(k, sin, cos),
    #                    kernel_size=self.inner_kernel_size, rpb=self.rpb)
    #     attn = torch.softmax(attn, dim=-1)

    #     v_4d = rearrange(v, 'b g h w c -> b (g c) h w')
    #     lepe_output = self.lepe(v_4d)
    #     attn_output = rearrange(na2d_av(attn, v, self.inner_kernel_size),
    #                             'b h w g c -> b (g c) h w')
        
    #     lepe_output = F.interpolate(lepe_output, 
    #                                 size=attn_output.shape[2:4],  # 匹配高度和宽度
    #                                 mode='bilinear', 
    #                                 align_corners=False)
    #     print(f"attn_output shape: {attn_output.shape}")
    #     print(f"lepe_output shape: {lepe_output.shape}")
    #     return self.out_proj2(attn_output + lepe_output)
    def _gl_fusion(self, q, k, pos_enc):
        # 打印输入张量初始形状
        # print(f"\n=== 进入_gl_fusion方法 ===")
        # print(f"输入q的形状: {q.shape}")
        # print(f"输入k的形状: {k.shape}")
        # print(f"pos_enc中sin的形状: {pos_enc[0].shape}")
        # print(f"pos_enc中cos的形状: {pos_enc[1].shape}")
        
        # 处理q的维度重排
        q_reshaped = rearrange(self.q(q), 'b (g c) h w -> b g h w c', g=self.num_heads)
        # print(f"self.q(q)后的形状: {self.q(q).shape}")
        # print(f"q重排后的形状: {q_reshaped.shape} (b g h w c)")
        
        # 处理kv的维度重排并分离
        kv_processed = self.kv(k)
        # print(f"self.kv(k)后的形状: {kv_processed.shape}")
        kv_reshaped = rearrange(kv_processed, 'b (kv g c) h w -> kv b g h w c', kv=2, g=self.num_heads)
        # print(f"kv重排后的形状: {kv_reshaped.shape} (kv b g h w c)")
        
        k, v = kv_reshaped[0], kv_reshaped[1]
        del kv_reshaped
        # print(f"分离后k的形状: {k.shape} (b g h w c)")
        # print(f"分离后v的形状: {v.shape} (b g h w c)")
        
        # 提取位置编码
        sin, cos = pos_enc
        # print(f"使用的sin形状: {sin.shape}")
        # print(f"使用的cos形状: {cos.shape}")
        
        # 计算theta_shift后的q和k
        q_theta = theta_shift(q_reshaped, sin, cos)
        k_theta = theta_shift(k, sin, cos)
        # print(f"theta_shift后q的形状: {q_theta.shape}")
        # print(f"theta_shift后k的形状: {k_theta.shape}")
        
        # 计算注意力权重
        attn = na2d_qk(q_theta * self.scale, k_theta,
                    kernel_size=self.inner_kernel_size, rpb=self.rpb)
        # print(f"na2d_qk后attn的形状: {attn.shape}")
        
        # 注意力softmax
        attn_softmax = torch.softmax(attn, dim=-1)
        # print(f"softmax后attn的形状: {attn_softmax.shape}")
        
        # 计算注意力输出
        attn_av = na2d_av(attn_softmax, v, self.inner_kernel_size)
        # print(f"na2d_av后的形状: {attn_av.shape}")  # 应为 [16, 24, 7, 7, 32] (b g h w c)
        
        # 修正维度重排：确保通道数=24×32=768，空间维度保持(7,7)
        # 正确映射：b g h w c -> b (g c) h w
        attn_output = rearrange(attn_av, 'b g h w c -> b (g c) h w')
        # print(f"attn_output重排后的形状: {attn_output.shape} (b (g c) h w)")  # 预期 [16, 768, 7, 7]
        
        # 处理lepe输出（v是5D，需先转为4D才能输入Conv2d）
        v_4d = rearrange(v, 'b g h w c -> b (g c) h w')  # 转为 [16, 768, 7, 7]
        lepe_output = self.lepe(v_4d)
        # print(f"self.lepe(v_4d)后的形状: {lepe_output.shape}")  # 预期 [16, 768, 7, 7]
        
        # 打印相加前的形状检查
        # print(f"\n=== 相加前形状检查 ===")
        # print(f"attn_output形状: {attn_output.shape}")  # 应与lepe_output一致
        # print(f"lepe_output形状: {lepe_output.shape}")
        
        # 执行相加并通过输出投影
        result = self.out_proj2(attn_output + lepe_output)
        # print(f"相加后通过out_proj2的形状: {result.shape}")
        
        return result
        # return self.out_proj2(rearrange(na2d_av(attn, v, self.inner_kernel_size),
        #                                 'b h w g c -> b (g c) h w') + self.lepe(v))

    def forward(self, art, pv, dl, gl, pos_enc):
        # multi scales episode fusion with local enhance mamba
        local_en = self.local_en(gl)
        msef = self.mixer(F.silu(
            torch.stack((self.dwc(art), self.dwc(pv), self.dwc(dl)), dim=2), inplace=True
        ))
        x1, x2, x3 = msef[:, :, 0], msef[:, :, 1], msef[:, :, 2]
        art = art + self.out_proj(local_en + self.norm(x1))
        pv = pv + self.out_proj(local_en + self.norm(x2))
        dl = dl + self.out_proj(local_en + self.norm(x3))
        del msef, x1, x2, x3, local_en

        # neighbor cross attention fusion
        # pv and delay fusion into art
        art = self._gl_fusion(art, gl, pos_enc)

        # art and delay fusion into pv
        pv = self._gl_fusion(pv, gl, pos_enc)

        # art and delay fusion into delay
        dl = self._gl_fusion(dl, gl, pos_enc)

        return art, pv, dl


class MultiEpisodeFusionBlock(nn.Module):
    def __init__(self, dim, ssm_ratio=1.0, exp_ratio=4.0, inner_kernel_size=3, num_heads=8, use_rpb=False, drop_path=0):
        super().__init__()
        self.dim = dim

        self.cpe1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.GroupNorm(dim, dim)
        self.token_mixer = MultiEpisodeFusion(dim, ssm_ratio=ssm_ratio, inner_kernel_size=inner_kernel_size,
                                              num_heads=num_heads, use_rpb=use_rpb)
        self.cpe2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = nn.GroupNorm(dim, dim)
        self.mlp = FFN(dim, int(dim * exp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, art, pv, dl, gl, pos_enc):
        # assert art.shape = pv.shape = dl.shape = gl.shape
        art = art + self.cpe1(art)
        pv = pv + self.cpe2(art)
        dl = dl + self.cpe1(dl)

        x1, x2, x3 = self.token_mixer(self.norm1(art), self.norm1(pv), self.norm1(dl), gl, pos_enc)
        art = art + self.drop_path(x1)
        pv = pv + self.drop_path(x2)
        dl = dl + self.drop_path(x3)

        art = art + self.cpe2(art)
        pv = pv + self.cpe2(pv)
        dl = dl + self.cpe2(dl)
        art = art + self.drop_path(self.mlp(art))
        pv = pv + self.drop_path(self.mlp(pv))
        dl = dl + self.drop_path(self.mlp(dl))
        return art, pv, dl


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SS2D(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=1,
            expansion_ratio=1.0,
            dt_rank="auto",
            norm_layer=LayerNorm2d,
            dropout=0.0,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            k_groups=4,
            **kwargs,
    ):

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(expansion_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # # # out proj =======================================
        # self.out_norm = norm_layer(d_inner)

        # self.expansion_ratio = expansion_ratio
        #
        # if self.expansion_ratio != 1.0:
        #     self.proj = nn.Linear(d_model, d_inner)
        #     self.yproj = nn.Linear(d_inner, d_model)

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_groups, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_groups, merge=True)  # (K * D)

        # self.factor1 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
        # self.factor2 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=None,
                        backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    @staticmethod
    def _cross_scan(x):
        x = torch.stack((
            x.flatten(2),
            x.permute(0, 1, 3, 2, 4).contiguous().flatten(2),
            x.permute(0, 1, 3, 4, 2).contiguous().flatten(2)
        ), dim=1)
        return torch.cat((x, x.flip([-1])), dim=1).contiguous()

    @staticmethod
    def _cross_merge(x):
        b, k, c, _, h, w = x.shape
        x1, x2 = x.chunk(2, dim=1)
        x = x1.flatten(3) + x2.flatten(3).flip([-1]).contiguous()
        x1, x2, x3 = x[:, 0, ...], x[:, 1, ...], x[:, 2, ...]
        x = x1.view(b, c, 3, h, w) + x2.view(b, c, h, 3, w).permute(0, 1, 3, 2, 4).contiguous() + \
            x3.view(b, c, h, w, 3).permute(0, 1, 4, 2, 3).contiguous()
        return x

    def forward(self, x, to_dtype=False, force_fp32=False):
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, _, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W * 3

        xs = self._cross_scan(x).reshape(B, -1, L)
        x_dbl = F.conv1d(xs, self.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)

        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys = self._selective_scan(xs, dts, As, Bs,
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)

        y = self._cross_merge(ys.reshape(B, K, -1, 3, H, W))

        if to_dtype:
            y = y.to(x.dtype)

        return y


if __name__ == '__main__':
    num_heads = 1
    art = torch.randn(1, 3, 224, 224).to('cuda')
    pv = torch.randn(1, 3, 224, 224).to('cuda')
    dl = torch.randn(1, 3, 224, 224).to('cuda')
    gl = torch.randn(1, 3, 224, 224).to('cuda')
    rope = RoPE(3, num_heads)
    pos_enc = rope((art.shape[2:]))
    model = MultiEpisodeFusionBlock(3, 1, 4, 3, 1, True, 0)
    art, pv, dl = model(art, pv, dl, gl, pos_enc)

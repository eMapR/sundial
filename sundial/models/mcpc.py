import copy
import lightning as L
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUAutoregressive(nn.Module):
    """
    Original CPC autoregressive model.
    Processes z_1, ..., z_t sequentially and outputs c_t at each step.

    c_t = GRU(z_t, c_{t-1})

    Works well for 1D sequences (audio, time series).
    For spatial data, flatten the spatial grid into a sequence first.
    """
    def __init__(self, z_dim, c_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=z_dim,
            hidden_size=c_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.c_dim = c_dim

    def forward(self, z, hidden=None):
        """
        z: (B, T, z_dim) — sequence of local encodings
        Returns:
            c: (B, T, c_dim) — context at each timestep
            hidden: final hidden state for stateful inference
        """
        c, hidden = self.gru(z, hidden)    # (B, T', c_dim)
        return c[:,-1]                     # (B, c_dim) 


class SelfAttention(nn.Module):
    def __init__(self, c_dim, num_heads, dropout, num_registers, seq_len, causal):
        super().__init__()
        assert c_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = c_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(c_dim, 3 * c_dim, bias=False)
        self.out_proj = nn.Linear(c_dim, c_dim, bias=False)
        if causal:
            mask = torch.triu(torch.full((seq_len+num_registers, seq_len+num_registers), float('-inf')), diagonal=1)
            mask[:, :num_registers] = 0
            mask[:num_registers, :] = 0
            self.register_buffer('mask', mask)
        else:
            self.mask = None

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self.mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, c_dim, num_heads, mlp_ratio, dropout, num_registers, seq_len, causal):
        super().__init__()
        self.norm1 = nn.LayerNorm(c_dim)
        self.attn  = SelfAttention(c_dim, num_heads, dropout, num_registers, seq_len, causal)
        self.norm2 = nn.LayerNorm(c_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(c_dim, int(c_dim * mlp_ratio), bias=False),
            nn.GELU(approximate='tanh'),
            nn.Linear(int(c_dim * mlp_ratio), c_dim, bias=False),
        )
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.resid_drop(self.attn(self.norm1(x)))
        x = x + self.resid_drop(self.mlp(self.norm2(x)))
        return x


class TransformerAutoregressive(nn.Module):
    def __init__(self, z_dim=128, c_dim=128, num_layers=14, num_heads=8,
                 mlp_ratio=4.0, dropout=0.0, seq_len=14,
                 num_registers=1, causal=True):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_registers = num_registers
        
        self.input_proj = nn.Linear(z_dim, c_dim)
        self.pos_embed = nn.Embedding(seq_len, c_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c_dim))
        # self.registers = nn.Parameter(torch.zeros(1, num_registers, c_dim))
        
        self.layers  = nn.ModuleList([
            TransformerBlock(c_dim, num_heads, mlp_ratio, dropout, num_registers+1, seq_len, causal)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(c_dim)

        self._init_special_tokens()

    def _init_special_tokens(self):
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.registers,  std=0.02)

    def forward(self, z):
        """
        z: (B, T, z_dim)
        Returns:
            cls: (B, c_dim)          sequence-level summary
            tokens: (B, T, c_dim)    per-timestep representations (registers stripped)
        """
        B, T, _ = z.shape
        
        x = self.input_proj(z)
        pos = torch.arange(T, device=z.device)
        x += self.pos_embed(pos).unsqueeze(0)           # (B, T, c_dim)

        clss = self.cls_token.expand(B, -1, -1)         # (B, 1, c_dim)
        # regs = self.registers.expand(B, -1, -1)         # (B, R, c_dim)
        # x = torch.cat([clss, regs, x], dim=1)           # (B, 1+R+T, c_dim)
        x = torch.cat([clss, x], dim=1)                 # (B, 1+T, c_dim)

        for block in self.layers:
            x = block(x)

        x = self.norm_out(x)

        return x[:, -1:, :]          # final timestep only / strip CLS and registers


class PatchEncoder3D(nn.Module):
    """
    Encodes spatiotemporal video patches.
    Tube-based: each token covers (temporal_patch x spatial_patch^2).
    Local receptive field only — no temporal attention here.
    """
    def __init__(self, in_channels, e_dim, z_dim, spatial_patch, temporal_patch):
        super().__init__()
        self.spatial_patch = spatial_patch
        self.temporal_patch = temporal_patch

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, e_dim,
                      kernel_size=(temporal_patch, spatial_patch, spatial_patch),
                      stride=1),
            nn.GroupNorm(64, e_dim),
            nn.GELU(),
            nn.Conv3d(e_dim, e_dim, kernel_size=1),
            nn.GroupNorm(64, e_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(e_dim, z_dim)

    def forward(self, x):
        """x: (B, C, T, H, W) → z: (B, L, z_dim)"""
        feat = self.encoder(x)                  # (B, hidden, T', 1, 1)
        feat = feat.permute(0, 2, 3, 4, 1)      # (B, T', 1, 1, hidden)
        feat = torch.squeeze(feat, dim=(2,3))   # (B, T', hidden)
        z = self.proj(feat)                     # (B, T', z_dim)
        
        return z


class MCPC(L.LightningModule):
    def __init__(
        self,
        in_channels=6,
        z_dim=128,
        e_dim=128,
        c_dim=128,
        spatial_patch=3,
        temporal_patch=3,
        num_prediction_steps=3,
        temperature=0.07,
        dropout=0.0,
        ar_type='transformer',
        ar_layers=14,
        seq_len=14,
        num_registers=3,
        pretrain=True,
        ema_decay=0.99,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.pretrain = pretrain
        self.ema_decay = ema_decay
        
        self.embed = PatchEncoder3D(
            in_channels=in_channels,
            e_dim=e_dim,
            z_dim=z_dim,
            spatial_patch=spatial_patch,
            temporal_patch=temporal_patch,
        )

        if ar_type == 'gru':
            self.ar_model = GRUAutoregressive(z_dim=z_dim, c_dim=c_dim, num_layers=ar_layers, seq_len=seq_len)
        else:
            self.ar_model = TransformerAutoregressive(
                z_dim=z_dim, c_dim=c_dim, num_layers=ar_layers, num_registers=num_registers, seq_len=seq_len, dropout=dropout)

        self.target_embed = copy.deepcopy(self.embed)
        self.target_embed.eval()
        for p in self.target_embed.parameters():
            p.requires_grad = False

        self.target_ar_model = copy.deepcopy(self.ar_model)
        self.target_ar_model.eval()
        for p in self.target_ar_model.parameters():
            p.requires_grad = False

        if pretrain:
            self.prediction_head_fwd = PredictionHeads(c_dim, z_dim, num_prediction_steps)

        self.info_nce = InfoNCELoss(temperature)
    
    def forward_features(self, x):
        data = x["inpt"]
        B, C, N, T, H, W = data["cur"].shape
        cur = data["cur"].permute(0, 2, 1, 3, 4, 5).reshape(B*N, C, T, H, W)
        z_cur = self.embed(cur)
        c_cur = self.ar_model(z_cur).reshape(B, N, -1).permute(0, 2, 1)
        return c_cur
        
    def forward(self, x):
        data = x["inpt"]

        B, C, N, T, H, W = data["cur"].shape
        cur = data["cur"].permute(0, 2, 1, 3, 4, 5).reshape(B*N, C, T, H, W)
        z_cur = self.embed(cur)
        c_cur = self.ar_model(z_cur)                                         # (B, c_dim)
        c_fwd_hat = self.prediction_head_fwd(c_cur)                          # (B, N_fwd, c_dim)

        N_bwd = data["bwd"].shape[2]
        N_fwd = data["fwd"].shape[2]
        
        bwd = data["bwd"].permute(0, 2, 1, 3, 4, 5).reshape(B*N_bwd, C, T, H, W)
        fwd = data["fwd"].permute(0, 2, 1, 3, 4, 5).reshape(B*N_fwd, C, T, H, W)

        return self.forward_loss(c_fwd_hat, fwd, bwd, N_fwd, N_bwd)
        
    def forward_loss(self, c_hat, z_pos, z_neg, N_pos, N_neg, scaling=None):
        B = c_hat.shape[0]
        with torch.no_grad():
            c_pos = self.target_ar_model(self.target_embed(z_pos)).reshape(B, N_pos, -1)  # (B, N_neg, c_dim)
            c_neg = self.target_ar_model(self.target_embed(z_neg)).reshape(B, N_neg, -1)  # (B, N_neg, c_dim)

        loss = self.info_nce(c_hat, c_pos, c_neg, scaling)
        return loss

    @torch.no_grad()
    def update_ema(self):
        for online, target in zip(self.embed.parameters(), self.target_embed.parameters()):
            target.data = self.ema_decay * target.data + (1 - self.ema_decay) * online.data
        for online, target in zip(self.ar_model.parameters(), self.target_ar_model.parameters()):
            target.data = self.ema_decay * target.data + (1 - self.ema_decay) * online.data

    def training_step(self, batch, batch_idx):
        return self(batch)

    def validation_step(self, batch):
        return self(batch)

    def test_step(self, batch):
        output = self.forward_features(batch)
        return output 

    def predict_step(self, batch):
        output = self.forward_features(batch)
        return output
    

class PredictionHeads(nn.Module):
    def __init__(self, c_dim, z_dim, num_steps, nonlinear=False):
        super().__init__()
        self.num_steps = num_steps

        if nonlinear:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(c_dim, c_dim),
                    nn.GELU(),
                    nn.Linear(c_dim, z_dim),
                )
                for _ in range(num_steps)
            ])
        else:
            # Original CPC: simple linear map
            self.heads = nn.ModuleList([
                nn.Linear(c_dim, z_dim, bias=False)
                for _ in range(num_steps)
            ])

    def forward(self, c):
        c_hat = []
        for k in range(self.num_steps):
            c_hat.append(self.heads[k](c))
        return torch.concat(c_hat, dim=1)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, c_hat, c_pos, c_neg, scaling):
        B, N_hat, c_dim = c_hat.shape
        _, N_pos, _     = c_pos.shape
        _, N_neg, _     = c_neg.shape

        c_hat = F.normalize(c_hat, dim=-1)
        c_pos = F.normalize(c_pos, dim=-1)
        c_neg = F.normalize(c_neg, dim=-1)
        
        pos_sim = torch.einsum('bid,bid->bi', c_hat, c_pos) / self.temperature   # (B, N_hat)
        neg_sim = torch.einsum('bid,bjd->bij', c_hat, c_neg) / self.temperature  # (B, N_hat, N_neg)

        pos_sim = pos_sim.unsqueeze(-1)                        # (B, N_hat, 1)
        logits = torch.cat([pos_sim, neg_sim], dim=-1)         # (B, N_hat, 1+N_neg)
        logits = logits.view(B * N_hat, 1 + N_neg)

        labels = torch.zeros(B * N_hat, dtype=torch.long, device=c_hat.device)
        loss = F.cross_entropy(logits, labels)

        if scaling is not None:
            loss *= scaling

        with torch.no_grad():
            metrics = {
                "max_neg_sim": neg_sim.max().item(),
                "min_neg_sim": neg_sim.min().item(),
                "max_pos_sim": pos_sim.max().item(),
                "min_pos_sim": pos_sim.min().item(),
                "mean_neg_sim": neg_sim.mean().item(),
                "mean_pos_sim": pos_sim.mean().item(),
                "stdv_neg_sim": neg_sim.std().item(),
                "stdv_pos_sim": pos_sim.std().item(),
                "loss": loss,
            }
            for i in range(N_hat):
                metrics[f"pos_sim_{i+1}"] = pos_sim[:, i].mean().item()
            for i in range(N_hat):
                metrics[f"neg_sim_{3-i}"] = neg_sim[:, i].mean().item()

        return metrics

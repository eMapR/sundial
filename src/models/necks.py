from torch import nn


class UpscaleNeck(nn.Module):
    def __init__(self, embed_dim: int, depth: int, dropout: bool = True):
        super().__init__()

        def build_block(in_ch, out_ch): return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(0.1) if dropout else nn.Identity(),
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2))

        self.block = nn.Sequential(
            *[build_block(embed_dim, embed_dim) for _ in range(depth)]
        )

    def forward(self, x):
        return self.block(x)

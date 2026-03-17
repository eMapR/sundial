import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
      
      
class Conv3DRBM(L.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, k=1, padding=0):
        super().__init__()
        self.k = k
        self.stride = stride
        self.padding = padding
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(in_channels))
        self.h_bias = nn.Parameter(torch.zeros(out_channels))

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.conv3d(v, self.W, self.h_bias, stride=self.stride, padding=self.padding))
        return p_h, torch.bernoulli(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.conv_transpose3d(h, self.W, self.v_bias, stride=self.stride, padding=self.padding))
        return p_v, torch.bernoulli(p_v)

    def contrastive_divergence(self, v):
        p_h0, h0 = self.v_to_h(v)

        hk = h0
        for _ in range(self.k):
            p_vk, vk = self.h_to_v(hk)
            p_hk, hk = self.v_to_h(vk)

        return v, p_h0, vk, p_hk

    def forward(self, v):
        p_h, _ = self.v_to_h(v)
        return p_h

    def training_step(self, batch, batch_idx):
        v0, p_h0, vk, p_hk = self.contrastive_divergence(batch["chip"])
        loss = F.mse_loss(vk, v0)
        return {"loss": loss}
      
    def validation_step(self, batch):
        output = {"output": self(batch)}
        return output 

    def test_step(self, batch):
        output = {"output": self(batch)}
        return output 

    def predict_step(self, batch):
        output = {"output": self(batch)}
        return output


class CDBN(nn.Module):
    def __init__(self, in_channels=6, layer_configs=None):
        super().__init__()
        if layer_configs is None:
            layer_configs = [(32, 4), (64, 4)]

        self.rbm_layers = nn.ModuleList()
        c = in_channels
        for out_c, k in layer_configs:
            self.rbm_layers.append(ConvRBM(c, out_c, k))
            c = out_c

    def forward(self, x):
        for rbm in self.rbm_layers:
            x = rbm(x)
        return x
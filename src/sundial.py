import lightning as L
import torch


from transformers import VideoMAEConfig, VideoMAEModel, VideoMAEForPreTraining
from settings import SUNDIAL as configs


class Sundial(L.LightningModule):
    def __init__(self,
                 image_size: int = configs["image_size"],
                 patch_size: int = configs["patch_size"],
                 num_channels: int = configs["num_channels"],
                 num_frames: int = configs["num_frames"],
                 tubelet_size: int = configs["tubelet_size"],
                 hidden_size: int = configs["hidden_size"],
                 num_hidden_layers: int = configs["num_hidden_layers"],
                 num_attention_heads: int = configs["num_attention_heads"],
                 intermediate_size: int = configs["intermediate_size"],
                 hidden_act: str = configs["hidden_act"],
                 hidden_dropout_prob: float = configs["hidden_dropout_prob"],
                 attention_probs_dropout_prob: float = configs["attention_probs_dropout_prob"],
                 initializer_range: float = configs["initializer_range"],
                 layer_norm_eps: float = configs["layer_norm_eps"],
                 qkv_bias: bool = configs["qkv_bias"],
                 use_mean_pooling: bool = configs["use_mean_pooling"],
                 decoder_num_attention_heads: int = configs["decoder_num_attention_heads"],
                 decoder_hidden_size: int = configs["decoder_hidden_size"],
                 decoder_num_hidden_layers: int = configs["decoder_num_hidden_layers"],
                 decoder_intermediate_size: int = configs["decoder_intermediate_size"],
                 norm_pix_loss: bool = configs["norm_pix_loss"],
                 learning_rate: float = configs["learning_rate"],
                 **kwargs):
        super().__init__()
        self.config = VideoMAEConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            qkv_bias=qkv_bias,
            use_mean_pooling=use_mean_pooling,
            decoder_num_attention_heads=decoder_num_attention_heads,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_hidden_layers=decoder_num_hidden_layers,
            decoder_intermediate_size=decoder_intermediate_size,
            norm_pix_loss=norm_pix_loss,
            **kwargs
        )
        self.back_bone = VideoMAEForPreTraining(self.config)
        self.learning_rate = learning_rate

    def forward(self, inputs) -> torch.Tensor:
        return self.back_bone(inputs)

    def training_step(self, batch, *args) -> torch.Tensor:
        outputs = self.back_bone(batch)
        return outputs.loss

    def validation_step(self, batch, *args) -> torch.Tensor:
        outputs = self.back_bone(batch)
        return outputs.loss

    def predict_step(self, batch, *args) -> torch.Tensor:
        outputs = self.back_bone(batch)
        return outputs.loss

    def configure_optimizers(self):
        # TODO: implement dynamic learning rate
        return torch.optim.AdamW(lr=self.learning_rate)

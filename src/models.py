import lightning as L
import torch

from torch import nn


class UpscaleNeck(nn.Module):
    def __init__(self, embed_dim: int, depth: int):
        super().__init__()

        def build_block(in_ch, out_ch): return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2))

        self.block = nn.Sequential(
            *[build_block(embed_dim, embed_dim) for _ in range(depth)])

    def forward(self, x):
        return self.block(x)


class GradualUpscaleNeck(nn.Module):
    def __init__(self, embed_dims: int):
        super().__init__()

        def build_block(in_ch, out_ch): return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2),
            nn.ReLU())

        self.block = nn.Sequential(
            *[build_block(embed_dims[i], embed_dims[i+1])
              for i in range(len(embed_dims) - 1)])

    def forward(self, x):
        return self.block(x)


class PrithviFCN(L.LightningModule):
    def __init__(self,
                 num_classes: int,
                 view_size: int,
                 upscale_depth: int,
                 prithvi_path: str,
                 prithvi_params: dict,
                 prithvi_freeze: bool,
                 criterion: str):
        super().__init__()
        self.save_hyperparameters()

        # init hyper params for convenience (avoiding self.hparams everywhere)
        self.num_classes = num_classes
        self.view_size = view_size
        self.upscale_depth = upscale_depth
        self.prithvi_path = prithvi_path
        self.prithvi_params = prithvi_params
        self.prithvi_freeze = prithvi_freeze

        self.example_input_array = torch.rand((1,
                                               self.prithvi_params["model_args"]["in_chans"],
                                               self.prithvi_params["model_args"]["num_frames"],
                                               self.prithvi_params["model_args"]["img_size"],
                                               self.prithvi_params["model_args"]["img_size"]))

        # Initializing Prithvi Backbone per prithvi documentation
        from backbones.prithvi.Prithvi import MaskedAutoencoderViT
        self.backbone = MaskedAutoencoderViT(
            **self.prithvi_params["model_args"])
        if self.prithvi_freeze:
            self.backbone.eval()
        if self.prithvi_path is not None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(self.prithvi_path,
                                    map_location=map_location)
            del checkpoint['pos_embed']
            del checkpoint['decoder_pos_embed']
            _ = self.backbone.load_state_dict(checkpoint, strict=False)

        # Initializing upscaling neck
        embed_dim = self.prithvi_params["model_args"]["embed_dim"] * \
            self.prithvi_params["model_args"]["num_frames"]
        self.neck = UpscaleNeck(embed_dim, self.upscale_depth)

        # Initializing FCNHead
        from torchvision.models.segmentation.fcn import FCNHead
        self.head = FCNHead(embed_dim, self.num_classes)

        # Defining criterion and associated activation
        match criterion:
            case "ce":
                self.criterion = nn.CrossEntropyLoss(reduction="sum")
                self.activation = torch.nn.LogSoftmax()
            case "bce":
                self.criterion = nn.BCEWithLogitsLoss(reduction="sum")
                self.activation = torch.nn.Sigmoid()

    def forward(self, chips):
        # gathering features
        features, _, _ = self.backbone.forward_encoder(
            chips, mask_ratio=self.prithvi_params["train_params"]["mask_ratio"])

        # removeing class token and reshaping to 2D representation
        features = features[:, 1:, :]
        features = features.view(
            features.shape[0],
            -1,
            self.view_size,
            self.view_size)

        # performating segmentation
        features = self.neck(features)
        logits = self.head(features)

        return logits

    def training_step(self, batch):
        chips, annotations, _ = batch

        # forward pass
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        return {"loss": loss}

    def validation_step(self, batch):
        chips, annotations, _ = batch

        # forward pass
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        return {"loss": loss}

    def test_step(self, batch):
        chips, annotations, _ = batch

        # forward pass
        logits = self(chips)
        loss = self.criterion(logits, annotations)

        return {"loss": loss, "logits": logits}

    def predict_step(self, batch):
        chips, _ = batch

        # forward pass
        logits = self(chips)
        classes = self.activation(logits)

        return {"classes": classes}


class Prithvi(L.LightningModule):
    def __init__(self,
                 prithvi_params: dict,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.prithvi_params = prithvi_params
        from backbones.prithvi.Prithvi import MaskedAutoencoderViT
        self.model = MaskedAutoencoderViT(
            **self.prithvi_params["model_args"])

        self.example_input_array = torch.rand((1,
                                               self.prithvi_params["model_args"]["in_chans"],
                                               self.prithvi_params["model_args"]["num_frames"],
                                               self.prithvi_params["model_args"]["img_size"],
                                               self.prithvi_params["model_args"]["img_size"]))

    def forward(self, chips):
        return self.model.forward(chips, mask_ratio=self.prithvi_params["train_params"]["mask_ratio"])

    def training_step(self, batch):
        chips, _ = batch
        loss, _, _ = self(chips)
        return {"loss": loss}

    def validation_step(self, batch):
        chips, _ = batch
        loss, _, _ = self(chips)
        return {"loss": loss}

    def test_step(self, batch):
        chips, _ = batch
        loss, _, _ = self(chips)
        return {"loss": loss}

    def predict_step(self, batch):
        chips, _ = batch
        features, _, _ = self.model.forward_encoder(
            chips, mask_ratio=self.prithvi_params["train_params"]["mask_ratio"])
        return {"features": features}

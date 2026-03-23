from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.mlp(avg_values) + self.mlp(max_values)
        return x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.bn(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * torch.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class DoubleConvDS(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                mid_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownDS(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SmaAtUNet2D(nn.Module):
    """Original single-step SmaAt-UNet (kept for backward compatibility)."""

    def __init__(
        self,
        n_channels,
        n_classes,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
    ):
        super().__init__()
        self.inc = DoubleConvDS(n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2_att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3_att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4_att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5_att = self.cbam5(x5)
        x = self.up1(x5_att, x4_att)
        x = self.up2(x, x3_att)
        x = self.up3(x, x2_att)
        x = self.up4(x, x1_att)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Temporal model: SmaAt-UNet encoder + ConvLSTM bottleneck + SmaAt-UNet decoder
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x, h, c):
        i, f, g, o = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class SmaAtEncoder(nn.Module):
    """Encodes a single-channel frame, returns bottleneck + skip connections."""

    def __init__(self, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super().__init__()
        factor = 2 if bilinear else 1
        self.inc = DoubleConvDS(1, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)

    def forward(self, x):
        # x: (B, 1, H, W)
        x1 = self.inc(x)
        x1_att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2_att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3_att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4_att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5_att = self.cbam5(x5)
        return x5_att, (x1_att, x2_att, x3_att, x4_att)


class SmaAtDecoder(nn.Module):
    """Decodes bottleneck + skip connections into a single-channel frame."""

    def __init__(self, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super().__init__()
        factor = 2 if bilinear else 1
        self.up1 = UpDS(1024, 512 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, 1)

    def forward(self, bottleneck, skips):
        x1_att, x2_att, x3_att, x4_att = skips
        x = self.up1(bottleneck, x4_att)
        x = self.up2(x, x3_att)
        x = self.up3(x, x2_att)
        x = self.up4(x, x1_att)
        return self.outc(x)


class SmaAtUNetTemporal(nn.Module):
    """
    SmaAt-UNet with ConvLSTM bottleneck for multi-step nowcasting.

    Encoding phase: each of T_in input frames is encoded by the shared
    SmaAtEncoder; bottleneck features are projected to lstm_hidden_dim via
    enc_proj and fed through ConvLSTMCell to build a temporal hidden state.

    Decoding phase: autoregressive frame-by-frame generation. Starting from
    the last input frame, each step encodes the previous prediction, updates
    the ConvLSTM with the new bottleneck, and decodes the next frame using
    the fresh skip connections from that encoding. This means:
      - predictions feed back as input (autoregressive),
      - skip connections refresh every step instead of being a frozen average.
    """

    def __init__(self, T_in, T_out, kernels_per_layer=2, bilinear=True, reduction_ratio=16,
                 lstm_hidden_dim=256):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        factor = 2 if bilinear else 1
        self.bottleneck_dim = 1024 // factor  # 512 when bilinear=True
        self.lstm_hidden_dim = lstm_hidden_dim

        self.encoder = SmaAtEncoder(
            kernels_per_layer=kernels_per_layer,
            bilinear=bilinear,
            reduction_ratio=reduction_ratio,
        )
        self.enc_proj = nn.Conv2d(self.bottleneck_dim, lstm_hidden_dim, 1)
        self.convlstm = ConvLSTMCell(
            input_dim=lstm_hidden_dim,
            hidden_dim=lstm_hidden_dim,
        )
        self.dec_proj = nn.Conv2d(lstm_hidden_dim, self.bottleneck_dim, 1)
        self.decoder = SmaAtDecoder(
            kernels_per_layer=kernels_per_layer,
            bilinear=bilinear,
            reduction_ratio=reduction_ratio,
        )

    def forward(self, x):
        # x: (B, T_in, H, W)
        B, T_in, H, W = x.shape

        # --- Encoding phase: build LSTM state from input frames ---
        h, c = None, None
        for t in range(T_in):
            bottleneck, skips = self.encoder(x[:, t:t + 1])  # (B, 1, H, W)
            lstm_in = self.enc_proj(bottleneck)
            if h is None:
                h, c = self.convlstm.init_hidden(
                    B, lstm_in.shape[2], lstm_in.shape[3], x.device
                )
            h, c = self.convlstm(lstm_in, h, c)

        # --- Decoding phase: autoregressive frame-by-frame ---
        # prev_frame starts as the last input frame; each decoded frame
        # becomes the input for the next step (skip connections refresh
        # every step from the encoding of prev_frame).
        preds = []
        prev_frame = x[:, -1:]  # (B, 1, H, W)

        for _ in range(self.T_out):
            bottleneck, skips = self.encoder(prev_frame)
            lstm_in = self.enc_proj(bottleneck)
            h, c = self.convlstm(lstm_in, h, c)
            frame_pred = self.decoder(self.dec_proj(h), skips)  # (B, 1, H, W)
            preds.append(frame_pred.unsqueeze(1))               # (B, 1, 1, H, W)
            prev_frame = frame_pred

        return torch.cat(preds, dim=1)  # (B, T_out, 1, H, W)


class SmaAtNowcastBackbone(nn.Module):
    """DiffCast-compatible deterministic backbone using SmaAt-UNet with ConvLSTM bottleneck."""

    def __init__(self, in_shape, T_in, T_out, **kwargs):
        super().__init__()
        channels, _, _ = in_shape
        if channels != 1:
            raise ValueError("SmaAt backbone currently supports single-channel radar input.")
        self.pre_seq_length = T_in
        self.aft_seq_length = T_out
        self.core = SmaAtUNetTemporal(
            T_in=T_in,
            T_out=T_out,
            kernels_per_layer=kwargs.get("kernels_per_layer", 2),
            bilinear=kwargs.get("bilinear", True),
            reduction_ratio=kwargs.get("reduction_ratio", 16),
            lstm_hidden_dim=kwargs.get("lstm_hidden_dim", 256),
        )

    def predict(self, frames_in, frames_gt=None, compute_loss=False, **kwargs):
        # frames_in: (B, T_in, C, H, W) where C=1
        requested_t_out = kwargs.get("T_out")
        if requested_t_out is not None and requested_t_out != self.aft_seq_length:
            raise ValueError(
                f"SmaAt backbone was initialized for T_out={self.aft_seq_length}, "
                f"but predict(..., T_out={requested_t_out}) was requested."
            )
        x = frames_in[:, :, 0]       # (B, T_in, H, W)
        pred = self.core(x)           # (B, T_out, 1, H, W)
        if not compute_loss:
            return pred, None
        if frames_gt is None:
            raise ValueError("frames_gt is required for compute_loss=True")
        loss = F.mse_loss(pred, frames_gt)
        return pred, loss

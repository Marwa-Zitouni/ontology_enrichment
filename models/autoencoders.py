"""
Module 2
 """

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ResidualConv1dBlock(nn.Module):
  
    def __init__(self, c_in: int, c_out: int, k: int = 5, dilation: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(8, c_out), c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


# Encoder / Decoder
class TSConvEncoder(nn.Module):
    

    def __init__(self, in_channels: int = 3, seq_len: int = 50,
                 widths=(32, 64, 128), latent_dim: int = 32,
                 kernel: int = 5, dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.widths = widths
        layers = []
        c_prev = in_channels
        for w in widths:
            layers.append(ResidualConv1dBlock(c_prev, w, k=kernel, dropout=dropout))
            layers.append(nn.Conv1d(w, w, 4, stride=2, padding=1))  # /2 along T
            layers.append(nn.GELU())
            c_prev = w
        self.body = nn.Sequential(*layers)
        # Length after k stride-2 convs is ceil(seq_len / 2^k); use floor for
        # padding=1, kernel=4: length' = floor((T + 2 - 4)/2 + 1) = floor(T/2).
        out_len = seq_len
        for _ in widths:
            out_len = out_len // 2
        self.out_len = max(out_len, 1)
        self.fc = nn.Linear(widths[-1] * self.out_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        h = h.flatten(1)
        return self.fc(h)


class TSConvDecoder(nn.Module):

    def __init__(self, out_channels: int = 3, seq_len: int = 50,
                 widths=(32, 64, 128), latent_dim: int = 32,
                 bottleneck_len: int = 6, kernel: int = 5,
                 dropout: float = 0.0):
        super().__init__()
        self.widths = widths
        self.bottleneck_len = bottleneck_len
        self.fc = nn.Linear(latent_dim, widths[-1] * bottleneck_len)
        layers = []
        rev = list(reversed(widths))
        for i, w in enumerate(rev):
            c_next = rev[i + 1] if i + 1 < len(rev) else widths[0]
            layers.append(nn.ConvTranspose1d(w, c_next, 4, stride=2, padding=1))
            layers.append(nn.GELU())
            layers.append(ResidualConv1dBlock(c_next, c_next, k=kernel,
                                              dropout=dropout))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(widths[0], out_channels, 1)
        self.seq_len = seq_len

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), self.widths[-1], self.bottleneck_len)
        h = self.body(h)
        # Force exact output length to match the encoder input.
        if h.size(-1) != self.seq_len:
            h = F.interpolate(h, size=self.seq_len, mode="linear", align_corners=False)
        return self.head(h)



# autoencoder
class TimeSeriesAutoencoder(nn.Module):
    

    def __init__(self, input_dim: int = 3, seq_len: int = 50,
                 latent_dim: int = 32, widths=(32, 64, 128),
                 kernel: int = 5, dropout: float = 0.05):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder = TSConvEncoder(input_dim, seq_len, widths, latent_dim,
                                     kernel=kernel, dropout=dropout)
        self.decoder = TSConvDecoder(input_dim, seq_len, widths, latent_dim,
                                     bottleneck_len=self.encoder.out_len,
                                     kernel=kernel, dropout=dropout)

    def _to_btC(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, T, C) and convert to (B, C, T) for Conv1d.
        if x.dim() != 3:
            raise ValueError(f"expected 3-D tensor, got shape {tuple(x.shape)}")
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor):
        x_ct = self._to_btC(x)
        z = self.encoder(x_ct)
        x_recon_ct = self.decoder(z)
        x_recon = x_recon_ct.transpose(1, 2)
        return x_recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x_recon, _ = self.forward(x)
        return ((x - x_recon) ** 2).mean(dim=(1, 2))


# Backward-compatibility wrapper used by pipeline.py 
class MultimodalFusionDetector(nn.Module):
    

    def __init__(self, ts_latent_dim: int = 32, img_latent_dim: int = 32,
                 seq_len: int = 50, input_dim: int = 3,
                 widths=(32, 64, 128), kernel: int = 5, dropout: float = 0.05,
                 **_unused):
        super().__init__()
        self.seq_len = seq_len
        self.ts_autoencoder = TimeSeriesAutoencoder(
            input_dim=input_dim, seq_len=seq_len, latent_dim=ts_latent_dim,
            widths=widths, kernel=kernel, dropout=dropout,
        )
        # Kept for pipeline.py compatibility (no longer trained).
        self.img_autoencoder = None

    
    def forward(self, ts_input: torch.Tensor, img_input=None):
        ts_recon, ts_z = self.ts_autoencoder(ts_input)
        # Return placeholders for the now-removed image branch and anomaly
        # head so the existing 4-tuple unpack in pipeline.py keeps working.
        B = ts_input.size(0)
        device = ts_input.device
        img_recon_placeholder = torch.zeros(B, 1, 1, 1, device=device)
        anomaly_head_placeholder = torch.zeros(B, device=device)
        return ts_recon, img_recon_placeholder, anomaly_head_placeholder, ts_z

    def compute_anomaly_scores(self, ts_input: torch.Tensor, img_input=None,
                               alpha: float = 1.0):
        ts_err = self.ts_autoencoder.reconstruction_error(ts_input)
        zeros = torch.zeros_like(ts_err)
        return ts_err, ts_err, zeros


# Training utilities 
def train_autoencoders(model, ts_data, img_data=None, labels=None,
                       epochs: int = 30, lr: float = 1e-3,
                       batch_size: int = 32, weight_decay: float = 1e-5,
                       grad_clip: float = 1.0, verbose: bool = True):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    mse = nn.MSELoss()

    ts_tensor = torch.as_tensor(ts_data, dtype=torch.float32, device=device)
    n_samples = ts_tensor.size(0)
    if n_samples == 0:
        raise ValueError("train_autoencoders: empty training set")

    history = {"loss": [], "ts_loss": [], "img_loss": []}
    if verbose:
        print(f"Training TS-only AE on {n_samples} samples "
              f"for {epochs} epochs (batch={batch_size}, lr={lr}).")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        ts_shuf = ts_tensor[perm]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            ts_batch = ts_shuf[i:i + batch_size]

            # Use the wrapper's 4-tuple call so pipeline.py-style models keep
            # working; pull `ts_recon` out and ignore the placeholders.
            out = model(ts_batch, None) if isinstance(model, MultimodalFusionDetector) \
                else model(ts_batch)
            if isinstance(out, tuple) and len(out) == 4:
                ts_recon = out[0]
            else:
                ts_recon = out[0]

            loss = mse(ts_recon, ts_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)
        history["ts_loss"].append(avg_loss)
        history["img_loss"].append(0.0)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    return history


def get_anomaly_threshold(model, ts_data, img_data=None, percentile: int = 95):
    """Compute the anomaly threshold as a percentile of normal-data scores."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        ts_tensor = torch.as_tensor(ts_data, dtype=torch.float32, device=device)
        if isinstance(model, MultimodalFusionDetector):
            scores, _, _ = model.compute_anomaly_scores(ts_tensor, None)
        else:
            scores = model.reconstruction_error(ts_tensor)
    return float(np.percentile(scores.detach().cpu().numpy(), percentile))

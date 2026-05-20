
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 3, seq_len: int = 50,
                 latent_dim: int = 16, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True,
            dropout=dropout,
        )
        self.to_latent = nn.Linear(2 * hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        _, (h, _) = self.encoder(x)
        h_cat = torch.cat([h[0], h[1]], dim=-1)  # (B, 2H)
        z = self.to_latent(h_cat)                # (B, latent)
        h0 = self.from_latent(z).unsqueeze(0)    # (1, B, H)
        c0 = torch.zeros_like(h0)
        # Repeat the latent across time-steps as decoder input
        dec_in = h0[0].unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, H)
        out, _ = self.decoder(dec_in, (h0, c0))
        x_recon = self.head(out)
        return x_recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x_recon, _ = self.forward(x)
        return ((x - x_recon) ** 2).mean(dim=(1, 2))


def train_lstm_autoencoder(model: LSTMAutoencoder, ts_data: np.ndarray,
                           epochs: int = 30, lr: float = 1e-3,
                           batch_size: int = 32, weight_decay: float = 1e-5,
                           grad_clip: float = 1.0, verbose: bool = True) -> dict:
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    mse = nn.MSELoss()
    x = torch.as_tensor(ts_data, dtype=torch.float32, device=device)
    n = x.size(0)
    if n == 0:
        raise ValueError("train_lstm_autoencoder: empty training set")

    history: dict = {"loss": []}
    if verbose:
        print(f"Training LSTM-AE on {n} normal windows for {epochs} epochs "
              f"(batch={batch_size}, lr={lr}).")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        x_shuf = x[perm]
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            xb = x_shuf[i:i + batch_size]
            x_recon, _ = model(xb)
            loss = mse(x_recon, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    return history


def lstm_anomaly_threshold(model: LSTMAutoencoder, ts_normal: np.ndarray,
                           percentile: int = 95) -> float:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.as_tensor(ts_normal, dtype=torch.float32, device=device)
        scores = model.reconstruction_error(x).cpu().numpy()
    return float(np.percentile(scores, percentile))


def lstm_score(model: LSTMAutoencoder, ts_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-window scores, per-window reconstructions)."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.as_tensor(ts_data, dtype=torch.float32, device=device)
        x_recon, _ = model(x)
        scores = ((x - x_recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
        return scores, x_recon.cpu().numpy()

"""LSTM Autoencoder architecture."""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Encode and reconstruct multivariate sensor windows."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.input_size = input_size
        self.latent_size = latent_size

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one latent vector per input window."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                "inputs must have shape "
                f"(batch, sequence, {self.input_size})."
            )
        _, (hidden, _) = self.encoder(inputs)
        return self.to_latent(hidden[-1])

    def decode(
        self,
        latent: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """Reconstruct a full sequence from latent vectors."""
        decoder_input = latent.unsqueeze(1).expand(
            -1,
            sequence_length,
            -1,
        )
        decoded, _ = self.decoder(decoder_input)
        return self.output_layer(decoded)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        return self.decode(latent, inputs.shape[1])


def build_model(config: dict, device: str | torch.device) -> LSTMAutoencoder:
    """Build a model from the central configuration."""
    model_cfg = config["model"]
    input_size = len(config["data"]["feature_cols"])
    return LSTMAutoencoder(
        input_size=input_size,
        hidden_size=int(model_cfg["hidden_size"]),
        latent_size=int(model_cfg["latent_size"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)


def window_reconstruction_errors(
    inputs: torch.Tensor,
    reconstruction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return total and sensor-level MSE for each window."""
    squared_error = (reconstruction - inputs).pow(2)
    sensor_errors = squared_error.mean(dim=1)
    total_errors = sensor_errors.mean(dim=1)
    return total_errors, sensor_errors

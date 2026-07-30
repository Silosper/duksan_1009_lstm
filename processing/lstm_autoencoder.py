"""LSTM Autoencoder model for minute-level sensor window reconstruction."""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Reconstruct multivariate sensor windows with an encoder-decoder LSTM.

    Input and output shapes are both
    ``(batch_size, sequence_length, input_size)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        latent_size: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # 입력값 검증
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if latent_size <= 0:
            raise ValueError("latent_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        # 입력 시계열을 마지막 은닉 상태로 압축
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)

        # 잠재 벡터를 각 시점에 반복해 원래 시계열 복원
        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

        self.input_size = input_size
        self.hidden_size = hidden_size # 은닉 상태 사이즈
        self.latent_size = latent_size # 잠재 벡터 사이즈
        self.num_layers = num_layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sensor window into a latent vector."""
        self._validate_input(x)
        _, (hidden, _) = self.encoder(x)

        # 마지막 LSTM 레이어의 은닉 상태 사용
        return self.to_latent(hidden[-1])

    def decode(
        self,
        latent: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """Decode latent vectors into sensor windows."""
        if latent.ndim != 2:
            raise ValueError("latent must have shape (batch_size, latent_size).")
        if latent.shape[1] != self.latent_size:
            raise ValueError(
                f"latent feature size must be {self.latent_size}, "
                f"but received {latent.shape[1]}."
            )
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        decoder_input = latent.unsqueeze(1).repeat(1, sequence_length, 1)
        decoded, _ = self.decoder(decoder_input)

        return self.output_layer(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a reconstruction with the same shape as ``x``."""
        latent = self.encode(x)
        return self.decode(latent, sequence_length=x.shape[1])

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, sequence_length, input_size)."
            )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"input feature size must be {self.input_size}, "
                f"but received {x.shape[-1]}."
            )

# 재구성 오차 계산
def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return mean squared reconstruction error."""
    if reconstruction.shape != target.shape:
        raise ValueError(
            "reconstruction and target must have identical shapes. "
            f"Received {tuple(reconstruction.shape)} and {tuple(target.shape)}."
        )

    return nn.functional.mse_loss(reconstruction, target)

# 학습 가능한 파라미터 수 계산
def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

"""Baseline recurrent model for S&P 500 price forecasting.

Defines `RNNBaseline`, an LSTM regressor that maps a window of past
observations to a forecast. The module is deliberately agnostic to both the
window length (RNNs consume arbitrary-length sequences) and the number of input
features: the feature count is inferred on the first forward pass, or fixed
explicitly via `build`. This lets the same architecture serve the price-only
baseline and, mirrored in `rnn_augmented`, the pattern-augmented variant, so the
two stay directly comparable.
"""

from __future__ import annotations

import torch
from torch import nn


class RNNBaseline(nn.Module):
    """LSTM regressor that forecasts from a sequence of feature vectors.

    The network is an LSTM followed by a linear head applied to the final
    timestep's hidden state. Input shape is ``(batch, seq_len, num_features)``;
    ``seq_len`` may vary between calls. ``num_features`` is not required at
    construction -- it is inferred from the first input (or set via `build`) and
    then fixed for the model's lifetime.

    Attributes:
        hidden_size: Hidden-state size of the LSTM.
        num_layers: Number of stacked LSTM layers.
        dropout: Inter-layer dropout probability (ignored when
            ``num_layers == 1``).
        output_size: Number of values forecast per sequence.
        num_features: Inferred input feature count (``None`` until built).
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        """Store hyperparameters; the LSTM and head are built lazily.

        Args:
            hidden_size: Hidden-state size of the LSTM.
            num_layers: Number of stacked LSTM layers.
            dropout: Inter-layer dropout probability (used only when
                ``num_layers > 1``).
            output_size: Number of values to forecast per sequence.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.num_features: int | None = None
        self.lstm: nn.LSTM | None = None
        self.head: nn.Linear | None = None

    def build(self, num_features: int) -> RNNBaseline:
        """Instantiate the LSTM and head for a given feature count.

        Call this before constructing an optimizer if you want the parameters
        to exist eagerly; otherwise `forward` builds them on first use.

        Args:
            num_features: Number of input features per timestep.

        Returns:
            The model itself, to allow chaining.

        Raises:
            ValueError: If already built for a different feature count.
        """
        if self.lstm is not None:
            if num_features != self.num_features:
                raise ValueError(
                    f"Model already built for {self.num_features} features; "
                    f"got {num_features}."
                )
            return self

        self.num_features = num_features
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.hidden_size, self.output_size)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from a batch of feature sequences.

        Args:
            x: Input tensor of shape ``(batch, seq_len, num_features)``.

        Returns:
            Forecast tensor of shape ``(batch, output_size)``.

        Raises:
            ValueError: If `x` is not a 3-D tensor.
        """
        if x.dim() != 3:
            raise ValueError(
                "Expected input of shape (batch, seq_len, num_features); "
                f"got {tuple(x.shape)}."
            )

        if self.lstm is None:
            self.build(x.shape[-1]).to(x.device)

        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.head(last_step)


def _demo() -> None:
    """Build the model lazily and run a random batch as a shape check."""
    model = RNNBaseline()
    x = torch.randn(4, 30, 5)  # (batch, window, features)
    y = model(x)
    print(
        f"input {tuple(x.shape)} -> output {tuple(y.shape)} "
        f"(inferred {model.num_features} features)"
    )


if __name__ == "__main__":
    _demo()

"""Model + loss for transformer-based contrastive spectrum embedding."""

import torch
import torch.nn as nn
from depthcharge.encoders import FloatEncoder
from depthcharge.transformers import SpectrumTransformerEncoder


class ContrastiveLoss(nn.Module):
    """Hadsell-style contrastive loss on Euclidean distance between embeddings.

    label=1 → same peptide (pull together, squared distance penalty)
    label=0 → different peptide (push apart up to `margin`)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        output1: torch.Tensor,
        output2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(
                   torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()


class CustomSpectrumTransformerEncoder(SpectrumTransformerEncoder):
    """depthcharge SpectrumTransformerEncoder with precursor m/z + charge baked
    into the global spectrum token via a learned float encoder.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )
        # Owned by the model so .to(device) moves it.
        self.float_encoder = FloatEncoder(d_model=d_model)

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        precursor_mass = kwargs.get("pepmass", None)
        charge = kwargs.get("charge", None)
        if precursor_mass is None:
            raise ValueError("Missing precursor mass: pepmass kwarg is required.")
        if charge is None:
            raise ValueError("Missing charge: charge kwarg is required.")

        precursor_mass_encoded = self.float_encoder(precursor_mass.float())
        charge_encoded = self.float_encoder(charge.float())
        combined = precursor_mass_encoded + charge_encoded
        # (batch, 1, d_model) → (batch, d_model)
        return combined.squeeze(1)

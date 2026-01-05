"""Neural network models for spectrum encoding."""

import torch
import torch.nn as nn
from depthcharge.transformers import SpectrumTransformerEncoder
from depthcharge.encoders import FloatEncoder

from . import config


class CustomSpectrumTransformerEncoder(SpectrumTransformerEncoder):
    """Custom spectrum transformer encoder with precursor mass and charge encoding."""
    
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
        # Create float encoder as part of the model so it moves to device with model
        self.float_encoder = FloatEncoder(d_model=d_model)
        
    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Custom implementation of the global token hook.
        Encodes precursor mass and charge into the global spectrum token.

        Args:
            mz_array: M/Z values
            intensity_array: Intensity values
            **kwargs: Must contain 'pepmass' and 'charge'

        Returns:
            Combined encoding of precursor mass and charge
        """
        precursor_mass = kwargs.get("pepmass", None)
        charge = kwargs.get("charge", None)

        # Error handling for missing precursor mass or charge
        if precursor_mass is None:
            raise ValueError("Missing precursor mass: precursor_mass is required but is None.")

        if charge is None:
            raise ValueError("Missing charge: charge is required but is None.")

        # Encode precursor mass and charge using the model's float encoder
        precursor_mass_encoded = self.float_encoder(precursor_mass.float())
        charge_encoded = self.float_encoder(charge.float())

        # Combine the two encodings
        combined_encoding = precursor_mass_encoded + charge_encoded

        # Ensure it's the correct shape: (batch_size, d_model)
        combined_encoding = combined_encoding.squeeze(1)

        return combined_encoding


class ContrastiveLoss(nn.Module):
    """Contrastive loss for spectrum pair similarity learning."""
    
    def __init__(self, margin: float = None):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for dissimilar pairs (defaults to config.CONTRASTIVE_MARGIN)
        """
        super().__init__()
        self.margin = margin if margin is not None else config.CONTRASTIVE_MARGIN

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            output1: Embeddings for first spectrum in pair
            output2: Embeddings for second spectrum in pair
            label: 1 if similar, 0 if dissimilar
            
        Returns:
            Mean contrastive loss
        """
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()

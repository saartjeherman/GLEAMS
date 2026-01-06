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
        dropout: float = 0.0,
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
    """
    Contrastive loss for spectrum pair similarity learning.
    
    Adapted from the original GLEAMS CNN implementation (Hadsell et al. 2006).
    Uses a ramp function for positive pairs to cap distances at the margin,
    which forces the model to keep similar pairs within the margin boundary.
    
    Key difference from standard contrastive loss:
    - Positive pairs: loss = min(distance, margin)² (ramp function)
    - Negative pairs: loss = max(0, margin - distance)² (standard)
    """
    
    def __init__(self, margin: float = None, label_certainty: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for dissimilar pairs (defaults to config.CONTRASTIVE_MARGIN)
            label_certainty: Confidence in labels [0-1]. Helps handle noisy labels.
                           1.0 = fully certain (default), <1.0 = uncertain labels
        """
        super().__init__()
        self.margin = margin if margin is not None else config.CONTRASTIVE_MARGIN
        self.label_certainty = label_certainty

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss using the GLEAMS ramp function approach.
        
        Args:
            output1: Embeddings for first spectrum in pair [batch_size, embedding_dim]
            output2: Embeddings for second spectrum in pair [batch_size, embedding_dim]
            label: 1 if similar, 0 if dissimilar [batch_size]
            
        Returns:
            Mean contrastive loss
        """
        # Compute Euclidean distance on RAW embeddings (no normalization)
        # The original GLEAMS CNN does NOT normalize embeddings before computing distance
        # Normalization forces embeddings onto a unit sphere, causing clustering issues
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        
        # Positive pairs: use ramp function to cap at margin
        # This forces similar pairs to have distance < margin
        margin_tensor = torch.tensor(self.margin, device=euclidean_distance.device)
        ramp_square = torch.pow(torch.minimum(euclidean_distance, margin_tensor), 2)
        
        # Negative pairs: standard margin-based penalty
        margin_square = torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        # Combine with label certainty weighting
        # For positive pairs (label=1): loss = label_certainty * min(distance, margin)²
        # For negative pairs (label=0): loss = (1 - 0) * max(0, margin - distance)² = full penalty
        loss = (label * self.label_certainty * ramp_square + 
                (1 - label * self.label_certainty) * margin_square)
        
        # Debug: Print first batch statistics (only once to avoid spam)
        if not hasattr(self, '_debug_printed'):
            pos_mask = label == 1
            neg_mask = label == 0
            if pos_mask.any() and neg_mask.any():
                print(f"\n🔍 Loss function debug (first batch):")
                print(f"   Margin: {self.margin}, Label certainty: {self.label_certainty}")
                print(f"   Positive distances: mean={euclidean_distance[pos_mask].mean():.4f}, "
                      f"loss={loss[pos_mask].mean():.4f}")
                print(f"   Negative distances: mean={euclidean_distance[neg_mask].mean():.4f}, "
                      f"loss={loss[neg_mask].mean():.4f}")
                print(f"   Positive weight: {self.label_certainty}")
                print(f"   Negative weight: 1.0 (full penalty)\n")
            self._debug_printed = True
        
        return loss.mean()

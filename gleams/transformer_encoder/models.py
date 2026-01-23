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
    
    Uses normalized embeddings with cosine distance to prevent gradient explosion
    and ensure stable training. Distances are bounded [0, 2] which prevents
    unlimited positive weight issues.
    
    Loss formulation:
    - Positive pairs: minimize cosine distance²
    - Negative pairs: penalize if cosine distance < margin
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
        Compute contrastive loss with normalized embeddings.
        
        Args:
            output1: Embeddings for first spectrum in pair [batch_size, embedding_dim]
            output2: Embeddings for second spectrum in pair [batch_size, embedding_dim]
            label: 1 if similar, 0 if dissimilar [batch_size]
            
        Returns:
            Mean contrastive loss
        """
        # Normalize embeddings to unit sphere (L2 normalization)
        output1_norm = torch.nn.functional.normalize(output1, p=2, dim=1)
        output2_norm = torch.nn.functional.normalize(output2, p=2, dim=1)
        
        # Compute cosine similarity
        cosine_similarity = (output1_norm * output2_norm).sum(dim=1)
        
        # Convert to cosine distance: distance = 1 - similarity
        # This bounds distances to [0, 2] (prevents explosion)
        cosine_distance = 1 - cosine_similarity
        
        # Contrastive loss with bounded distances
        # Positive pairs: minimize cosine distance²
        pos_loss = label * torch.pow(cosine_distance, 2)
        
        # Negative pairs: penalize if distance < margin
        neg_loss = (1 - label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2)
        
        # Combine losses
        loss = pos_loss + neg_loss
        
        # Debug: Print first batch statistics (only once to avoid spam)
        if not hasattr(self, '_debug_printed'):
            pos_mask = label == 1
            neg_mask = label == 0
            if pos_mask.any() and neg_mask.any():
                print(f"\n🔍 Loss function debug (first batch):")
                print(f"   Margin: {self.margin}, Label certainty: {self.label_certainty}")
                print(f"   Positive cosine distances: mean={cosine_distance[pos_mask].mean():.4f}, "
                      f"loss={pos_loss[pos_mask].mean():.4f}")
                print(f"   Negative cosine distances: mean={cosine_distance[neg_mask].mean():.4f}, "
                      f"loss={neg_loss[neg_mask].mean():.4f}")
                print(f"   Using NORMALIZED embeddings with cosine distance")
                print(f"   Distance range: [0, 2] (bounded, prevents explosion)")
                print(f"   Positive weight: distance² (bounded by normalization)")
                print(f"   Negative weight: clamp(margin - distance)²\n")
            self._debug_printed = True
        
        return loss.mean()
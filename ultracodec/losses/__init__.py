"""Loss functions and discriminators for UltraCodec."""
from .discriminator import (
    CombinedDiscriminator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    PeriodDiscriminator,
    ScaleDiscriminator,
)
from .losses import (
    AdversarialLoss,
    AFRRegularizationLoss,
    CodebookLoss,
    CommitmentLoss,
    FeatureMatchingLoss,
    MelReconstructionLoss,
    MultiResolutionSTFTLoss,
    TimeDomainLoss,
    UltraCodecLoss,
)

__all__ = [
    # Losses
    "AdversarialLoss",
    "AFRRegularizationLoss",
    "CodebookLoss",
    "CommitmentLoss",
    "FeatureMatchingLoss",
    "MelReconstructionLoss",
    "MultiResolutionSTFTLoss",
    "TimeDomainLoss",
    "UltraCodecLoss",
    # Discriminators
    "CombinedDiscriminator",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "PeriodDiscriminator",
    "ScaleDiscriminator",
]

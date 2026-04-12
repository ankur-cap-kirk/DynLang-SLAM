"""Language pipeline for DynLang-SLAM.

Provides CLIP feature extraction, SAM mask generation, and pixel-aligned
feature map construction following the LangSplat pipeline.
"""

from .clip_extractor import CLIPExtractor
from .sam_extractor import SAMExtractor
from .feature_map import FeatureMapBuilder
from .autoencoder import LanguageAutoencoder

__all__ = ["CLIPExtractor", "SAMExtractor", "FeatureMapBuilder", "LanguageAutoencoder"]

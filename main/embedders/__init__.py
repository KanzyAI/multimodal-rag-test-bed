"""
Embedder Framework

This package provides a generalized interface for working with different embedders,
allowing easy switching between text and multimodal embedding models.
"""

from .base_embedder import (
    BaseEmbedder,
    EmbedderConfig,
    EmbedderFactory
)

# Import text embedders to register them
from .text import *

# Import multimodal embedders to register them  
from .multimodal import *

__all__ = [
    'BaseEmbedder',
    'EmbedderConfig',
    'EmbedderFactory'
] 
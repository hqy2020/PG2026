#
# X-Field: A Physically Informed Representation for 3D X-ray Reconstruction
# NeurIPS 2025 Spotlight
# https://github.com/Brack-Wang/X-Field
#

from .config import XFieldConfig
from .model import XFieldGaussianModel
from .renderer import render_xfield
from .trainer import training_xfield

__all__ = [
    'XFieldConfig',
    'XFieldGaussianModel',
    'render_xfield',
    'training_xfield',
]

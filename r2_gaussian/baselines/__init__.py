#
# Baseline methods for CT reconstruction comparison
#
# Supported methods:
#   - xgaussian: X-Gaussian (3DGS-based)
#   - corgs: CoR-GS (3DGS-based, ECCV 2024)
#   - dngaussian: DNGaussian (3DGS-based, CVPR 2024)
#   - fsgs: FSGS (3DGS-based, ECCV 2024)
#   - xfield: X-Field (3DGS-based, NeurIPS 2025 Spotlight)
#   - naf: Neural Attenuation Fields (NeRF-based)
#   - tensorf: TensoRF (NeRF-based)
#   - saxnerf: SAX-NeRF with Lineformer (NeRF-based)
#

from .registry import METHOD_REGISTRY, get_method_config

__all__ = ['METHOD_REGISTRY', 'get_method_config']

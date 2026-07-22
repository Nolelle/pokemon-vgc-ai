"""Candidate-aware reinforcement learning for complete legal VGC joint orders.

The package is intentionally separate from the default ladder policy. Importing
``vgc`` does not import torch, and nothing here is enabled by ``PolicyConfig`` yet.
"""

from vgc.rl.encoding import CandidateFeatures, encode_candidates, encode_live_state

__all__ = ["CandidateFeatures", "encode_candidates", "encode_live_state"]

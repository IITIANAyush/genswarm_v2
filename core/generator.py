# Dedicated geometry prompt — separate from the general WRITE_FUNCTION_PROMPT
GEOMETRY_PROMPT = """\
Write a Python function that generates n target positions arranged in the requested shape.

Function signature (EXACTLY):
def compute_formation_points(n, center=None):

Arguments:
  n      : int   — number of points to return
  center : np.ndarray [x,y] or None (default to np.zeros(2) when None)

Returns: np.ndarray shape (n, 2)  — ALL n points, UNIFORMLY SPACED, NO CLUSTERS

Shape to generate:
{shape_description}

Exact geometry to use:
{shape_math}

CRITICAL SPACING REQUIREMENT:
Points must be EQUALLY SPACED along the shape outline, NOT clustered.
- All pairwise distances should be approximately equal
- No close pairs (clustering) allowed except by geometric necessity (e.g., star points)
- Use consistent arc-length sampling: divide perimeter into n equal segments
- Spacing uniformity will be validated with coefficient of variation < 0.4

Rules:
- import numpy as np at top (use np.pi, np.cos, np.sin etc.)
- Centre all points at `center` (add center to every point)
- n can be any integer 1..100 — do NOT hardcode for a specific n
- Arc-length parametrisation: divide total perimeter into n EQUAL segments
- Sample one point per segment at regular intervals
- Return np.ndarray(n, 2) — shape must be EXACTLY (n, 2)
- NO markdown, NO prose — ONLY valid Python starting with `def compute_formation_points(`
"""

"""idyn — identifiability of modular nonlinear latent dynamics.

See CLAUDE.md for the mathematical setting and the list of known problems with
the original conjecture (§3). Nothing in this package assumes the original
conjecture; the corrected statements live in theory/identifiability.md.
"""

from idyn import behavior, cocycle, linear, metrics, normalform, spectra, systems

__all__ = ["systems", "linear", "spectra", "cocycle", "metrics", "normalform", "behavior"]
__version__ = "0.1.0"

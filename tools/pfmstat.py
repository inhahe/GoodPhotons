"""One-line health summary of a PFM, for tools/modesweep.sh.

Flags only what is genuinely wrong. NEGATIVE reports a pixel whose channels are ALL below zero:
single-channel negatives are expected in scene-linear spectral output (a saturated firefly
converts out of the sRGB gamut) and flagging those would cry wolf on every stochastic mode.
"""
import sys, numpy as np
from pfm import read_pfm  # noqa: E402
a=read_pfm(sys.argv[1], top_down=True)
fin=np.isfinite(a).all(); mn=float(np.nanmin(a)); mx=float(np.nanmax(a)); mean=float(np.nanmean(a))
flags=[]
if not fin: flags.append('NON-FINITE')
if mean<=0.0: flags.append('ALL-BLACK')
if (a < 0).all(axis=-1).any(): flags.append('ALL-NEGATIVE')   # transport bug; per-channel is gamut
print('mean=%-10.4g max=%-10.4g %s' % (mean, mx, ' '.join(flags) if flags else 'ok'))

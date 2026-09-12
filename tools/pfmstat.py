"""One-line health summary of a PFM, for tools/modesweep.sh.

Flags only what is genuinely wrong. NEGATIVE reports a pixel whose channels are ALL below zero:
single-channel negatives are expected in scene-linear spectral output (a saturated firefly
converts out of the sRGB gamut) and flagging those would cry wolf on every stochastic mode.
"""
import sys, numpy as np
def readpfm(p):
    with open(p,'rb') as f:
        hdr=f.readline().strip(); nc=3 if hdr==b'PF' else 1
        line=f.readline()
        while line.startswith(b'#'): line=f.readline()
        w,h=map(int,line.split()); sc=float(f.readline())
        return np.frombuffer(f.read(w*h*nc*4),dtype='<f4' if sc<0 else '>f4').reshape(h,w,nc)[::-1].astype(np.float64)
a=readpfm(sys.argv[1])
fin=np.isfinite(a).all(); mn=float(np.nanmin(a)); mx=float(np.nanmax(a)); mean=float(np.nanmean(a))
flags=[]
if not fin: flags.append('NON-FINITE')
if mean<=0.0: flags.append('ALL-BLACK')
if (a < 0).all(axis=-1).any(): flags.append('ALL-NEGATIVE')   # transport bug; per-channel is gamut
print('mean=%-10.4g max=%-10.4g %s' % (mean, mx, ' '.join(flags) if flags else 'ok'))

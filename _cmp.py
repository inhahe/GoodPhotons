import sys, struct, statistics, math
def read_ppm(p):
    with open(p,'rb') as f:
        assert f.readline().strip()==b'P6'
        w,h=map(int,f.readline().split())
        mx=int(f.readline()); data=f.read(w*h*3)
    return w,h,data
def inv_srgb(u): # u in [0,1] -> linear
    return u/12.92 if u<=0.04045 else ((u+0.055)/1.055)**2.4
def lin_lum(data,i,expo):
    r=inv_srgb(data[3*i]/255)/expo
    g=inv_srgb(data[3*i+1]/255)/expo
    b=inv_srgb(data[3*i+2]/255)/expo
    return 0.2126*r+0.7152*g+0.0722*b
# (path, exposure)
gpu=('_gd_gpu.ppm',2.1e-13); cpu=('_gd_cpu.ppm',2.03e-13); ref=('_gd_ref.ppm',1.96e-13)
def load(t):
    w,h,d=read_ppm(t[0]); return w,h,d,t[1]
wg,hg,dg,eg=load(gpu); wc,hc,dc,ec=load(cpu); wr,hr,dr,er=load(ref)
assert (wg,hg)==(wc,hc)==(wr,hr)
N=wg*hg
def ratios(dA,eA,dB,eB):
    rs=[]
    for i in range(N):
        la=lin_lum(dA,i,eA); lb=lin_lum(dB,i,eB)
        # skip near-black and near-saturated (any channel==255)
        if lb<1e-6: continue
        if max(dA[3*i:3*i+3])>=255 or max(dB[3*i:3*i+3])>=255: continue
        rs.append(la/lb)
    return rs
def report(name,rs):
    rs.sort(); n=len(rs)
    med=rs[n//2]; p25=rs[n//4]; p75=rs[3*n//4]
    mean=sum(rs)/n
    print(f"{name}: n={n} median={med:.4f} mean={mean:.4f} IQR=[{p25:.4f},{p75:.4f}]")
report("GPU-D / CPU-D", ratios(dg,eg,dc,ec))
report("CPU-D / R-ref", ratios(dc,ec,dr,er))
report("GPU-D / R-ref", ratios(dg,eg,dr,er))

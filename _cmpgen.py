import sys, statistics
def read_ppm(p):
    with open(p,'rb') as f:
        assert f.readline().strip()==b'P6'
        w,h=map(int,f.readline().split())
        int(f.readline()); data=f.read(w*h*3)
    return w,h,data
def inv_srgb(u):
    return u/12.92 if u<=0.04045 else ((u+0.055)/1.055)**2.4
def lum(d,i,e):
    r=inv_srgb(d[3*i]/255)/e; g=inv_srgb(d[3*i+1]/255)/e; b=inv_srgb(d[3*i+2]/255)/e
    return 0.2126*r+0.7152*g+0.0722*b
# args: fileA expoA fileB expoB
fa,ea,fb,eb=sys.argv[1],float(sys.argv[2]),sys.argv[3],float(sys.argv[4])
wa,ha,da=read_ppm(fa); wb,hb,db=read_ppm(fb)
assert (wa,ha)==(wb,hb)
N=wa*ha; rs=[]
for i in range(N):
    lb=lum(db,i,eb); la=lum(da,i,ea)
    if lb<1e-6: continue
    if max(da[3*i:3*i+3])>=255 or max(db[3*i:3*i+3])>=255: continue
    rs.append(la/lb)
rs.sort()
n=len(rs)
print(f"{fa} vs {fb}: N={n} median={statistics.median(rs):.4f} "
      f"mean={statistics.fmean(rs):.4f} IQR=[{rs[n//4]:.3f},{rs[3*n//4]:.3f}]")

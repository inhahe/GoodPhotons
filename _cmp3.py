def read_ppm(p):
    with open(p,'rb') as f:
        assert f.readline().strip()==b'P6'
        w,h=map(int,f.readline().split()); int(f.readline()); d=f.read(w*h*3)
    return w,h,d
def inv(u): return u/12.92 if u<=0.04045 else ((u+0.055)/1.055)**2.4
def lum(d,i,e):
    return (0.2126*inv(d[3*i]/255)+0.7152*inv(d[3*i+1]/255)+0.0722*inv(d[3*i+2]/255))/e
def cmp(a,ea,b,eb):
    wa,ha,da=read_ppm(a); wb,hb,db=read_ppm(b); N=wa*ha; rs=[]
    for i in range(N):
        lb=lum(db,i,eb)
        if lb<1e-6: continue
        if max(da[3*i:3*i+3])>=255 or max(db[3*i:3*i+3])>=255: continue
        rs.append(lum(da,i,ea)/lb)
    rs.sort(); n=len(rs)
    print(f"{a} / {b}: n={n} median={rs[n//2]:.4f} mean={sum(rs)/n:.4f} IQR=[{rs[n//4]:.4f},{rs[3*n//4]:.4f}]")
cmp('_fm_g.ppm',3.6e-13,'_fm_c.ppm',3.81e-13)

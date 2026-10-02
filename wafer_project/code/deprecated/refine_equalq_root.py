import importlib.util, os, json, numpy as np
src='/mnt/data/wafer_pitch_equalQ_report/equal_rack_flow_navier_stokes.py'
spec=importlib.util.spec_from_file_location('ns', src)
ns=importlib.util.module_from_spec(spec); spec.loader.exec_module(ns)
ns.NX=241; ns.NY=109; ns.DT=0.0010
Q=0.00554627456
cases=[('Small',0.040,0.5605),('Medium',0.060,0.3000),('Large',0.080,0.1833)]
out={}
for name,pitch,uin in cases:
    c=ns.setup(pitch)
    init=None
    history=[]
    for it,steps in enumerate([900,500,500,500]):
        u,v,p,spd,qr,res,n=ns.solve(c,uin,max_steps=steps,init=init)
        history.append([uin,qr,res,n])
        err=(qr-Q)/Q
        print(name,it,'uin',uin,'q',qr,'err%',100*err,'res',res,'n',n, flush=True)
        init=(u,v)
        if abs(err)<0.001:
            break
        # multiplicative control update; not profile fitting, just enforcing integral constraint
        fac=Q/qr
        # damp large corrections for stability
        fac=max(0.75,min(1.25,fac))
        uin*=fac
    # one final longer solve at the corrected uin if not within tolerance
    if abs((qr-Q)/Q)>=0.001:
        u,v,p,spd,qr,res,n=ns.solve(c,uin,max_steps=700,init=init)
        history.append([uin,qr,res,n])
        print(name,'final','uin',uin,'q',qr,'err%',100*(qr-Q)/Q,'res',res,'n',n, flush=True)
    out[name]={
      'pitch':pitch,'uin':uin,'q':qr,'error_pct':100*(qr-Q)/Q,'res':res,'steps':n,
      'x':c['x'].tolist(),'y':c['y'].tolist(),'iy':int(c['iy']),
      'wc':c['wc'].tolist(),'solid_cut':c['solid'][c['iy'],:].astype(int).tolist(),
      'gaps':c['gaps'].astype(int).tolist(),'vcut':v[c['iy'],:].tolist(),
      'history':history,
    }
with open('/mnt/data/refined_equalq_root_results.json','w') as f: json.dump(out,f)

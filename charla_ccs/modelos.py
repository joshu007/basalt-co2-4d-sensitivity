"""Modelos educativos: arenisca continua y almacenamiento en basalto.
Ejecutar: python modelos.py --out resultados --frequency 30 --noise 0.002
Unidades internas SI salvo modulos GPa. Arrays (profundidad, distancia).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

NAMES=['Cobertura','Sello sedimentario','Arenisca','Unidad inferior',
       'Basalto masivo','Basalto vesicular','Corredor fracturado']
COLORS=['#dfc89c','#7c9170','#efb744','#81716d','#414856','#54b4ab','#b16bac']
# phi, K mineral, K seco, mu seco, densidad mineral.
# Parametros impuestos, NO calibracion de Serra Geral; toda phi es conectada.
PROPS=np.array([[.15,36,13,10,2650],[.06,32,19,12,2680],
 [.23,37,8,7,2650],[.04,45,29,20,2750],
 [.012,75,65,34,2950],[.12,75,24,15,2900],[.045,75,21,11,2920]])

def geology(kind,x,z):
    X,Z=np.meshgrid(x,z)
    fold=-65*np.exp(-((X-2100)/950)**2)
    f=np.zeros(X.shape,dtype=int)
    if kind=='arenisca':
        f[Z>350+fold*.4]=1
        f[Z>900+fold]=2
        f[Z>1120+fold]=3
        S=np.where(f==2,.6*np.exp(-((X-2100)/650)**2)*
                   np.exp(-((Z-(980+fold))/95)**2),0)
    else:
        # Desplazamiento a lo largo de una discontinuidad inclinada.
        fault=2700-.28*(Z-800)
        u=Z-fold-np.where(X>fault,95,0)
        f[u>350]=4
        f[u>1400]=3
        # Lentes entre derrames, pinzamientos laterales y relieve irregular.
        for j,center in enumerate([590,785,1010,1220]):
            roof=center+28*np.sin(X/260+j)+16*np.sin(X/115+2*j)
            thick=12+45*np.maximum(0,np.cos(X/380+j))
            extent=(np.sin(X/510+j)>.0)
            lens=(u>roof)&(u<roof+thick)&extent&(f==4)
            f[lens]=5
        # Corredores son zonas efectivas de decenas de metros, no grietas resueltas.
        c1=np.abs(X-(1650+.34*(Z-950)))<38
        c2=np.abs(X-fault)<45
        f[(c1|c2)&(Z>530)&(Z<1390)&np.isin(f,[4,5])]=6
        spread=np.maximum(np.exp(-((X-(1650+.34*(Z-950)))/480)**2),
                          .8*np.exp(-((X-fault)/360)**2))
        S=np.where(np.isin(f,[5,6]),.6*spread*np.exp(-((Z-990)/360)**2),0)
    return f,S

def elastic(f,S):
    phi,K0,Kd,mu,rm=[PROPS[f,i] for i in range(5)]
    Kf=1/((1-S)/2.5+S/.08)
    K=Kd+(1-Kd/K0)**2/(phi/Kf+(1-phi)/K0-Kd/K0**2)
    rho=(1-phi)*rm+phi*((1-S)*1030+S*650)
    return np.sqrt((K+4*mu/3)*1e9/rho),np.sqrt(mu*1e9/rho),rho

def synth(vp,rho,z,t,w):
    # Deposita reflectividades de interfaces en tiempo con pesos lineales.
    dz=np.diff(z)[:,None]
    twt=np.vstack([np.zeros(vp.shape[1]),np.cumsum(dz*(1/vp[:-1]+1/vp[1:]),axis=0)])
    ti=(twt[:-1]+twt[1:])/2
    ai=vp*rho
    r=np.diff(ai,axis=0)/(ai[:-1]+ai[1:])
    out=np.zeros((len(t),vp.shape[1]))
    for ix in range(vp.shape[1]):
        p=ti[:,ix]/(t[1]-t[0]); k=np.floor(p).astype(int); a=p-k
        valid=(k>=0)&(k+1<len(t))
        np.add.at(out[:,ix],k[valid],r[valid,ix]*(1-a[valid]))
        np.add.at(out[:,ix],k[valid]+1,r[valid,ix]*a[valid])
    return np.apply_along_axis(lambda tr:np.convolve(tr,w,'same'),0,out)

def panels(cases,x,y,keys,titles,path,cmap,limits=None,depth=True):
    fig,axs=plt.subplots(2,len(keys),figsize=(6*len(keys),8),squeeze=False,constrained_layout=True)
    for col,(key,title) in enumerate(zip(keys,titles)):
        vals=[c[key] for c in cases]
        lo,hi=limits[col] if limits else (min(a.min() for a in vals),max(a.max() for a in vals))
        for row,c in enumerate(cases):
            ax=axs[row,col]
            im=ax.imshow(c[key],extent=[x[0],x[-1],y[-1],y[0]],aspect='auto',
                         cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
            ax.set(title=c['name']+' — '+title,xlabel='Distancia (m)',
                   ylabel='Profundidad (m)' if depth else 'TWT (s)')
            if not depth: ax.set_ylim(.9,0)
        fig.colorbar(im,ax=axs[:,col].tolist(),shrink=.82)
    fig.savefig(path,dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='resultados');p.add_argument('--frequency',type=float,default=30)
    p.add_argument('--noise',type=float,default=.002,help='Desviacion estandar absoluta de amplitud')
    args=p.parse_args()
    if not 0<args.frequency<125 or args.noise<0: p.error('Frecuencia entre 0 y 125 Hz; ruido >=0')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    x=np.arange(401)*10.;z=np.arange(401)*4.;t=np.arange(1001)*.002
    wt=np.arange(-.128,.129,.002);a=(np.pi*args.frequency*wt)**2;w=(1-2*a)*np.exp(-a)
    cases=[];metrics={}
    for i,kind in enumerate(['arenisca','basalto']):
        f,S=geology(kind,x,z);v0,vs0,r0=elastic(f,S*0);v1,vs1,r1=elastic(f,S)
        b=synth(v0,r0,z,t,w);m=synth(v1,r1,z,t,w)
        rng=np.random.default_rng(45+i)
        bn=b+rng.normal(0,args.noise,b.shape);mn=m+rng.normal(0,args.noise,m.shape)
        d=dict(facies=f,saturation=S,vp=v0,delta_vp=v1-v0,delta_rho=r1-r0,
               delta_ai=(v1*r1-v0*r0)/1e6,baseline=bn,monitor=mn,difference=mn-bn,
               difference_clean=m-b,vp_monitor=v1,vs_baseline=vs0,vs_monitor=vs1,
               rho_baseline=r0,rho_monitor=r1,phi=PROPS[f,0])
        assert np.isfinite(v0).all() and np.all(v0>vs0) and np.all(r1>0)
        assert np.allclose(synth(*[v0,r0],z,t,w)-b,0)
        assert np.allclose((v1-v0)[S==0],0)
        if kind=='basalto': assert not np.any((S>0)&~np.isin(f,[5,6]))
        np.savez_compressed(out/(kind+'.npz'),x_m=x,z_m=z,time_s=t,**d)
        mask=S>.05
        metrics[kind]={'mean_delta_vp_s_gt_005_m_s':float((v1-v0)[mask].mean()),
             'max_abs_clean_4d':float(np.max(np.abs(m-b))),
             'co2_pore_area_m2_per_m_strike':float(np.sum(PROPS[f,0]*S)*10*4)}
        d['name']='Arenisca continua' if i==0 else 'Reservorio basaltico';cases.append(d)
    # Escalas compartidas por columna, incluida la diferencia 4D.
    panels(cases,x,z,['vp','saturation'],['Vp base (m/s)','Saturacion CO2'],out/'01_modelos.png','viridis',[(2500,6500),(0,.6)])
    lim=[max(np.max(np.abs(c[k])) for c in cases) for k in ['delta_vp','delta_rho','delta_ai']]
    panels(cases,x,z,['delta_vp','delta_rho','delta_ai'],['Delta Vp (m/s)','Delta densidad (kg/m3)','Delta AI (MRayl)'],out/'02_cambios_elasticos.png','RdBu_r',[(-v,v) for v in lim])
    amax=max(np.max(np.abs(c[k])) for c in cases for k in ['baseline','monitor'])
    dmax=max(np.max(np.abs(c['difference'])) for c in cases)
    panels(cases,x,t,['baseline','monitor','difference'],['Sismica base','Sismica monitor','Monitor menos base'],out/'03_sismica.png','seismic',[(-amax,amax),(-amax,amax),(-dmax,dmax)],False)
    fig,axs=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    for ax,c in zip(axs,cases):
        im=ax.imshow(c['facies'],extent=[0,x[-1],z[-1],0],aspect='auto',
            cmap=ListedColormap(COLORS),norm=BoundaryNorm(np.arange(8)-.5,7))
        ax.set(title=c['name'],xlabel='Distancia (m)',ylabel='Profundidad (m)')
    cb=fig.colorbar(im,ax=axs.tolist(),ticks=range(7));cb.ax.set_yticklabels(NAMES)
    fig.savefig(out/'00_geologia.png',dpi=160);plt.close(fig)
    (out/'metricas.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2));print('Figuras y arrays en',out.resolve())
if __name__=='__main__':main()

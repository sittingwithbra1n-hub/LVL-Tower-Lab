"""Experimental 3D Euler-Bernoulli space-frame FEM.

DOFs per node: (ux,uy,uz,rx,ry,rz). Units: m, kN, kN/m².
Only isotropic linear static response and dead load; no buckling / nonlinear LVL.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

GRAVITY = 9.80665

# СП 20.13330.2016 (изм. №5): табл. 11.1 и 11.2.
WIND_W0_KPA = {'Ia':0.17,'I':0.23,'II':0.30,'III':0.38,'IV':0.48,'V':0.60,'VI':0.73,'VII':0.85}
WIND_HEIGHT_M = np.array([5,10,20,40,60,80,100,150,200,250,300],dtype=float)
WIND_K = {
    'A': np.array([.75,1.00,1.25,1.50,1.70,1.85,2.00,2.25,2.45,2.65,2.75]),
    'B': np.array([.50,.65,.85,1.10,1.30,1.45,1.60,1.90,2.10,2.30,2.50]),
    'C': np.array([.40,.40,.55,.80,1.00,1.15,1.25,1.55,1.80,2.00,2.20])}

def wind_k(z:float,terrain:str)->float:
    if terrain not in WIND_K: raise ValueError('Неизвестная категория местности.')
    if z > 300: raise ValueError('Таблица СП 20 для z > 300 м не реализована.')
    return float(np.interp(max(float(z),0), WIND_HEIGHT_M, WIND_K[terrain]))

@dataclass(frozen=True)
class WindSpec:
    region: str='II'
    terrain: str='B'
    cx: float=1.4
    azimuth_deg: float=0.0
    pressure_multiplier: float=1.0

    def valid(self):
        if self.region not in WIND_W0_KPA or self.terrain not in WIND_K or not 0<=self.cx<=5 or self.pressure_multiplier<=0:
            raise ValueError('Неправильные параметры ветра.')


@dataclass(frozen=True)
class Material:
    b_mm: float=150
    h_mm: float=150
    e_mpa: float=12000
    nu: float=0.2
    density_kg_m3: float=500


def rectangle(m:Material):
    if m.b_mm<=0 or m.h_mm<=0 or m.e_mpa<=0 or not (-0.95 < m.nu < 0.5) or m.density_kg_m3<=0:
        raise ValueError('Проверьте материал и размеры сечения.')
    b,h = m.b_mm/1000,m.h_mm/1000
    a=b*h
    iy=b*h**3/12
    iz=h*b**3/12
    larger,smaller=max(b,h),min(b,h)
    j=larger*smaller**3*(1/3-0.21*(smaller/larger)*(1-(smaller/larger)**4/12))
    e=m.e_mpa*1000
    g=e/(2*(1+m.nu))
    w=m.density_kg_m3*GRAVITY/1000
    return dict(A=a,Iy=iy,Iz=iz,J=j,E=e,G=g,gamma=w)


def create_tower(height=3.0, rb=3.,rt=0.519615,angle=90.,n=16,steps=10):
    """Straight segments between two rotated regular polygons, plus ring chords.

    This is ONE family of generators and horizontal rings, not a two-family
    crossed Shukhov lattice. For parameter studies, bottom is the WIDE ring.
    """
    if height<=0 or rb<=0 or rt<=0 or n<4 or steps<1:
        raise ValueError('Некорректные параметры геометрии.')
    pts=[]
    twist=np.deg2rad(angle)
    for i in range(n):
        theta=np.pi/2 + 2*np.pi*i/n
        b=np.array([rb*np.cos(theta),rb*np.sin(theta),0.])
        t=np.array([rt*np.cos(theta+twist),rt*np.sin(theta+twist),height])
        for k in range(steps+1):
            f=k/steps
            pts.append((1-f)*b+f*t)
    bars=[]
    for i in range(n):
        off=i*(steps+1)
        for j in range(steps):
            bars.append((off+j,off+j+1))
    for j in range(steps+1):
        for i in range(n):
            ni=i*(steps+1)+j
            nj=((i+1)%n)*(steps+1)+j
            bars.append((ni,nj))
    nodes=np.asarray(pts)
    fix=np.concatenate([np.arange(6*i*(steps+1),6*i*(steps+1)+6) for i in range(n)])
    return nodes, np.asarray(bars, dtype=int), fix


def local_stiffness(l,sec):
    e,g,a,iy,iz,j=(sec[k] for k in ('E','G','A','Iy','Iz','J'))
    k=np.zeros((12,12))
    for i,other,st in ((0,6,e*a/l),(3,9,g*j/l)):
        k[i,i]+=st;k[other,other]+=st;k[i,other]-=st;k[other,i]-=st
    ids=[1,5,7,11]
    kz=e*iz/l**3*np.array([[12,6*l,-12,6*l],[6*l,4*l*l,-6*l,2*l*l],[-12,-6*l,12,-6*l],[6*l,2*l*l,-6*l,4*l*l]])
    k[np.ix_(ids,ids)]+=kz
    ids=[2,4,8,10]
    ky=e*iy/l**3*np.array([[12,-6*l,-12,-6*l],[-6*l,4*l*l,6*l,2*l*l],[-12,6*l,12,6*l],[-6*l,2*l*l,6*l,4*l*l]])
    k[np.ix_(ids,ids)]+=ky
    return k


def rotation(x0,x1):
    d=x1-x0;l=float(np.linalg.norm(d))
    if l<1e-8:
        raise ValueError('Стержень нулевой длины.')
    ex=d/l
    reference=np.array([0.,0.,1.])
    if abs(np.dot(ex,reference))>0.90:
        reference=np.array([0.,1.,0.])
    ey=np.cross(reference,ex);ey/=np.linalg.norm(ey)
    ez=np.cross(ex,ey)
    r=np.array([ex,ey,ez])
    t=np.zeros((12,12))
    for offset in (0,3,6,9):t[offset:offset+3,offset:offset+3]=r
    return r,t,l


def distributed_load_local(l,r,q_global):
    qx,qy,qz=r @ np.asarray(q_global,dtype=float)
    f=np.zeros(12)
    f[[0,6]]=qx*l/2
    f[[1,7]]=qy*l/2
    f[5],f[11]=qy*l*l/12,-qy*l*l/12
    f[[2,8]]=qz*l/2
    f[4],f[10]=-qz*l*l/12,qz*l*l/12
    return f


def solve(coords,bars,fixed,material:Material, self_weight:bool=True, wind:WindSpec|None=None):
    """Linear space-frame solution. Wind is a preliminary mean-force approximation.

    Wind takes w0 and k(z) from SP20. Aerodynamic projection per member uses
    effective width sqrt(bh)*sin(angle with wind); c_x is user-supplied, and
    shielding and pulsation are NOT modelled. It is not normative verification.
    """
    coords=np.asarray(coords,dtype=float)
    bars=np.asarray(bars,dtype=int)
    fixed=np.unique(np.asarray(fixed,dtype=int))
    ndof=6*len(coords)
    if len(coords)==0 or len(bars)==0 or len(fixed)==0:
        raise ValueError('Пустая схема или отсутствуют закрепления.')
    if bars.min()<0 or bars.max()>=len(coords) or fixed.min()<0 or fixed.max()>=ndof:
        raise ValueError('Индекс узла или закрепления вне диапазона.')
    if wind is not None: wind.valid()
    prop=rectangle(material)
    rows,cols,values=[],[],[]
    load=np.zeros(ndof)
    cache=[]
    wind_q_vectors=[]
    length_total=0.
    total_wind=np.zeros(3)
    wind_dir=np.array([np.cos(np.deg2rad(wind.azimuth_deg)),np.sin(np.deg2rad(wind.azimuth_deg)),0.]) if wind else None
    for a,b in bars:
        r,t,l=rotation(coords[a],coords[b])
        kl=local_stiffness(l,prop)
        q=np.array([0.,0.,-prop['gamma']*prop['A']]) if self_weight else np.zeros(3)
        q_wind=np.zeros(3)
        if wind is not None:
            z_mid=max(0,float((coords[a,2]+coords[b,2])/2)-float(coords[:,2].min()))
            kz=wind_k(z_mid,wind.terrain)
            sin_v=float(np.sqrt(max(0,1-np.dot(r[0],wind_dir)**2)))
            # The strip model: pressure * member projected breadth * sin(angle).
            q_wind=WIND_W0_KPA[wind.region]*kz*wind.cx*wind.pressure_multiplier*np.sqrt(prop['A'])*sin_v*wind_dir
            q+=q_wind
            total_wind+=q_wind*l
        wind_q_vectors.append(q_wind.copy())
        fl=distributed_load_local(l,r,q)
        kg=t.T @ kl @ t
        fg=t.T @ fl
        ids=np.r_[np.arange(6*a,6*a+6),np.arange(6*b,6*b+6)]
        rows.extend(np.repeat(ids,12));cols.extend(np.tile(ids,12));values.extend(kg.ravel())
        load[ids]+=fg
        cache.append((ids,t,kl,fl,l))
        length_total+=l
    k=coo_matrix((values,(rows,cols)),shape=(ndof,ndof)).tocsr()
    free=np.setdiff1d(np.arange(ndof),fixed)
    u=np.zeros(ndof)
    u[free]=spsolve(k[free][:,free],load[free])
    if not np.all(np.isfinite(u)):
        raise ValueError('Система вырождена: проверьте закрепления/связность.')
    react=np.asarray(k @ u-load).reshape(-1,6)
    disp=u.reshape(-1,6)
    axial=[]; moments=[]; my=[]; mz=[]; mk=[]
    end_local=[]
    for ids,t,kl,fl,l in cache:
        fle=kl @ (t @ u[ids])-fl
        end_local.append([fle.copy()])
        axial.append((fle[6]-fle[0])/2)
        my.append(max(abs(fle[4]),abs(fle[10])))
        mz.append(max(abs(fle[5]),abs(fle[11])))
        mk.append(max(abs(fle[3]),abs(fle[9])))
        moments.append(max(np.hypot(fle[4],fle[5]),np.hypot(fle[10],fle[11])))
    end_local=np.asarray([q[0] for q in end_local])
    weight=prop['gamma']*prop['A']*length_total
    transl=disp[:,:3]*1000
    return {'coords':coords,'bars':bars,'fixed':fixed,'disp':disp,
            'axial':np.asarray(axial),'moments':np.asarray(moments),
            'my':np.asarray(my),'mz':np.asarray(mz),'mk':np.asarray(mk),
            'end_local':end_local,'reactions':react,
            'weight_kn':weight,'weight_applied_kn':weight if self_weight else 0,
            'wind_q_global_kn_per_m':np.asarray(wind_q_vectors),
            'wind_forces_kn':total_wind,'mass_t':material.density_kg_m3*prop['A']*length_total/1000,
            'length_m':length_total,
            'max_disp_mm':float(np.max(np.linalg.norm(disp[:,:3],axis=1))*1000),
            'abs_disp_xyz_mm':np.max(abs(transl),axis=0),
            'uz_min_mm':float(np.min(disp[:,2])*1000),
            'max_axial_kn':float(np.max(np.abs(axial))),
            'max_mx_knm':float(np.max(mz)),  # Mx in UI maps to local Mz about local z; see notes
            'max_my_knm':float(np.max(my)),
            'max_mk_knm':float(np.max(mk)),
            'max_endmoment_knm':float(np.max(moments)),
            'reaction_kn':float(react[np.unique(fixed//6),2].sum()) if len(fixed) else 0.0,
            'reaction_balance_error_kn':float(react[:,2].sum()-(weight if self_weight else 0)),
            'load_balance_error_kn':np.sum(react[:,:3],axis=0)-np.array([total_wind[0],total_wind[1],-(weight if self_weight else 0)])*(-1),
            'material_properties':prop}

def combined_stress_mpa(res):
    """Conservative max of |N|/A + |My|/Wy + |Mz|/Wz at element ends, not stability."""
    p=res['material_properties']; A=p['A']; b=np.sqrt(A)
    # For rectangle, infer side lengths from Iy/Iz and area.
    h=np.sqrt(12*p['Iy']/A); b=A/h
    Wy=p['Iy']/(h/2)
    Wz=p['Iz']/(b/2)
    forces=res['end_local']
    sigma=np.maximum(
        np.abs(forces[:,0])/A+np.abs(forces[:,4])/Wy+np.abs(forces[:,5])/Wz,
        np.abs(forces[:,6])/A+np.abs(forces[:,10])/Wy+np.abs(forces[:,11])/Wz)
    return float(np.max(sigma)/1000),np.asarray(sigma)/1000

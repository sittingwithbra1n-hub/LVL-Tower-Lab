"""3D tower presentation. No change to FEM state or coordinates."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from fem import rotation

# Palettes affect only the displayed frame members; never stiffness or load data.
BAR_PALETTES = {
    'Бирюза и янтарь': ('#15a2aa', '#d4a05f', '#ffdc5e'),
    'Графит и медь': ('#5e7281', '#b87349', '#f1cb73'),
    'Синий и серебро': ('#397cbc', '#a2b3c5', '#ffd575'),
    'Лес и золото': ('#3d996c', '#c4a452', '#ffda74'),
    'Фиолетовый и коралл': ('#9277c4', '#db8979', '#ffe28a'),
    'Монохром': ('#7d8b98', '#c4ccd4', '#ffd362'),
}


def make_figure(res, show_fixed=True, show_human=True, show_grid=False,
                axes_mode='Выбранного КЭ', selected_element=1,
                show_wind=False, palette='Бирюза и янтарь'):
    xyz=np.asarray(res['coords']); bars=np.asarray(res['bars'])
    fig=go.Figure()
    generator_color, ring_color, highlight_color = BAR_PALETTES.get(palette, BAR_PALETTES['Бирюза и янтарь'])
    zz=np.isclose(xyz[bars[:,0],2],xyz[bars[:,1],2],atol=1e-7)
    # Separate one FE trace segment from the next, and assign its number to
    # every vertex for a click on a line where WebGL reports plotly_click.
    for mask,color,label in ((~zz,generator_color,'Образующие'),(zz,ring_color,'Кольцевые стержни')):
        X,Y,Z,C=[],[],[],[]
        for idx in np.flatnonzero(mask):
            a,b=bars[idx]
            X.extend([xyz[a,0],xyz[b,0],None]);Y.extend([xyz[a,1],xyz[b,1],None]);Z.extend([xyz[a,2],xyz[b,2],None])
            C.extend([int(idx+1),int(idx+1),None])
        # FE line carries the element ID at its vertices. No gray midpoint markers.
        fig.add_trace(go.Scatter3d(x=X,y=Y,z=Z,mode='lines',customdata=C,
                  line=dict(color=color,width=7),name=label,
                  hovertemplate='КЭ № %{customdata}<extra></extra>',showlegend=False))
        # Near-transparent wider line enhances hit testing on a 3D bar itself.
        # It is not a separate selectable background or a marker.
        fig.add_trace(go.Scatter3d(x=X,y=Y,z=Z,mode='lines',customdata=C,
                  line=dict(color=color,width=16),opacity=.015,
                  hovertemplate='КЭ № %{customdata}<extra></extra>',showlegend=False))
    if 1<=selected_element<=len(bars):
        a_sel,b_sel=bars[selected_element-1]
        line_xyz=xyz[[a_sel,b_sel]]
        fig.add_trace(go.Scatter3d(x=line_xyz[:,0],y=line_xyz[:,1],z=line_xyz[:,2],
                mode='lines',line=dict(color=highlight_color,width=11),
                customdata=[selected_element,selected_element],
                name=f'Выбран КЭ № {selected_element}',showlegend=False,
                hovertemplate='КЭ № %{customdata}<extra></extra>'))
    if show_fixed:
        ids=np.unique(res['fixed']//6)
        fig.add_trace(go.Scatter3d(x=xyz[ids,0],y=xyz[ids,1],z=xyz[ids,2],mode='markers',
                                   marker=dict(size=4,color='#e96970'),name='Закрепления',hoverinfo='skip'))
    if show_human:
        ground=float(xyz[:,2].min()); midxy=np.mean(xyz[:,:2],axis=0)
        radius=float(np.max(np.linalg.norm(xyz[:,:2]-midxy,axis=1)))
        hx=float(midxy[0]+radius+0.6);hy=float(midxy[1]);
        segments=[((0,0,.76),(0,0,1.43)),((0,0,1.2),(-.19,0,.9)),((0,0,1.2),(.19,0,.9)),
                  ((0,0,.76),(-.16,0,0)),((0,0,.76),(.16,0,0))]
        X,Y,Z=[],[],[]
        for s,t in segments:
            X.extend([hx+s[0],hx+t[0],None]); Y.extend([hy+s[1],hy+t[1],None]); Z.extend([ground+s[2],ground+t[2],None])
        fig.add_trace(go.Scatter3d(x=X,y=Y,z=Z,mode='lines',line=dict(color='#929eae',width=6),
                                   name='Человек 1,70 м',hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=[hx],y=[hy],z=[ground+1.59],mode='markers',
                                   marker=dict(size=8,color='#929eae'),showlegend=False,hoverinfo='skip'))
    if axes_mode!='Выкл' and (axes_mode=='Всех КЭ' or selected_element>0):
        selected_indices=[selected_element-1] if axes_mode=='Выбранного КЭ' else list(range(len(bars)))
        axis_colors=('#df6565','#69ad72','#588ce0')
        for axis in range(3):
            X,Y,Z,XT,YT,ZT=[],[],[],[],[],[]
            for idx in selected_indices:
                a,b=bars[idx]
                r,_,length=rotation(xyz[a],xyz[b])
                base=(xyz[a]+xyz[b])*0.5
                # Local axes in rotation() are the *row* direction vectors.
                length_axis=(0.35 if axes_mode=='Выбранного КЭ' else .13)*length
                end=base+length_axis*r[axis]
                X.extend([base[0],end[0],None]);Y.extend([base[1],end[1],None]);Z.extend([base[2],end[2],None])
                XT.append(end[0]);YT.append(end[1]);ZT.append(end[2])
            fig.add_trace(go.Scatter3d(x=X,y=Y,z=Z,mode='lines',line=dict(color=axis_colors[axis],width=7),
                                       name=('x₁ продольная','y₁ поперечная','z₁ поперечная')[axis],hoverinfo='skip'))
            if axes_mode=='Выбранного КЭ':
                fig.add_trace(go.Scatter3d(x=XT,y=YT,z=ZT,mode='markers+text',
                         marker=dict(color=axis_colors[axis],size=5),
                         text=['x₁','y₁','z₁'] if False else [('x₁','y₁','z₁')[axis]],
                         textposition='top center',showlegend=False,hoverinfo='skip'))
    if show_wind and 'wind_q_global_kn_per_m' in res:
        q=res['wind_q_global_kn_per_m'];nz=np.flatnonzero(np.linalg.norm(q,axis=1)>1e-12)
        if len(nz):
            # Up to twelve representative FEs across elevations (not every loaded FE).
            height_mid=(xyz[bars[:,0],2]+xyz[bars[:,1],2])/2
            by_height=nz[np.argsort(height_mid[nz])]
            ids=by_height[np.unique(np.linspace(0,len(by_height)-1,min(12,len(by_height)),dtype=int))]
            overall_span=max(float(np.ptp(xyz,axis=0).max()),.5)
            qmax=float(np.linalg.norm(q,axis=1).max())
            X,Y,Z=[],[],[]; arrow_xyz=[]; arrow_uv=[]
            for idx in ids:
                a,b=bars[idx];src=(xyz[a]+xyz[b])/2
                direction=q[idx]/np.linalg.norm(q[idx])
                dst=src+q[idx]/qmax*overall_span*.12
                X.extend([src[0],dst[0],None]);Y.extend([src[1],dst[1],None]);Z.extend([src[2],dst[2],None])
                arrow_xyz.append(dst); arrow_uv.append(direction)
            fig.add_trace(go.Scatter3d(x=X,y=Y,z=Z,mode='lines',line=dict(color='#75abed',width=6),
                          name='Ветер (условные стрелки)',hoverinfo='skip'))
            axyz=np.asarray(arrow_xyz); avec=np.asarray(arrow_uv)
            fig.add_trace(go.Cone(x=axyz[:,0],y=axyz[:,1],z=axyz[:,2],
                u=avec[:,0],v=avec[:,1],w=avec[:,2],anchor='tip',
                sizemode='absolute',sizeref=max(.04,overall_span*.045),
                showscale=False,colorscale=[[0,'#75abed'],[1,'#75abed']],
                name='Направление ветра',hoverinfo='skip',showlegend=False))
    # Grid OFF means all XYZ tick numbers, axis labels, grid and background disappear.
    axis_kwargs=dict(visible=show_grid,showgrid=show_grid,showbackground=False,
                      zeroline=False,showspikes=False,showticklabels=show_grid,
                      ticks='' if not show_grid else 'outside',title='')
    fig.update_layout(height=626,margin=dict(l=0,r=0,t=0,b=0),hovermode='closest',
         scene=dict(xaxis={**axis_kwargs,'title':'X, м' if show_grid else ''},
                   yaxis={**axis_kwargs,'title':'Y, м' if show_grid else ''},
                   zaxis={**axis_kwargs,'title':'Z, м' if show_grid else ''},
                   aspectmode='data',dragmode='orbit',
                   camera=dict(eye=dict(x=1.65,y=1.7,z=.92))),
         showlegend=False,
         uirevision='preserve-camera')
    return fig

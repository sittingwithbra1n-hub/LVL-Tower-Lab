"""Run with `python verify.py`. Checks importer, gravity balance and parameterization."""
from pathlib import Path
import numpy as np
from scad_txt import load_scad_txt, geometric_summary, wide_base_model
from fem import Material, create_tower, solve,rectangle

sample=Path(__file__).parent/'sample'/'оболочка.txt'
model=load_scad_txt(sample.read_bytes())
geo=geometric_summary(model)
assert model.coords.shape == (176,3)
assert model.bars.shape == (336,2)
assert len(model.fixed_dofs) == 16*6
assert geo['fixed_low']==16 and geo['fixed_high']==0
assert abs(geo['h']-3.)<1e-8
assert abs(geo['r_zmin']-0.519615)<1e-5
assert abs(geo['r_zmax']-3.)<1e-5
assert abs(model.e_mpa-12000)<10
mat=Material(model.section_b_mm,model.section_h_mm,model.e_mpa,model.poisson,model.density_kg_m3)
r=solve(model.coords,model.bars,model.fixed_dofs,mat)
assert abs(r['reaction_kn']-r['weight_kn'])<1e-7
assert r['max_disp_mm']>0
coords,bars,fix=create_tower()
r2=solve(coords,bars,fix,mat)
assert abs(r2['weight_kn']-r['weight_kn'])<2e-4
assert abs(r2['reaction_kn']-r2['weight_kn'])<1e-7
print(f"SCAD TXT: {len(model.coords)} узлов, {len(model.bars)} КЭ, h = {geo['h']:.2f} м.")
print(f"В файле: нижний радиус {geo['r_zmin']:.6f} м, верхний {geo['r_zmax']:.6f} м; закреплено нижних узлов {geo['fixed_low']}.")
print(f"Собственный вес: {r['weight_kn']:.6f} кН; опорные реакции {r['reaction_kn']:.6f} кН; макс. |u| {r['max_disp_mm']:.6f} мм.")
print(f"После изменения ориентации и опоры: вес {r2['weight_kn']:.6f} кН, макс. |u| {r2['max_disp_mm']:.6f} мм.")
print('Проверки пройдены; соответствие SCAD по усилиям/перемещениям пока не подтверждено.')

coords3,bars3,fixed3,flipped3=wide_base_model(model)
assert flipped3
assert np.isclose(coords3[:,2].min(),0)
assert np.isclose(coords3[:,2].max(),3)
assert len(fixed3)==96
assert all(np.isclose(coords3[np.unique(fixed3//6),2],0))
assert np.isclose(np.linalg.norm(coords3[np.unique(fixed3//6),:2],axis=1).mean(),3,rtol=1e-6)
r3=solve(coords3,bars3,fixed3,mat)
assert abs(r3['weight_kn']-r['weight_kn'])<1e-6
assert abs(r3['reaction_kn']-r3['weight_kn'])<1e-7
assert r3['max_disp_mm']<r['max_disp_mm']
print(f'Широкое основание: вес {r3["weight_kn"]:.6f} кН, перемещение {r3["max_disp_mm"]:.6f} мм; расчёт МКЭ выполнен.')

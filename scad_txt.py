"""Small, deliberately strict SCAD text importer for a rod-based model.

Supported SCAD record types from the bundled sample: (1) bar connectivity,
(3) single rectangular S0 material/section, (4) compressed node positions,
(5) complete DOF restraints by sequences, (6)-(7) self-weight metadata.
Unknown conventions are reported, not silently interpreted.
"""
from dataclasses import dataclass
import re
import numpy as np

TF_TO_KN = 9.80665

@dataclass
class ScadModel:
    coords: np.ndarray
    bars: np.ndarray
    fixed_dofs: np.ndarray
    e_mpa: float
    poisson: float
    density_kg_m3: float
    section_b_mm: float
    section_h_mm: float
    description: str
    raw_element_count: int


def _section(source: str, ident: int) -> str:
    m = re.search(r'\(' + str(ident) + r'/(.*?)\)', source, re.S)
    if not m:
        raise ValueError(f'В TXT отсутствует раздел ({ident}).')
    return m.group(1)


def load_scad_txt(raw: bytes) -> ScadModel:
    try:
        source = raw.decode('cp1251')
    except UnicodeError:
        source = raw.decode('utf-8-sig')
    unit_line = re.search(r'33;([^/]*)/', source)
    if not unit_line or not re.search(r'\bм\s+1\s+см\s+100\s+Т\s+1\b', unit_line.group(1)):
        raise ValueError('Примерный импортёр поддерживает только систему единиц: м, см, Т, как в образце.')

    items = []
    for rec in _section(source, 1).split('/'):
        vals = rec.strip().split()
        if not vals:
            continue
        if len(vals) != 4:
            raise ValueError(f'Неподдерживаемая запись КЭ: {rec.strip()}')
        typ, prop, a, b = map(int, vals)
        if typ != 5 or prop != 1:
            raise ValueError('Сейчас поддерживаются только стержни КЭ 5 с единственной группой жёсткости №1.')
        items.append((a-1, b-1))
    bars = np.asarray(items, dtype=int)

    coords = []
    for token in _section(source, 4).split('/'):
        token = token.strip()
        if not token:
            continue
        if token.startswith('r '):
            m = re.fullmatch(r'r\s+1\s+(\d+)\s*:\s*(.+)', token, re.S)
            if not m:
                raise ValueError(f'Неизвестный способ сжатия координат: {token[:50]}')
            count = int(m.group(1))
            endpoint = np.asarray([float(v) for v in m.group(2).split()], float)
            if len(endpoint) != 3 or not coords:
                raise ValueError('Ошибка последовательности сжатых координат.')
            begin = coords[-1].copy()
            for k in range(1, count+1):
                coords.append(begin+(endpoint-begin)*(k/count))
        else:
            vals = [float(v) for v in token.split()]
            if len(vals) not in (2,3):
                raise ValueError(f'Ожидались две или три координаты, получено: {token[:50]}')
            if len(vals) == 2:
                vals.append(0.0)
            coords.append(np.asarray(vals,float))
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 0 or bars.max() >= len(coords):
        raise ValueError('Число узлов не согласуется с конечными элементами.')

    sec3 = _section(source, 3)
    m = re.search(r'\b1\s+S0\s+([\deE+.-]+)\s+([\deE+.-]+)\s+([\deE+.-]+)\s+NU\s+([\deE+.-]+)\s+RO\s+([\deE+.-]+)', sec3)
    if not m:
        raise ValueError('Пока поддерживается описание сечения S0 + NU + RO.')
    e_tf_m2, b_cm, h_cm, nu, ro = map(float, m.groups())
    # E in tonne-force/m² -> MPa; RO is assumed t/m³ as in this sample.
    e_mpa = e_tf_m2 * TF_TO_KN / 1000
    density = ro * 1000

    fixed = []
    sec5 = _section(source, 5)
    for line in sec5.split('/'):
        line = line.strip()
        if not line:
            continue
        m = re.fullmatch(r'(\d+)\s*-\s*(\d+)\s*:\s*(\d+)\s+r\s+(\d+)\s+(\d+)', line)
        if not m:
            raise ValueError('Неподдерживаемая запись закреплений: '+line)
        lo, hi, start, end, stride = map(int, m.groups())
        if not(1 <= lo <= hi <= 6) or stride < 1:
            raise ValueError('Некорректные закрепления.')
        for nd in range(start, end + 1, stride):
            if not 1 <= nd <= len(coords):
                raise ValueError('Закрепление ссылается на неизвестный узел.')
            fixed.extend(6*(nd-1)+(dof-1) for dof in range(lo, hi+1))
    fixed = np.unique(np.asarray(fixed,dtype=int))

    sec6 = _section(source, 6)
    sec7 = _section(source, 7)
    if 'собственный вес' not in sec6.lower() or not re.search(r'96\s+3\s+Load=1', sec7):
        raise ValueError('Загружение отличается от проверенного образца: требуется отдельная расшифровка.')
    covered = re.search(r'1:1-(\d+)', sec7)
    if covered is None or int(covered.group(1)) != len(bars):
        raise ValueError('Собственный вес назначен не всем элементам либо формат загружения иной.')

    return ScadModel(coords=coords,bars=bars,fixed_dofs=fixed,e_mpa=e_mpa,
                     poisson=nu,density_kg_m3=density,section_b_mm=b_cm*10,
                     section_h_mm=h_cm*10,description='Файл SCAD TXT — исходная сетка и закрепления',
                     raw_element_count=len(bars))


def geometric_summary(model: ScadModel):
    coords = model.coords
    z = coords[:,2]
    zlo,zhi = float(z.min()),float(z.max())
    minmask = np.isclose(z,zlo,atol=1e-5)
    maxmask = np.isclose(z,zhi,atol=1e-5)
    rb = float(np.mean(np.linalg.norm(coords[minmask,:2],axis=1)))
    rt = float(np.mean(np.linalg.norm(coords[maxmask,:2],axis=1)))
    fixed_nodes = np.unique(model.fixed_dofs//6)
    return {'h':zhi-zlo,'r_zmin':rb,'r_zmax':rt,
            'base_nodes':np.flatnonzero(minmask),'top_nodes':np.flatnonzero(maxmask),
            'fixed_nodes':fixed_nodes,'fixed_low':int(np.isin(fixed_nodes,np.flatnonzero(minmask)).sum()),
            'fixed_high':int(np.isin(fixed_nodes,np.flatnonzero(maxmask)).sum())}


def wide_base_model(model: ScadModel):
    """Create an explicit *modified* model with the wider ring at Z=0.

    Keeps SCAD element connectivity, section and node numbering, but
    reorients vertical coordinates when necessary and replaces the support
    set with six fixed DOFs at all nodes of the wide, bottom end ring.
    This is NOT the original TXT scheme, and must not be compared with its
    original SCAD numerical results without making the same changes there.
    """
    coords = model.coords.copy()
    zlo, zhi = float(coords[:,2].min()), float(coords[:,2].max())
    end_lo=np.isclose(coords[:,2],zlo,atol=1e-5)
    end_hi=np.isclose(coords[:,2],zhi,atol=1e-5)
    rlo=float(np.mean(np.linalg.norm(coords[end_lo,:2],axis=1)))
    rhi=float(np.mean(np.linalg.norm(coords[end_hi,:2],axis=1)))
    if rlo >= rhi:
        coords[:,2] -= zlo
        bottom_nodes=np.flatnonzero(end_lo)
        flipped=False
    else:
        coords[:,2] = zhi - coords[:,2]
        bottom_nodes=np.flatnonzero(end_hi)
        flipped=True
    fixed=np.concatenate([np.arange(6*i,6*i+6) for i in bottom_nodes]).astype(int)
    return coords, model.bars.copy(), fixed, flipped

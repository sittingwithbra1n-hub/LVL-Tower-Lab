"""LVL Tower Lab: import SCAD TXT, linear spatial frame, preliminary static wind."""
from __future__ import annotations
import csv
import hashlib
import json
import io
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from fem import (
    Material,WindSpec,WIND_W0_KPA,wind_k,rectangle,combined_stress_mpa,
    create_tower,solve,
)
from scad_txt import geometric_summary,load_scad_txt,wide_base_model
from tower_graph import BAR_PALETTES,make_figure
from tower_view import tower_plot

st.set_page_config(page_title='LVL Tower Lab — МКЭ гиперболоидной башни',layout='wide',page_icon='🏗️')
st.markdown('''<style>
.block-container{padding-top:1.5rem;max-width:1520px}
[data-testid="stMetricValue"]{font-size:1.4rem}
</style>''', unsafe_allow_html=True)
st.title('LVL Tower Lab')
st.caption('Гиперболоидные башни из LVL · пространственные стержни КЭ 5 · линейная статика')

with st.sidebar:
    st.header('Расчётная схема')
    modes=['Загрузить схему из SCAD','Построить башню самостоятельно']
    mode=st.radio('Режим работы',modes,index=0)
    if mode==modes[0]:
        uploaded=st.file_uploader('Экспорт SCAD (*.txt)',type=['txt'])
        raw=uploaded.getvalue() if uploaded else (Path(__file__).parent/'sample'/'оболочка.txt').read_bytes()
        try: model=load_scad_txt(raw)
        except Exception as exc:
            st.error('Не удалось прочитать TXT: '+str(exc));st.stop()
        coords,bars,fixed,flipped=wide_base_model(model)
        m0=Material(model.section_b_mm,model.section_h_mm,model.e_mpa,model.poisson,model.density_kg_m3)
        zlo,zhi=float(coords[:,2].min()),float(coords[:,2].max())
        lower=np.isclose(coords[:,2],zlo,atol=1e-5)
        upper=np.isclose(coords[:,2],zhi,atol=1e-5)
        rbot=float(np.mean(np.linalg.norm(coords[lower,:2],axis=1)))
        rtop=float(np.mean(np.linalg.norm(coords[upper,:2],axis=1)))
        with st.expander('Импортированная модель',expanded=False):
            st.write(f'{len(coords)} узлов · {len(bars)} КЭ · высота {zhi-zlo:.3f} м')
            st.write(f'Радиус нижнего кольца {rbot:.3f} м · верхнего {rtop:.3f} м')
            st.write(f'Закреплено {len(np.unique(fixed//6))} узлов основания.')
        st.caption('Для единого теста схема ориентируется широким кольцом вниз (Z=0), '
                   'нижний широкий контур жёстко закрепляется. Исходный TXT не изменяется; '
                   'расчёт может отличаться от исходной схемы SCAD.')
    else:
        st.subheader('Геометрия')
        height=st.number_input('Высота, м',0.5,150.0,3.0,step=.5)
        rb=st.number_input('Радиус широкого основания, м',0.1,30.0,3.0,step=.2)
        rt=st.number_input('Радиус верхнего кольца, м',min_value=.1,max_value=float(rb),
                           value=min(.519615,float(rb)),step=.1,format='%.6f')
        angle=st.slider('Поворот верхнего кольца относительно нижнего, °',0,180,90,5,
                         help='Поворот точек крепления верхнего кольца в плане вокруг Z. '
                              'Не равен углу наклона стержня к вертикали.')
        with st.expander('Что означает поворот верхнего кольца?',expanded=False):
            st.write('Нижнее и верхнее кольца имеют точки соединения с продольными стержнями. '
                     'Поворот φ — угол между направлениями от центра к соответствующим '
                     'узлам двух колец в горизонтальной проекции. Стержень соединяет эти узлы. '
                     'Это **не** угол, под которым стержень примыкает к верхнему кольцу.')
        n=st.select_slider('Количество образующих',options=[8,12,16,20,24,32],value=16)
        steps=st.slider('Делений по высоте',2,20,10)
        coords,bars,fixed=create_tower(height,rb,rt,angle,n,steps)
        m0=Material()
        delta=np.sqrt(max(0.,rb**2+rt**2-2*rb*rt*np.cos(np.deg2rad(angle))))
        st.metric('Фактический наклон образующей к вертикали',f'{np.rad2deg(np.arctan2(delta,height)):.2f}°')
        st.caption('Широкое кольцо всегда снизу. Сетка пока включает одну семью образующих и горизонтальные кольца.')
    st.subheader('Характеристики сечения')
    b=st.number_input('Ширина сечения, мм',25.,600.,float(m0.b_mm),step=5.)
    h=st.number_input('Высота сечения, мм',25.,600.,float(m0.h_mm),step=5.)
    e=st.number_input('E, МПа',500.,40000.,round(float(m0.e_mpa),2),step=500.)
    nu=st.number_input('Коэффициент Пуассона',0.,.49,float(m0.nu),step=.05)
    density=st.number_input('Плотность, кг/м³',200.,1000.,float(m0.density_kg_m3),step=25.)
    st.subheader('Отображение')
    show_fixed=st.checkbox('Показать закрепления',value=True)
    show_human=st.checkbox('Показать человека (1,70 м)',value=True)
    show_grid=st.checkbox('Показать координатную сетку',value=False,
                          help='Плоскости и сетка не участвуют в выборе стержней.')
    axes_mode=st.selectbox('Местные оси КЭ', ['Выкл','Выбранного КЭ','Всех КЭ'],index=1)
    bar_palette=st.selectbox('Цветовая тема стержней',list(BAR_PALETTES),index=0,
                             help='Меняются только цвета образующих и кольцевых стержней на модели. На расчёт это не влияет.')

# In-plane drawing origin: first mesh node, preserving lengths and FEM response.
coords=np.asarray(coords,dtype=float).copy()
coords[:,0]-=coords[0,0]
coords[:,1]-=coords[0,1]
mat=Material(b,h,e,nu,density)

# Settings in the loads tab are evaluated first, then used in both other tabs.
tab_results,tab_loads,tab_optim,tab_todo=st.tabs(['Результаты расчёта','Нагрузки','Подбор сечения','Нужно доработать'])

# Examples for preselection only: city may span distinct wind regions; verify map.
CITY_REGION={'Вручную':'II','Москва':'I','Санкт-Петербург':'II','Казань':'II',
             'Екатеринбург':'II','Новосибирск':'III','Самара':'III','Ростов-на-Дону':'III'}
if 'wind_region' not in st.session_state:st.session_state['wind_region']='II'
def city_changed():
    st.session_state['wind_region']=CITY_REGION[st.session_state['city_choice']]

with tab_loads:
    st.subheader('Загружения')
    st.caption('Собственный вес включён всегда и автоматически пересчитывается при изменении геометрии, сечения или плотности.')
    use_weight=True
    use_wind=st.checkbox('Учитывать ветер в расчёте',value=False,
                         help='Если включено, решается сочетание собственного веса и предварительной средней составляющей ветра.')
    show_wind_arrows=st.checkbox('Показывать стрелки ветра на модели',value=True,disabled=not use_wind,
                                 help='Только отображение: отключение стрелок НЕ убирает ветер из расчёта.')
    case='Собственный вес + ветер' if use_wind else 'Собственный вес'
    st.markdown('**Ветровая нагрузка — предварительная модель**')
    c1,c2,c3=st.columns(3)
    with c1:
        city=st.selectbox('Город',list(CITY_REGION),key='city_choice',on_change=city_changed,
                          help='Справочная привязка к ветровому району; фактический район площадки проверяется по карте 2 приложения Е СП 20.')
        region=st.selectbox('Ветровой район',list(WIND_W0_KPA),key='wind_region')
    with c2:
        terrain=st.selectbox('Тип местности',['A','B','C'],index=1,format_func=lambda v:{
            'A':'A — открытая местность','B':'B — городская/лесная','C':'C — плотная высотная застройка'}[v])
        wind_angle=st.slider('Направление ветра в плане, °',0,355,0,5,help='0° — вдоль +X; 90° — вдоль +Y.')
    with c3:
        c_drag=st.number_input('Принятый аэродинамический коэффициент c',0.1,3.0,1.4,.1,
                               help='Учебный параметр; НЕ нормативный коэффициент всей башни.')
        wind_mult=st.number_input('Коэффициент к средней нагрузке',0.1,3.0,1.0,.1,
                                  help='Множитель для параметрических опытов; по умолчанию 1,0.')
    w0=WIND_W0_KPA[region]
    tower_height=float(np.ptp(coords[:,2]))
    kz=wind_k(min(tower_height,300),terrain)
    v1,v2,v3=st.columns(3)
    v1.metric('w₀ (табл. 11.1)',f'{w0:.2f} кПа')
    v2.metric('k(H) (табл. 11.2)',f'{kz:.3f}')
    v3.metric('w₀ × k(H)',f'{w0*kz:.3f} кПа')
    st.info('В основе средней составляющей — wₘ = w₀·k(z)·c (СП 20.13330.2016, п. 11.1.3). '
            'На каждый стержень нагрузка назначается приближённо по его проекции на плоскость, '
            'перпендикулярную ветру. Затенение элементов, коэффициенты для пространственной решётки, '
            'пульсационная составляющая, динамические эффекты и сочетания по СП пока НЕ реализованы. '
            'Это предварительный расчёт, не нормативное обоснование башни.')
    if city!='Вручную':
        st.caption(f'Для «{city}» предварительно выбран район {CITY_REGION[city]}. '
                   'При выборе фактической площадки обязательно проверь район по карте 2 приложения Е.')
    st.caption('СП 20.13330.2016: таблицы 11.1 и 11.2; раздел 11.1; приложение В.1.14. '
               'Ветровой район выбирается по карте 2 приложения Е (в отдельных редакциях нумерация приложения может отличаться).')
    st.markdown('[Текст СП 20.13330.2016](https://meganorm.ru/mega_doc/norm/pravila/0/sp_20_13330_2016_svod_pravil_nagruzki_i_vozdeystviya.html)')

wind=WindSpec(region=region,terrain=terrain,cx=c_drag,azimuth_deg=wind_angle,
              pressure_multiplier=wind_mult) if use_wind else None
shape_hash=hashlib.sha256(coords.tobytes()+bars.tobytes()+fixed.tobytes()+repr((e,nu,density)).encode()).hexdigest()
try:
    res=solve(coords,bars,fixed,mat,self_weight=use_weight,wind=wind)
except Exception as exc:
    st.error('Расчёт не выполнен: '+str(exc));st.stop()


def csv_bytes(rows):
    s=io.StringIO();w=csv.writer(s,delimiter=';');w.writerows(rows)
    return ('\ufeff'+s.getvalue()).encode('utf-8')


# Node belongs to every FE for which it is an endpoint (possibly several).
node_membership=[[] for _ in range(len(coords))]
for ei,(ni,nj) in enumerate(bars,1):
    node_membership[int(ni)].append(ei)
    node_membership[int(nj)].append(ei)


def elements_for_node(idx:int)->str:
    members=node_membership[idx]
    return ('Примыкающие КЭ №' + ', '.join(map(str,members)) + '.' if members
            else 'Нет примыкающих конечных элементов.')


def displacement_location(axis:int)->str:
    # Numeric max corresponds exactly to abs_disp_xyz_mm displayed in the metric.
    column=res['disp'][:,axis]*1000
    index=int(np.argmax(np.abs(column)))
    xyz=coords[index]
    return (f'Узел №{index+1}; значение {column[index]:+.4f} мм. '
            f'Координаты узла: X={xyz[0]:.3f} м, Y={xyz[1]:.3f} м, Z={xyz[2]:.3f} м. '
            f'{elements_for_node(index)} Показан максимум по модулю среди узлов.')


def force_location(key:str,local_dof:int|None=None)->str:
    member=int(np.argmax(np.abs(res[key]))); a,b=map(int,bars[member])
    if local_dof is None:
        value=res['axial'][member]
        return (f'КЭ №{member+1} (узлы №{a+1} → №{b+1}); '
                f'продольное усилие N={value:+.4f} кН. '
                'Показана максимальная по модулю усреднённая величина N для КЭ.')
    endvec=res['end_local'][member]
    end=0 if abs(endvec[local_dof])>=abs(endvec[local_dof+6]) else 1
    signed=float(endvec[local_dof+6*end])
    target=(a,b)[end]
    return (f'КЭ №{member+1}, {"начало" if end==0 else "конец"}, узел №{target+1}; '
            f'момент {signed:+.4f} кН·м. '
            f'{elements_for_node(target)} '
            'Показан максимум по модулю на концах КЭ, не экстремум внутри стержня.')

with tab_results:
    st.subheader('Результаты расчёта')
    st.caption(f'Учитывается: {case.lower()}. Под «деформациями» здесь понимаются узловые перемещения (мм), максимальные по модулю; усилия указаны для КЭ.')
    left,right=st.columns([1.5,1.05],gap='large')
    with left:
        if 'selected_element' not in st.session_state:
            st.session_state['selected_element']=0
        if st.session_state['selected_element']>len(bars):
            st.session_state['selected_element']=0
            st.session_state['sync_selected_number']=True
        if st.session_state.get('sync_selected_number',True):
            st.session_state['selected_number_widget']=st.session_state['selected_element']
            st.session_state['sync_selected_number']=False
        selector=st.number_input('Номер КЭ (0 — снять выделение)',min_value=0,
                    max_value=len(bars),step=1,key='selected_number_widget',
                    help='Нажми непосредственно на стержень, введи номер или нажми Esc, когда активна 3D-модель.')
        st.session_state['selected_element']=int(selector)
        fig=make_figure(res,show_fixed=show_fixed,show_human=show_human,
             show_grid=show_grid,axes_mode=axes_mode,selected_element=int(selector),
             show_wind=use_wind and show_wind_arrows,palette=bar_palette)
        pick=tower_plot(fig,key='tower-interactive-picker')
        if isinstance(pick,dict) and 'nonce' in pick and pick['nonce']!=st.session_state.get('last_click_nonce'):
            st.session_state['last_click_nonce']=pick['nonce']
            if pick.get('action')=='clear':
                if int(selector)!=0:
                    st.session_state['selected_element']=0
                    st.session_state['sync_selected_number']=True
                    st.rerun()
            elif 'element' in pick:
                picked=int(pick['element'])
                if 1<=picked<=len(bars) and picked!=int(selector):
                    st.session_state['selected_element']=picked
                    st.session_state['sync_selected_number']=True
                    st.rerun()
        st.caption('Нажми на стержень для просмотра его результатов. Esc снимает выбор, '
                   'когда курсор находится в 3D-окне. Можно также ввести 0 в поле номера КЭ.')
    with right:
        st.metric('Собственный вес конструкции',f'{res["weight_kn"]:.3f} кН')
        st.caption('Расчёт автоматически обновляется при изменении сечения, материала, геометрии или загружения.')
        x,y,z=st.columns(3)
        x.metric('Деформации по X',f'{res["abs_disp_xyz_mm"][0]:.4f} мм',help=displacement_location(0))
        y.metric('Деформации по Y',f'{res["abs_disp_xyz_mm"][1]:.4f} мм',help=displacement_location(1))
        z.metric('Деформации по Z',f'{res["abs_disp_xyz_mm"][2]:.4f} мм',help=displacement_location(2))
        x,y=st.columns(2)
        x.metric('Макс. |N|',f'{res["max_axial_kn"]:.3f} кН',help=force_location('axial'))
        y.metric('Макс. |Mx| — изгиб',f'{res["max_mx_knm"]:.3f} кН·м',help=force_location('mz',5))
        x,y=st.columns(2)
        x.metric('Макс. |My| — изгиб',f'{res["max_my_knm"]:.3f} кН·м',help=force_location('my',4))
        y.metric('Макс. |Mk| — кручение',f'{res["max_mk_knm"]:.3f} кН·м',help=force_location('mk',3))
        st.caption('Для КЭ 5 локальная ось x₁ направлена вдоль стержня. Подпись Mx в этом интерфейсе '
                   'обозначает второй изгибающий момент (Mz в принятой локальной системе МКЭ / SCAD); '
                   'My — изгиб вокруг y₁, Mk — кручение вокруг x₁.')
        st.write(f'**Узлы / элементы:** {len(coords)} / {len(bars)}')
        st.write(f'**Длина стержней:** {res["length_m"]:.3f} м')
        st.write(f'**Масса стержней:** {res["mass_t"]:.3f} т')
        st.write(f'**Сумма вертикальных реакций:** {res["reactions"][:,2].sum():.5f} кН')
        if use_wind:
            st.write(f'**Горизонтальная ветровая нагрузка, результирующая:** '
                     f'{np.linalg.norm(res["wind_forces_kn"][:2]):.3f} кН')
        st.write(f'**Невязка равновесия по силам:** {np.linalg.norm(res["load_balance_error_kn"]):.2e} кН')

    st.markdown('### Результаты выбранного конечного элемента')
    if int(st.session_state['selected_element'])==0:
        st.caption('КЭ не выбран. Нажми на стержень модели или введи его номер.')
    else:
        selected=int(st.session_state['selected_element'])-1
        ni,nj=map(int,bars[selected])
        st.caption(f'КЭ №{selected+1}, узлы {ni+1} → {nj+1}; длина {np.linalg.norm(coords[nj]-coords[ni]):.4f} м. '
                   'Моменты и усилия на концах приведены со знаками в местной системе координат нашего МКЭ.')
        end_force=res['end_local'][selected]
        displacement_rows=[]
        for label,node in [('Начало',ni),('Конец',nj)]:
            ux,uy,uz=res['disp'][node,:3]*1000
            displacement_rows.append({'Узел':f'{label} · №{node+1}',
                        'По X, мм':round(float(ux),6),'По Y, мм':round(float(uy),6),
                        'По Z, мм':round(float(uz),6)})
        col_u,col_f=st.columns(2)
        with col_u:
            st.markdown('**Деформации (перемещения узлов)**')
            st.dataframe(displacement_rows,hide_index=True,use_container_width=True)
        with col_f:
            st.markdown('**Усилия и моменты на концах КЭ**')
            force_rows=[]
            for label,offset in [('Начало',0),('Конец',6)]:
                force_rows.append({'Конец КЭ':label,
                    'N, кН':round(float(end_force[offset+0])*(-1 if offset==0 else 1),5),
                    'Mx, кН·м':round(float(end_force[offset+5]),5),
                    'My, кН·м':round(float(end_force[offset+4]),5),
                    'Mk, кН·м':round(float(end_force[offset+3]),5)})
            st.dataframe(force_rows,hide_index=True,use_container_width=True)
        if use_wind:
            qw=res['wind_q_global_kn_per_m'][selected]
            st.caption(f'Автоматически назначенная распределённая ветровая нагрузка на КЭ №{selected+1}: '
                       f'qX = {qw[0]:.5f}, qY = {qw[1]:.5f}, qZ = {qw[2]:.5f} кН/м.')
        st.caption('Mx обозначает изгиб вокруг местной z₁; My — изгиб вокруг местной y₁; '
                   'Mk — кручение вокруг x₁. На эпюрах концевые знаки требуют учёта принятой конвенции.')

    node_rows=[['Узел','X, м','Y, м','Z, м','Ux, мм','Uy, мм','Uz, мм','Rx, рад','Ry, рад','Rz, рад']]
    for i,(c,d) in enumerate(zip(coords,res['disp']),1):
        node_rows.append([i,*np.round(c,7),*np.round(d[:3]*1000,8),*np.round(d[3:],10)])
    element_rows=[['КЭ','Нач. узел','Кон. узел','N, кН (+ растяжение)',
                   'Mx, кН·м (|Mz лок.|)','My, кН·м','Mk, кН·м']]
    for i,((a,bb),n,mx,my,mk) in enumerate(zip(bars,res['axial'],res['mz'],res['my'],res['mk']),1):
        element_rows.append([i,a+1,bb+1,round(float(n),8),round(float(mx),8),round(float(my),8),round(float(mk),8)])
    reaction_rows=[['Узел','Rx, кН','Ry, кН','Rz, кН','Mx, кН·м','My, кН·м','Mz, кН·м']]
    for i in np.unique(fixed//6):reaction_rows.append([i+1,*np.round(res['reactions'][i],8)])
    st.subheader('Экспорт результатов')
    a1,a2,a3=st.columns(3)
    a1.download_button('Перемещения · CSV',csv_bytes(node_rows),'lvl_nodes.csv','text/csv')
    a2.download_button('Усилия · CSV',csv_bytes(element_rows),'lvl_elements.csv','text/csv')
    a3.download_button('Реакции · CSV',csv_bytes(reaction_rows),'lvl_reactions.csv','text/csv')
    with st.expander('Таблицы численных результатов'):
        st.markdown('**Перемещения по узлам**')
        st.dataframe([dict(zip(node_rows[0],row)) for row in node_rows[1:]],hide_index=True)
        st.markdown('**Усилия по элементам**')
        st.dataframe([dict(zip(element_rows[0],row)) for row in element_rows[1:]],hide_index=True)

with tab_optim:
    st.subheader('Подбор экономичных прямоугольных сечений')
    st.write('Программа перебирает несколько стандартных размеров и выбирает минимальную массу '
             'среди вариантов, удовлетворяющих заданным **условным ограничениям по перемещениям и напряжениям**.')
    c1,c2=st.columns(2)
    with c1:
        limit_disp=st.number_input('Предельное полное перемещение, мм',min_value=.001,max_value=1000.,value=15.,step=1.)
        limit_sigma=st.number_input('Допускаемое эквивалентное нормальное напряжение, МПа',
                                     min_value=.1,max_value=300.,value=12.,step=1.,
                                     help='Учебный предел, не нормативное расчётное сопротивление LVL.')
    with c2:
        sizes=[75,100,125,150,175,200,225,250,275,300,350,400]
        b_choices=st.multiselect('Ширины, мм',sizes,default=[100,125,150,175,200,250])
        h_choices=st.multiselect('Высоты, мм',sizes,default=[100,125,150,175,200,250])
    st.caption(f'Поиск выполняется для загружения: **{case.lower()}**. '
               'Напряжения оцениваются приближённо по |N|/A + |My|/Wy + |Mz|/Wz.')
    candidates=[(bx,hx) for bx in b_choices for hx in h_choices]
    if len(candidates)>60:st.warning('Оставь не более 60 сочетаний ширины и высоты.')
    can_run=bool(candidates) and len(candidates)<=60
    if st.button('Рассчитать и подобрать',type='primary',disabled=not can_run):
        found=[]
        with st.spinner('Проверка вариантов расчётной схемы...'):
            for bx,hx in candidates:
                try:
                    test=solve(coords,bars,fixed,Material(bx,hx,e,nu,density),self_weight=use_weight,wind=wind)
                    sigma,_=combined_stress_mpa(test)
                    pass_def=test['max_disp_mm']<=limit_disp
                    pass_stress=sigma<=limit_sigma
                    found.append({'Ширина, мм':bx,'Высота, мм':hx,'Масса, т':round(test['mass_t'],4),
                        'Перемещение, мм':round(test['max_disp_mm'],5),
                        'Напряжение, МПа':round(sigma,5),
                        'По перемещениям':pass_def,'По напряжениям':pass_stress,
                        'Соответствует заданным ограничениям':pass_def and pass_stress})
                except Exception as exc:
                    st.warning(f'Вариант {bx}×{hx} мм: {exc}')
        st.session_state['opt_records']=found
        st.session_state['opt_constraints']=(limit_disp,limit_sigma,case,region,terrain,c_drag,wind_angle,wind_mult,use_weight,use_wind,shape_hash,tuple(b_choices),tuple(h_choices))
    if st.session_state.get('opt_records'):
        if st.session_state.get('opt_constraints')!=(limit_disp,limit_sigma,case,region,terrain,c_drag,wind_angle,wind_mult,use_weight,use_wind,shape_hash,tuple(b_choices),tuple(h_choices)):
            st.warning('После подбора изменились геометрия, свойства, нагрузки или пределы. Нажми «Рассчитать и подобрать» повторно.')
        found=st.session_state['opt_records']
        valid=sorted((d for d in found if d['Соответствует заданным ограничениям']),key=lambda x:x['Масса, т'])
        if valid:
            best=valid[0]
            st.success(f'Минимальная масса среди проверенных вариантов: {best["Ширина, мм"]}×{best["Высота, мм"]} мм · '
                       f'{best["Масса, т"]:.3f} т. Проверено вариантов: {len(found)}, подходящих: {len(valid)}.')
        else:st.warning('Ни один перебранный вариант не прошёл заданные ограничения.')
        st.dataframe(sorted(found,key=lambda x:x['Масса, т']),hide_index=True)
        chart=go.Figure()
        for ok,label,col in ((True,'Подходит','#2ea48b'),(False,'Не подходит','#ed9262')):
            group=[f for f in found if f['Соответствует заданным ограничениям']==ok]
            if group:
                chart.add_trace(go.Scatter(x=[d['Масса, т'] for d in group],
                    y=[d['Перемещение, мм'] for d in group],mode='markers',name=label,
                    marker=dict(color=col,size=9),
                    text=[f"{d['Ширина, мм']}×{d['Высота, мм']} мм" for d in group],
                    hovertemplate='%{text}<br>Масса %{x:.3f} т<br>Перемещение %{y:.3f} мм<extra></extra>'))
        chart.update_layout(xaxis_title='Масса стержней, т',yaxis_title='Максимальное перемещение, мм',height=370)
        st.plotly_chart(chart,use_container_width=True)
    st.warning('Это НЕ нормативный подбор сечения LVL: не проверяются устойчивость сжатых элементов, '
               'прочность при сдвиге и кручении, анизотропия, узлы и соединения, длительность действия '
               'нагрузки, коэффициенты условий работы, геометрическая нелинейность и динамический ветер. '
               'Предел по напряжению вводится исследователем, а не определяется по СП 64 автоматически.')

# Research roadmap: chapter references refer to the uploaded individual dissertation plan.
# Checkboxes are manual confirmations, not automatic claims of FEM capability.
RESEARCH_GROUPS = [
    ('Развитие расчётной модели', [
        ('verify_scad', 'Сопоставить результаты МКЭ со SCAD при одинаковых геометрии, опорах и нагрузках'),
        ('second_family', 'Реализовать оба семейства образующих гиперболоидной решётки'),
        ('geometry', 'Исследовать влияние параметров ячеек, углов стержней и соотношений размеров сечений'),
        ('joints', 'Исследовать влияние эксцентриситетов и податливости узлов соединений'),
        ('anisotropy', 'Ввести анизотропию LVL и характеристики по направлениям относительно волокон'),
        ('nonlinear', 'Реализовать физическую нелинейность и соответствующие критерии прочности LVL'),
        ('viscoelastic', 'Учесть ползучесть, вязкоупругость, длительность и характер изменения нагрузки'),
        ('wind_full', 'Доработать ветер для пространственной решётки по СП 20, включая пульсации и динамику'),
        ('stability', 'Добавить проверку устойчивости стержней и конструкции'),
    ]),
    ('Испытания LVL: вдоль волокон', [
        ('compression_0', 'Провести испытания на сжатие вдоль волокон'),
        ('tension_0', 'Провести испытания на растяжение вдоль волокон'),
        ('shear_0', 'Провести испытания на скалывание вдоль волокон'),
        ('loading_regimes', 'Оценить влияние режимов и скорости нагружения на прочность и деформирование'),
    ]),
    ('Испытания LVL: под различными углами к волокнам', [
        ('compression_angles', 'Провести испытания на сжатие при разных углах к волокнам'),
        ('tension_angles', 'Провести испытания на растяжение при разных углах к волокнам'),
        ('shear_angles', 'Провести испытания на скалывание при разных углах к волокнам'),
        ('angle_loading_speed', 'Исследовать влияние скорости нагружения при разных углах'),
        ('angle_bending_optional', 'Дополнительно: провести испытания на изгиб при разных углах к волокнам'),
    ]),
    ('Испытания башенных конструкций и анализ', [
        ('tower_test', 'Испытать решётчатую гиперболоидную LVL-конструкцию с вариацией конструктивных параметров'),
        ('validate_tower', 'Сопоставить численные и экспериментальные перемещения, усилия и несущую способность'),
        ('regressions', 'Получить регрессионные зависимости результатов от сочетаний конструктивных параметров'),
        ('high_elastic', 'Исследовать предел вынужденных высокоэластических деформаций LVL'),
        ('recommendations', 'Разработать рекомендации по рациональному проектированию с учётом анизотропии и ползучести'),
    ]),
]

with tab_todo:
    st.subheader('Нужно доработать')
    st.write('Контрольный список исследований и будущих функций программы по индивидуальному плану диссертации. '
             'Основные группы соотносятся с главами 2–4; дополнительные технические задачи и изгиб '
             'отмечены для развития программы по твоим предложениям.')
    st.caption('Отмечай выполнение вручную: галочка не включает функцию в расчётном ядре и не подтверждает автоматически результаты испытаний.')
    all_task_ids=[task_id for _,group in RESEARCH_GROUPS for task_id,_ in group]
    imported_file=st.file_uploader('Восстановить отметки из JSON',type=['json'],key='roadmap_upload',
                                  help='Отметки сохраняются в сеансе браузера. Для переноса на другой компьютер скачай JSON.')
    if imported_file is not None and st.button('Загрузить отметки'):
        try:
            content=json.loads(imported_file.getvalue().decode('utf-8-sig'))
            saved=content.get('completed',{})
            if not isinstance(saved,dict):
                raise ValueError('В JSON поле completed должно быть объектом.')
            for task_id in all_task_ids:
                st.session_state[f'research_task_{task_id}']=bool(saved.get(task_id,False))
            st.rerun()
        except (ValueError,UnicodeDecodeError,TypeError) as exc:
            st.error(f'Не удалось загрузить отметки: {exc}')
    for group_title,tasks in RESEARCH_GROUPS:
        st.markdown(f'**{group_title}**')
        for task_id,label in tasks:
            st.checkbox(label,key=f'research_task_{task_id}')
    done={task_id:bool(st.session_state.get(f'research_task_{task_id}',False)) for task_id in all_task_ids}
    completed=sum(done.values())
    st.progress(completed/len(all_task_ids),text=f'Отмечено: {completed} из {len(all_task_ids)}')
    st.download_button('Сохранить отмеченные задачи · JSON',
        data=json.dumps({'version':1,'completed':done},ensure_ascii=False,indent=2).encode('utf-8'),
        file_name='lvl_research_progress.json',mime='application/json',
        help='Сохрани файл, чтобы восстановить галочки при новом запуске сайта.')
    st.caption('Основа списка: индивидуальный план (главы 2–4, пункты 2.1–2.3, 3.1–3.3, 4.1–4.2) '
               'и формулировки задач из документа о теме аспирантуры. '
               'Проверка устойчивости, расширенный расчёт ветра и испытания на изгиб — дополнения к техническому плану.')

st.divider()
st.caption('Версия 0.8 · предварительный исследовательский прототип. '
           'Ветер — средняя статическая составляющая с приближённым распределением по стержням; '
           'полный расчёт по СП 20 и верификация SCAD пока не завершены.')

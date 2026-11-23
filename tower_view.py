"""Plotly WebGL 3D click picker embedded as a local Streamlit v1 component.

All JS is shipped with the project; the chart works without external CDNs.
The component returns {action:'select',element,nonce} or {action:'clear',nonce} on Esc.
"""
from pathlib import Path
import json
import plotly.utils
import streamlit.components.v1 as components

_component=components.declare_component('lvl_tower_picker',path=str(Path(__file__).resolve().parent/'tower_picker'))

def tower_plot(figure,key='tower-picker'):
    return _component(figure_json=json.dumps(figure,cls=plotly.utils.PlotlyJSONEncoder),
                      revision='v08',key=key,default=None)

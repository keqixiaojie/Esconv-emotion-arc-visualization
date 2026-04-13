import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dash
from dash import html, dcc, Input, Output, State, ALL, ctx
import plotly.graph_objects as go
import numpy as np
from src.vad_extractor import VADExtractor
from src.emotion_smoothing import smooth_scores
from src.esconv_loader import ESConvLoader

LEXICON_PATH = "NRC-VAD-Lexicon-v2.1.txt"
ESCONV_PATH = "ESConv-strategy.json"
CACHE_DIR = "src/cache"
MARKERS_FILE = os.path.join(CACHE_DIR, "markers.json")
os.makedirs(CACHE_DIR, exist_ok=True)

vad_extractor = VADExtractor(LEXICON_PATH)
esconv_loader = ESConvLoader(ESCONV_PATH)
conv_ids = esconv_loader.get_conversation_ids()

MARKER_COLORS = {'seeker': '#4CAF50', 'supporter': '#2196F3'}
MARKER_ICONS = {'seeker': '🟢', 'supporter': '🔵'}

# 存储格式: { "conv_0": [{"turn":3,"speaker":"seeker","label":"..."}, ...] }
def _ck(cid): return f"conv_{cid}"

def load_all_markers():
    if os.path.exists(MARKERS_FILE):
        with open(MARKERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_all_markers(data):
    with open(MARKERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_conv_markers(cid):
    return load_all_markers().get(_ck(cid), [])

def add_marker(cid, turn, spk):
    a = load_all_markers(); k = _ck(cid); lst = a.get(k, [])
    if any(m['turn'] == turn for m in lst): return lst
    lst.append({'turn': turn, 'speaker': spk, 'label': ''})
    lst.sort(key=lambda m: m['turn']); a[k] = lst; save_all_markers(a); return lst

def remove_marker(cid, turn):
    a = load_all_markers(); k = _ck(cid)
    lst = [m for m in a.get(k, []) if m['turn'] != turn]
    a[k] = lst; save_all_markers(a); return lst

def update_label(cid, turn, label):
    a = load_all_markers(); k = _ck(cid)
    for m in a.get(k, []):
        if m['turn'] == turn: m['label'] = label; break
    save_all_markers(a); return a.get(k, [])

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "ESConv 对话情感弧线分析"

app.layout = html.Div([
    html.H2("ESConv 对话情感弧线分析", style={'textAlign': 'center', 'color': '#333'}),
    html.Div([
        html.Div([
            html.Label("对话 ID:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='conv-id-dropdown',
                options=[{'label': f'#{c}', 'value': c} for c in conv_ids],
                value=conv_ids[0] if conv_ids else None, clearable=False, style={'width': '160px'})
        ], style={'display': 'inline-block', 'marginRight': '25px', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("说话者:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(id='speaker-radio',
                options=[{'label': 'Seeker', 'value': 'seeker'},
                         {'label': 'Supporter', 'value': 'supporter'},
                         {'label': 'Both', 'value': 'both'}],
                value='seeker', labelStyle={'display': 'inline-block', 'marginRight': '8px'})
        ], style={'display': 'inline-block', 'marginRight': '25px', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("维度:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(id='dim-radio',
                options=[{'label': 'V', 'value': 'valence'},
                         {'label': 'A', 'value': 'arousal'},
                         {'label': 'D', 'value': 'dominance'}],
                value='valence', labelStyle={'display': 'inline-block', 'marginRight': '8px'})
        ], style={'display': 'inline-block', 'marginRight': '25px', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("显示拐点:", style={'fontWeight': 'bold'}),
            dcc.Checklist(id='marker-filter',
                options=[{'label': ' 🟢 Seeker', 'value': 'seeker'},
                         {'label': ' 🔵 Supporter', 'value': 'supporter'}],
                value=['seeker', 'supporter'],
                labelStyle={'display': 'inline-block', 'marginRight': '8px'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '8px'}),
    html.Div([
        html.Label("窗口大小:", style={'fontWeight': 'bold'}),
        dcc.Slider(id='window-slider', min=2, max=20, step=1, value=4,
            marks={i: str(i) for i in range(2, 21, 2)},
            tooltip={"placement": "bottom", "always_visible": True})
    ], style={'width': '55%', 'padding': '8px 12px'}),
    html.Div(id='meta-info', style={'padding': '8px 12px', 'backgroundColor': '#e9ecef',
        'borderRadius': '8px', 'marginBottom': '8px', 'fontSize': '13px'}),
    html.Div([
        html.Div([dcc.Graph(id='emotion-arc-graph', style={'height': '550px'})],
            style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Div([
                html.Span("📜 对话内容", style={'fontSize': '15px', 'fontWeight': 'bold'}),
                html.Span(" (点击: 新增标记 / 编辑已有标记)", style={'fontSize': '11px', 'color': '#888'}),
            ], style={'marginBottom': '8px'}),
            html.Div(id='dialog-panel', style={
                'maxHeight': '550px', 'overflowY': 'auto', 'padding': '8px',
                'backgroundColor': '#fafafa', 'border': '1px solid #ddd',
                'borderRadius': '6px', 'fontSize': '12px'})
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ]),
    html.Div(id='status-info', style={'padding': '4px 12px', 'color': '#666', 'fontSize': '12px'}),
    # 标签编辑弹窗
    dcc.Store(id='editing-turn', data=None),
    html.Div(id='label-modal-container', children=[
        html.Div(id='label-modal', children=[
            html.Div([
                html.H4("编辑拐点标签", id='modal-title', style={'margin': '0 0 10px 0'}),
                dcc.Input(id='label-input', type='text', placeholder='输入标签描述...',
                    style={'width': '100%', 'padding': '8px', 'fontSize': '14px', 'marginBottom': '10px',
                           'border': '1px solid #ccc', 'borderRadius': '4px'}),
                html.Div([
                    html.Button("💾 保存", id='btn-save-label', n_clicks=0,
                        style={'padding': '6px 16px', 'marginRight': '10px', 'backgroundColor': '#4CAF50',
                               'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Button("🗑️ 删除标记", id='btn-delete-marker', n_clicks=0,
                        style={'padding': '6px 16px', 'marginRight': '10px', 'backgroundColor': '#f44336',
                               'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Button("✖ 取消", id='btn-cancel-modal', n_clicks=0,
                        style={'padding': '6px 16px', 'backgroundColor': '#9e9e9e',
                               'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                ])
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px',
                      'boxShadow': '0 4px 20px rgba(0,0,0,0.3)', 'width': '400px', 'maxWidth': '90%'})
        ], style={'display': 'none', 'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '100%',
                  'backgroundColor': 'rgba(0,0,0,0.4)', 'zIndex': 9999,
                  'justifyContent': 'center', 'alignItems': 'center'})
    ]),
    dcc.Store(id='vad-cache-store'),
    dcc.Store(id='markers-store', data=[]),
    dcc.Store(id='hover-turn-idx', data=-1),
], style={'maxWidth': '1400px', 'margin': '0 auto', 'fontFamily': 'Arial, sans-serif', 'padding': '15px'})

app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var p = document.getElementById('dialog-panel');
            if (!p) return;
            var h = p.querySelector('[data-highlighted="true"]');
            if (h) { h.scrollIntoView({behavior: 'smooth', block: 'center'}); }
        }, 100);
        return -1;
    }
    """,
    Output('hover-turn-idx', 'data'),
    Input('dialog-panel', 'children'))


@app.callback(
    Output('vad-cache-store', 'data'),
    Output('meta-info', 'children'),
    Output('markers-store', 'data'),
    Input('conv-id-dropdown', 'value'),
    Input('speaker-radio', 'value'))
def on_conv_change(conv_id, speaker):
    if conv_id is None: return None, "请选择对话", []
    conv = esconv_loader.get_conversation(conv_id)
    if not conv: return None, f"未找到 #{conv_id}", []
    meta = conv.get('meta', {}); dialog = conv.get('dialog', [])
    meta_el = html.Div([
        html.Span(f"🆔 {meta.get('id', conv_id)}", style={'marginRight': '15px'}),
        html.Span(f"😟 {meta.get('emotion_type', 'N/A')}", style={'marginRight': '15px'}),
        html.Span(f"📋 {meta.get('problem_type', 'N/A')}", style={'marginRight': '15px'}),
        html.Span(f"🔥 初始: {meta.get('initial_emotion_intensity', 'N/A')}", style={'marginRight': '15px', 'color': '#d32f2f'}),
        html.Span(f"🌿 最终: {meta.get('final_emotion_intensity', 'N/A')}", style={'color': '#388e3c'}),
        html.Br(),
        html.Span(f"📝 {meta.get('situation', 'N/A')}", style={'fontStyle': 'italic'})])
    utterances = esconv_loader.filter_utterances(dialog, speaker)
    if not utterances: return None, meta_el, get_conv_markers(conv_id)
    text = esconv_loader.utterances_to_text(utterances)
    vad_results = vad_extractor.extract(text)
    tm = esconv_loader.build_turn_mapping(utterances, vad_results)
    for i, v in enumerate(vad_results):
        v['turn_info'] = tm[i] if i < len(tm) and tm[i] else None
    cache = {'conv_id': conv_id, 'speaker': speaker, 'results': vad_results, 'dialog': dialog}
    cp = os.path.join(CACHE_DIR, f"vad_conv{conv_id}_{speaker}.json")
    vad_extractor.save_cache(vad_results, cp, metadata={'conv_id': conv_id, 'speaker': speaker})
    return cache, meta_el, get_conv_markers(conv_id)


# 点击轮次：未标记→新增；已标记→打开编辑弹窗
@app.callback(
    Output('markers-store', 'data', allow_duplicate=True),
    Output('editing-turn', 'data'),
    Output('label-input', 'value'),
    Output('label-modal', 'style'),
    Output('modal-title', 'children'),
    Input({'type': 'turn-div', 'index': ALL}, 'n_clicks'),
    State('conv-id-dropdown', 'value'),
    State('vad-cache-store', 'data'),
    State('markers-store', 'data'),
    prevent_initial_call=True)
def on_turn_click(all_clicks, conv_id, cache, markers):
    no_modal = {'display': 'none'}
    show_modal = {'display': 'flex', 'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '100%',
                  'backgroundColor': 'rgba(0,0,0,0.4)', 'zIndex': 9999,
                  'justifyContent': 'center', 'alignItems': 'center'}
    if not ctx.triggered_id or conv_id is None or not cache:
        return dash.no_update, dash.no_update, dash.no_update, no_modal, ""
    turn_idx = ctx.triggered_id['index']
    if not all_clicks or sum(c for c in all_clicks if c) == 0:
        return dash.no_update, dash.no_update, dash.no_update, no_modal, ""
    markers = markers or []
    existing = [m for m in markers if m['turn'] == turn_idx]
    if existing:
        lbl = existing[0].get('label', '')
        spk = existing[0].get('speaker', '')
        icon = MARKER_ICONS.get(spk, '📌')
        return dash.no_update, turn_idx, lbl, show_modal, f"编辑拐点标签 {icon} 轮次[{turn_idx}]"
    else:
        dialog = cache.get('dialog', [])
        spk = dialog[turn_idx].get('speaker', 'seeker') if turn_idx < len(dialog) else 'seeker'
        new_markers = add_marker(conv_id, turn_idx, spk)
        return new_markers, dash.no_update, dash.no_update, no_modal, ""


# 弹窗按钮操作
@app.callback(
    Output('markers-store', 'data', allow_duplicate=True),
    Output('label-modal', 'style', allow_duplicate=True),
    Output('editing-turn', 'data', allow_duplicate=True),
    Input('btn-save-label', 'n_clicks'),
    Input('btn-delete-marker', 'n_clicks'),
    Input('btn-cancel-modal', 'n_clicks'),
    State('editing-turn', 'data'),
    State('label-input', 'value'),
    State('conv-id-dropdown', 'value'),
    prevent_initial_call=True)
def on_modal_action(save_c, del_c, cancel_c, editing_turn, label_val, conv_id):
    no_modal = {'display': 'none'}
    if not ctx.triggered_id or editing_turn is None or conv_id is None:
        return dash.no_update, no_modal, None
    tid = ctx.triggered_id
    if tid == 'btn-save-label':
        new_m = update_label(conv_id, editing_turn, label_val or '')
        return new_m, no_modal, None
    elif tid == 'btn-delete-marker':
        new_m = remove_marker(conv_id, editing_turn)
        return new_m, no_modal, None
    else:
        return dash.no_update, no_modal, None


@app.callback(
    Output('emotion-arc-graph', 'figure'),
    Output('status-info', 'children'),
    Input('dim-radio', 'value'),
    Input('window-slider', 'value'),
    Input('vad-cache-store', 'data'),
    Input('markers-store', 'data'),
    Input('marker-filter', 'value'))
def update_graph(dim, ws, cache, markers, mf):
    if not cache or not cache.get('results'):
        fig = go.Figure(); fig.update_layout(title="暂无数据"); return fig, "暂无数据"
    results = cache['results']
    scores = np.array([r[dim] for r in results])
    if len(scores) < ws:
        fig = go.Figure(); fig.update_layout(title=f"词数({len(scores)})<窗口({ws})"); return fig, "词数不足"
    smooth = smooth_scores(scores, ws)
    xd = np.arange(len(scores)); xs = np.arange(ws - 1, len(scores))
    ht_d, cd_d = [], []
    for i, r in enumerate(results):
        ti = r.get('turn_info'); tidx = ti['turn_index'] if ti else -1
        ht_d.append(f"{r['term']} {dim[0].upper()}={r[dim]:.3f} T[{tidx}]" if tidx >= 0
                     else f"{r['term']} {dim[0].upper()}={r[dim]:.3f}")
        cd_d.append(tidx)
    ht_s, cd_s = [], []
    for i in range(len(smooth)):
        center = i + ws // 2; tidx = -1
        if center < len(results):
            ti = results[center].get('turn_info')
            if ti: tidx = ti['turn_index']
        ht_s.append(f"平滑={smooth[i]:.3f} T[{tidx}]" if tidx >= 0 else f"平滑={smooth[i]:.3f}")
        cd_s.append(tidx)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xd.tolist(), y=scores.tolist(), mode='markers+lines', name='离散词',
        line=dict(dash='dot', color='rgba(150,150,150,0.5)'), marker=dict(size=6, color='gray'),
        text=ht_d, hoverinfo='text', customdata=cd_d))
    clr = {'valence': 'crimson', 'arousal': 'darkorange', 'dominance': 'steelblue'}
    fig.add_trace(go.Scatter(x=xs.tolist(), y=smooth.tolist(), mode='lines+markers',
        name=f'{dim.capitalize()} (W={ws})',
        line=dict(color=clr.get(dim, 'crimson'), width=3), marker=dict(size=4),
        text=ht_s, hoverinfo='text', customdata=cd_s))
    fig.add_hline(y=0.5, line_dash="dash", line_color="black", opacity=0.3)
    # 拐点标记：竖线 + 可hover的散点（显示标签）
    mf = mf or []
    if markers:
        y_max = float(max(scores)) + 0.05
        mk_x, mk_y, mk_text, mk_color = [], [], [], []
        for m in markers:
            if m['speaker'] not in mf: continue
            mt = m['turn']; ms = m['speaker']; ml = m.get('label', '')
            x_pos = [i for i, r in enumerate(results) if r.get('turn_info') and r['turn_info']['turn_index'] == mt]
            if not x_pos:
                all_t = [(i, r['turn_info']['turn_index']) for i, r in enumerate(results) if r.get('turn_info')]
                if not all_t: continue
                mid_x = min(all_t, key=lambda t: abs(t[1] - mt))[0]
            else:
                mid_x = x_pos[len(x_pos) // 2]
            mc = MARKER_COLORS.get(ms, '#9c27b0')
            ds = 'dash' if ms == 'seeker' else 'dot'
            fig.add_vline(x=mid_x, line_dash=ds, line_color=mc, opacity=0.5, line_width=2)
            icon = MARKER_ICONS.get(ms, '📌')
            fig.add_annotation(x=mid_x, y=y_max, text=f"{icon}T{mt}", showarrow=False,
                font=dict(size=9, color=mc))
            # 收集hover散点数据
            mk_x.append(mid_x); mk_y.append(y_max + 0.03)
            hover_label = f"<b>{icon} 轮次[{mt}] {ms}</b>"
            if ml: hover_label += f"<br>📝 {ml}"
            else: hover_label += "<br><i>(无标签)</i>"
            mk_text.append(hover_label); mk_color.append(mc)
        if mk_x:
            fig.add_trace(go.Scatter(x=mk_x, y=mk_y, mode='markers', name='拐点标记',
                marker=dict(size=12, color=mk_color, symbol='diamond', line=dict(width=1, color='white')),
                hovertext=mk_text, hoverinfo='text', showlegend=True))
    fig.update_layout(
        title=f"#{cache.get('conv_id','?')} | {cache.get('speaker','?')} | {dim.capitalize()}",
        xaxis_title="情感词索引", yaxis_title=f"{dim.capitalize()}",
        hovermode="closest", height=530, margin=dict(l=40, r=20, t=45, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    shown = len([m for m in (markers or []) if m['speaker'] in mf])
    return fig, f"✅ {len(results)}词 | W={ws} | 弧线{len(smooth)} | 拐点{shown}/{len(markers or [])}"


@app.callback(
    Output('dialog-panel', 'children'),
    Input('emotion-arc-graph', 'hoverData'),
    Input('vad-cache-store', 'data'),
    Input('markers-store', 'data'))
def on_hover_dialog(hoverData, cache, markers):
    if not cache or not cache.get('dialog'): return []
    dialog = cache['dialog']; results = cache.get('results', [])
    markers = markers or []; mm = {m['turn']: m for m in markers}
    hl = None
    if hoverData and 'points' in hoverData and hoverData['points']:
        pt = hoverData['points'][0]; cd = pt.get('customdata')
        if cd is not None and cd >= 0: hl = cd
        elif pt.get('x') is not None:
            x = pt['x']
            if 0 <= x < len(results):
                ti = results[x].get('turn_info')
                if ti: hl = ti.get('turn_index')
    els = []
    for idx, turn in enumerate(dialog):
        spk = turn.get('speaker', ''); content = turn.get('content', ''); strategy = turn.get('strategy', '')
        c = '#2196F3' if spk == 'supporter' else '#4CAF50'
        bg = '#e3f2fd' if spk == 'supporter' else '#e8f5e9'
        bdr = 'none'; is_hl = (idx == hl); is_mk = (idx in mm)
        if is_hl: bdr = '3px solid #ff5722'; bg = '#fff3e0'
        if is_mk:
            mc = MARKER_COLORS.get(mm[idx]['speaker'], '#9c27b0')
            bdr = f'3px solid {mc}'
            if not is_hl: bg = '#f1f8e9' if mm[idx]['speaker'] == 'seeker' else '#e1f5fe'
        extras = []
        if strategy: extras.append(html.Span(f" [{strategy}]", style={'color': '#ff9800', 'fontSize': '11px', 'fontWeight': 'bold'}))
        if is_mk:
            mi = MARKER_ICONS.get(mm[idx]['speaker'], '📌')
            lbl = mm[idx].get('label', '')
            tag = f" {mi}"
            if lbl: tag += f" {lbl}"
            extras.append(html.Span(tag, style={'fontSize': '12px', 'color': MARKER_COLORS.get(mm[idx]['speaker'], '#9c27b0'), 'fontWeight': 'bold'}))
        props = {'data-highlighted': 'true'} if is_hl else {}
        els.append(html.Div([
            html.Span(f"[{idx}] ", style={'color': '#999', 'fontSize': '11px'}),
            html.Span(f"{spk}: ", style={'color': c, 'fontWeight': 'bold'}),
            html.Span(content)] + extras,
            id={'type': 'turn-div', 'index': idx}, n_clicks=0,
            style={'padding': '6px 8px', 'margin': '2px 0', 'backgroundColor': bg,
                   'borderRadius': '4px', 'border': bdr, 'fontSize': '12px',
                   'cursor': 'pointer', 'transition': 'all 0.15s'}, **props))
    return els


if __name__ == '__main__':
    app.run(debug=True, port=8051)

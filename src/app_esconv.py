import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dash
from dash import html, dcc, Input, Output, State, ALL, ctx
import plotly.graph_objects as go
import numpy as np
from src.vad_extractor import VADExtractor, SentenceVADPredictor
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

# 句粒度 VAD 预测器（模型常驻 GPU）
CKPT_DIR = "ckpt/trained"
CONFIG_CACHE = "ckpt/roberta-large/model/config/"
VOCAB_CACHE = "ckpt/roberta-large/vocab/"
try:
    sent_predictor = SentenceVADPredictor(CKPT_DIR, CONFIG_CACHE, VOCAB_CACHE, epoch=15)
except Exception as e:
    print(f"⚠️ 句粒度模型加载失败: {e}，句粒度功能不可用")
    sent_predictor = None

MARKER_COLORS = {'seeker': '#4CAF50', 'supporter': '#2196F3'}
MARKER_ICONS = {'seeker': '🟢', 'supporter': '🔵'}
# 跟随型冷色系，引导型暖色系，Others灰
STRATEGY_COLORS = {
    'Question':                    '#42A5F5',  # 蓝
    'Reflection of feelings':      '#7E57C2',  # 紫
    'Restatement or Paraphrasing': '#26C6DA',  # 青
    'Self-disclosure':             '#5C6BC0',  # 靛蓝
    'Affirmation and Reassurance': '#66BB6A',  # 绿
    'Providing Suggestions':       '#FFA726',  # 橙
    'Information':                 '#FF7043',  # 深橙
    'Others':                      '#BDBDBD',  # 灰
}

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
            html.Label("显示拐点:", style={'fontWeight': 'bold'}),
            dcc.Checklist(id='marker-filter',
                options=[{'label': ' 🟢 Seeker', 'value': 'seeker'},
                         {'label': ' 🔵 Supporter', 'value': 'supporter'}],
                value=['seeker', 'supporter'],
                labelStyle={'display': 'inline-block', 'marginRight': '8px'})
        ], style={'display': 'inline-block', 'marginRight': '25px', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("分析粒度:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(id='granularity-radio',
                options=[{'label': '词粒度', 'value': 'word'},
                         {'label': '句粒度', 'value': 'sentence',
                          'disabled': sent_predictor is None}],
                value='word', labelStyle={'display': 'inline-block', 'marginRight': '8px'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '8px'}),
    html.Div([
        html.Div([
            html.Label("窗口大小:", style={'fontWeight': 'bold'}),
            dcc.Slider(id='window-slider', min=2, max=20, step=1, value=4,
                marks={i: str(i) for i in range(2, 21, 2)},
                tooltip={"placement": "bottom", "always_visible": True})
        ], style={'width': '55%', 'display': 'inline-block', 'verticalAlign': 'middle'}),
        html.Div([
            html.Label("平滑模式:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(id='smooth-mode-radio',
                options=[{'label': '标准均值', 'value': 'avg'},
                         {'label': '上文窗口', 'value': 'context'}],
                value='avg', labelStyle={'display': 'inline-block', 'marginRight': '10px'})
        ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'marginLeft': '30px'}),
    ], style={'padding': '8px 12px'}),
    html.Div(id='meta-info', style={'padding': '8px 12px', 'backgroundColor': '#e9ecef',
        'borderRadius': '8px', 'marginBottom': '8px', 'fontSize': '13px'}),
    html.Div([
        html.Div([
            dcc.Loading(type='circle', color='#666', children=[
                dcc.Graph(id='graph-valence'),
                dcc.Graph(id='graph-arousal'),
                dcc.Graph(id='graph-dominance'),
            ])
        ], style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Div([
                html.Span("📜 对话内容", style={'fontSize': '15px', 'fontWeight': 'bold'}),
                html.Span(" (点击: 新增标记 / 编辑已有标记)", style={'fontSize': '11px', 'color': '#888'}),
            ], style={'marginBottom': '8px'}),
            html.Div(id='dialog-panel', style={
                'maxHeight': '800px', 'overflowY': 'auto', 'padding': '8px',
                'backgroundColor': '#fafafa', 'border': '1px solid #ddd',
                'borderRadius': '6px', 'fontSize': '12px'})
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ]),
    dcc.Loading(type='dot', color='#888', children=[
        html.Div(id='status-info', style={'padding': '4px 12px', 'color': '#666', 'fontSize': '12px'})
    ]),
    html.Div([
        html.Div("📐 差值分析：Seeker 受引导（前）vs Seeker 被附和（后）",
                 style={'fontSize': '13px', 'fontWeight': 'bold', 'padding': '8px 12px 4px',
                        'color': '#555'}),
        dcc.Loading(type='circle', color='#666', children=[
            dcc.Graph(id='graph-diff-valence'),
            dcc.Graph(id='graph-diff-arousal'),
            dcc.Graph(id='graph-diff-dominance'),
        ])
    ], style={'marginTop': '8px', 'borderTop': '1px solid #ddd', 'paddingTop': '4px'}),
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
    Input('speaker-radio', 'value'),
    Input('granularity-radio', 'value'))
def on_conv_change(conv_id, speaker, granularity):
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

    def _compute_vad(utts):
        if granularity == 'sentence' and sent_predictor is not None:
            return sent_predictor.predict_utterances(utts), 'sentence'
        text = esconv_loader.utterances_to_text(utts)
        vad_r = vad_extractor.extract(text)
        tm = esconv_loader.build_turn_mapping(utts, vad_r)
        for i, v in enumerate(vad_r):
            v['turn_info'] = tm[i] if i < len(tm) and tm[i] else None
        return vad_r, 'word'

    vad_results, actual_gran = _compute_vad(utterances)
    suffix = '_sent' if actual_gran == 'sentence' else ''
    cp = os.path.join(CACHE_DIR, f"vad_conv{conv_id}_{speaker}{suffix}.json")
    vad_extractor.save_cache(vad_results, cp, metadata={'conv_id': conv_id, 'speaker': speaker, 'granularity': actual_gran})

    cache = {'conv_id': conv_id, 'speaker': speaker, 'granularity': actual_gran,
             'results': vad_results, 'dialog': dialog, 'utterances': utterances}

    # 同时计算背景说话者的弧线
    bg_spk = 'supporter' if speaker == 'seeker' else ('seeker' if speaker == 'supporter' else None)
    if bg_spk:
        bg_utts = esconv_loader.filter_utterances(dialog, bg_spk)
        if bg_utts:
            bg_results, _ = _compute_vad(bg_utts)
            cache['bg_speaker'] = bg_spk
            cache['bg_results'] = bg_results
            cache['bg_utterances'] = bg_utts

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
    Output('smooth-mode-radio', 'options'),
    Output('smooth-mode-radio', 'value'),
    Input('granularity-radio', 'value'),
    State('smooth-mode-radio', 'value'))
def update_smooth_mode_options(granularity, current_mode):
    options = [{'label': '标准均值', 'value': 'avg'},
               {'label': '上文窗口', 'value': 'context', 'disabled': granularity == 'word'}]
    value = 'avg' if granularity == 'word' else current_mode
    return options, value


def _build_figure(dim, ws, smooth_mode, cache, markers, mf):
    if not cache or not cache.get('results'):
        fig = go.Figure(); fig.update_layout(title="暂无数据"); return fig, "暂无数据"
    results = cache['results']
    gran = cache.get('granularity', 'word')
    is_sent = (gran == 'sentence')
    unit_label = '句' if is_sent else '词'
    scores = np.array([r[dim] for r in results])
    clr = {'valence': 'crimson', 'arousal': 'darkorange', 'dominance': 'steelblue'}
    mf = mf or []
    # 词粒度不支持上文窗口模式，回退到标准均值
    if smooth_mode == 'context' and not is_sent:
        smooth_mode = 'avg'

    # ---- 上文窗口模式 ----
    if smooth_mode == 'context':
        utterances = cache.get('utterances', [])
        if not utterances or len(utterances) < ws:
            fig = go.Figure()
            fig.update_layout(title=f"话语数({len(utterances or [])})<窗口({ws})")
            return fig, "话语数不足"

        # 句粒度对齐词粒度 x 轴：计算每句的词位置区间
        if is_sent:
            word_counts_ctx = [max(1, len(re.findall(r'\b[a-z]+\b', u['content'].lower()))) for u in utterances]
            word_starts_ctx = []
            cum = 0
            for c in word_counts_ctx:
                word_starts_ctx.append(cum)
                cum += c

        # 离散：每条话语的 VAD 得分
        if is_sent:
            utt_scores = scores.copy()
            utt_cd = [r['turn_info']['turn_index'] if r.get('turn_info') else -1 for r in results]
            utt_ht = [f"T[{utt_cd[i]}] {dim[0].upper()}={utt_scores[i]:.3f}" for i in range(len(results))]
            # 离散点画在词范围中点
            xd_utt = [word_starts_ctx[i] + (word_counts_ctx[i] - 1) // 2 for i in range(len(results))]
        else:
            from collections import defaultdict
            groups = defaultdict(list)
            for r in results:
                ti = r.get('turn_info')
                if ti:
                    groups[ti['turn_index']].append(r[dim])
            utt_scores_list, utt_ht, utt_cd, xd_utt = [], [], [], []
            for i, u in enumerate(utterances):
                tidx = u['turn_index']
                vals = groups[tidx]
                val = float(np.mean(vals)) if vals else 0.0
                utt_scores_list.append(val)
                utt_ht.append(f"T[{tidx}] {dim[0].upper()}={val:.3f} ({len(vals)} 词)")
                utt_cd.append(tidx)
                xd_utt.append(i)
            utt_scores = np.array(utt_scores_list)

        # 上文窗口：位置 i 取前 ws 条话语拼接为整体文本后统一打分
        ctx_scores_list, ctx_ht, ctx_cd = [], [], []
        for i in range(ws - 1, len(utterances)):
            window_utts = utterances[i - ws + 1 : i + 1]
            combined = ' '.join(u['content'] for u in window_utts)
            if is_sent and sent_predictor is not None:
                preds = sent_predictor._predict_batch([combined])
                dim_idx = {'valence': 0, 'arousal': 1, 'dominance': 2}[dim]
                val = (float(preds[0][dim_idx]) - 3.0) / 2.0
            else:
                wr = vad_extractor.extract(combined)
                val = float(np.mean([r[dim] for r in wr])) if wr else 0.0
            ctx_scores_list.append(val)
            t_start = utterances[i - ws + 1]['turn_index']
            t_end = utterances[i]['turn_index']
            ctx_ht.append(f"上文窗口 T[{t_start}~{t_end}] {dim[0].upper()}={val:.3f}")
            ctx_cd.append(utterances[i]['turn_index'])

        ctx_scores = np.array(ctx_scores_list)
        # 句粒度：平滑曲线从每句词起点开始（阶梯线）；词粒度：话语序号
        if is_sent:
            xs_ctx = [word_starts_ctx[i] for i in range(ws - 1, len(utterances))]
            ctx_line_shape = 'hv'
        else:
            xs_ctx = list(range(ws - 1, len(utterances)))
            ctx_line_shape = 'linear'

        discrete_name = '句离散' if is_sent else '词均值(轮次)'
        fig = go.Figure()

        # 背景说话者离散点（上文窗口模式，与主图同样的上文窗口计算方式）
        bg_ctx_turn_to_x = {}
        bg_spk_ctx = cache.get('bg_speaker', '')
        bg_utts_ctx = cache.get('bg_utterances', [])
        if bg_spk_ctx and bg_utts_ctx and len(bg_utts_ctx) >= ws:
            bg_ctx_scores, bg_ctx_cd = [], []
            for i in range(ws - 1, len(bg_utts_ctx)):
                window_utts = bg_utts_ctx[i - ws + 1 : i + 1]
                combined = ' '.join(u['content'] for u in window_utts)
                if is_sent and sent_predictor is not None:
                    preds = sent_predictor._predict_batch([combined])
                    dim_idx = {'valence': 0, 'arousal': 1, 'dominance': 2}[dim]
                    val = (float(preds[0][dim_idx]) - 3.0) / 2.0
                else:
                    wr = vad_extractor.extract(combined)
                    val = float(np.mean([r[dim] for r in wr])) if wr else 0.0
                bg_ctx_scores.append(val)
                bg_ctx_cd.append(bg_utts_ctx[i]['turn_index'])

            # 按"紧随其后的主句"分组，聚集在该主句 x 起点
            main_turn_indices_ctx = [u['turn_index'] for u in utterances]
            from collections import defaultdict as _dd3
            bg_ctx_groups = _dd3(list)
            for i_bg, (val, T_bg) in enumerate(zip(bg_ctx_scores, bg_ctx_cd)):
                strat_ctx = (bg_utts_ctx[i_bg + ws - 1].get('strategy') or 'Others')
                next_idx = next((i for i, T in enumerate(main_turn_indices_ctx) if T > T_bg),
                                len(main_turn_indices_ctx) - 1)
                bg_ctx_groups[next_idx].append((val, T_bg, strat_ctx))

            from collections import defaultdict as _dd4
            strat_ctx_dots = _dd4(lambda: {'x': [], 'y': [], 'text': [], 'cd': []})
            for main_idx in sorted(bg_ctx_groups.keys()):
                if main_idx >= len(utterances): continue
                x_base = float(word_starts_ctx[main_idx]) if is_sent else float(main_idx)
                for k, (val, T_bg, strat_ctx) in enumerate(bg_ctx_groups[main_idx]):
                    x_dot = x_base + k
                    strat_ctx_dots[strat_ctx]['x'].append(x_dot)
                    strat_ctx_dots[strat_ctx]['y'].append(val)
                    strat_ctx_dots[strat_ctx]['text'].append(f"T[{T_bg}] [{strat_ctx}] {dim[0].upper()}={val:.3f}")
                    strat_ctx_dots[strat_ctx]['cd'].append(T_bg)
                    bg_ctx_turn_to_x[T_bg] = x_dot

            for strat_ctx, dots in strat_ctx_dots.items():
                sc = STRATEGY_COLORS.get(strat_ctx, '#BDBDBD')
                fig.add_trace(go.Scatter(
                    x=dots['x'], y=dots['y'], mode='markers',
                    name=strat_ctx,
                    marker=dict(size=10, color=sc, opacity=0.75,
                                symbol='circle', line=dict(width=1.5, color='white')),
                    text=dots['text'], hoverinfo='text',
                    customdata=dots['cd']))

        fig.add_trace(go.Scatter(
            x=xd_utt, y=utt_scores.tolist(), mode='markers+lines', name=discrete_name,
            line=dict(dash='dot', color='rgba(150,150,150,0.5)'), marker=dict(size=6, color='gray'),
            text=utt_ht, hoverinfo='text', customdata=utt_cd))
        fig.add_trace(go.Scatter(
            x=xs_ctx, y=ctx_scores.tolist(), mode='lines+markers',
            name=f'{dim.capitalize()} 上文(W={ws})',
            line=dict(color=clr.get(dim, 'crimson'), width=3, shape=ctx_line_shape), marker=dict(size=4),
            text=ctx_ht, hoverinfo='text', customdata=ctx_cd))
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)

        # 拐点标记
        if markers and len(utt_scores) > 0:
            y_max = 1.0
            mk_x, mk_y, mk_text, mk_color = [], [], [], []
            for m in markers:
                if m['speaker'] not in mf: continue
                mt = m['turn']; ms = m['speaker']; ml = m.get('label', '')
                if ms == bg_spk_ctx and mt in bg_ctx_turn_to_x:
                    mid_x = bg_ctx_turn_to_x[mt]
                else:
                    sent_pos = [i for i, u in enumerate(utterances) if u['turn_index'] == mt]
                    if not sent_pos:
                        all_t = [(i, u['turn_index']) for i, u in enumerate(utterances)]
                        if not all_t: continue
                        si = min(all_t, key=lambda t: abs(t[1] - mt))[0]
                    else:
                        si = sent_pos[0]
                    mid_x = (word_starts_ctx[si] + (word_counts_ctx[si] - 1) // 2) if is_sent else si
                mc = MARKER_COLORS.get(ms, '#9c27b0')
                fig.add_vline(x=mid_x, line_dash='dash' if ms == 'seeker' else 'dot',
                              line_color=mc, opacity=0.5, line_width=2)
                icon = MARKER_ICONS.get(ms, '📌')
                fig.add_annotation(x=mid_x, y=y_max, text=f"{icon}T{mt}", showarrow=False,
                    font=dict(size=9, color=mc))
                mk_x.append(mid_x); mk_y.append(y_max)
                hover_label = f"<b>{icon} 轮次[{mt}] {ms}</b>"
                if ml: hover_label += f"<br>📝 {ml}"
                else: hover_label += "<br><i>(无标签)</i>"
                mk_text.append(hover_label); mk_color.append(mc)
            if mk_x:
                fig.add_trace(go.Scatter(x=mk_x, y=mk_y, mode='markers', name='拐点标记',
                    marker=dict(size=12, color=mk_color, symbol='diamond', line=dict(width=1, color='white')),
                    hovertext=mk_text, hoverinfo='text', showlegend=True))

        gran_tag = "句粒度" if is_sent else "词粒度"
        fig.update_layout(
            title=f"#{cache.get('conv_id','?')} | {cache.get('speaker','?')} | {dim.capitalize()} | {gran_tag} | 上文窗口",
            xaxis_title="情感词索引", yaxis_title=f"{dim.capitalize()}",
            yaxis=dict(range=[-1, 1]),
            hovermode="closest", height=260, margin=dict(l=40, r=20, t=35, b=25),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        shown = len([m for m in (markers or []) if m['speaker'] in mf])
        return fig, f"✅ {len(utterances)}话语 | W={ws} | 弧线{len(ctx_scores)} | 拐点{shown}/{len(markers or [])} | {gran_tag} | 上文窗口"

    # ---- 标准均值模式 ----
    if len(scores) < ws:
        fig = go.Figure(); fig.update_layout(title=f"{unit_label}数({len(scores)})<窗口({ws})"); return fig, f"{unit_label}数不足"
    smooth = smooth_scores(scores, ws)

    utterances = cache.get('utterances', [])
    if is_sent:
        # 句粒度对齐词粒度 x 轴：计算每句话的词位置区间
        utterances_c = utterances
        word_counts = [max(1, len(re.findall(r'\b[a-z]+\b', u['content'].lower()))) for u in utterances_c]
        word_starts = []
        cum = 0
        for c in word_counts:
            word_starts.append(cum)
            cum += c

        # 离散：每句画在其词范围中点
        xd, ht_d, cd_d = [], [], []
        for i, r in enumerate(results):
            ti = r.get('turn_info'); tidx = ti['turn_index'] if ti else -1
            xd.append(word_starts[i] + (word_counts[i] - 1) // 2)
            ht_d.append(f"T[{tidx}] {dim[0].upper()}={r[dim]:.3f}" if tidx >= 0 else f"{dim[0].upper()}={r[dim]:.3f}")
            cd_d.append(tidx)

        # 平滑曲线：smooth[j] 从 sentence(j+ws-1) 的词起点开始，阶梯线延伸到该句末尾
        xs, ht_s, cd_s = [], [], []
        for j in range(len(smooth)):
            ti = results[j + ws - 1].get('turn_info'); tidx = ti['turn_index'] if ti else -1
            xs.append(word_starts[j + ws - 1])
            ht_s.append(f"平滑={smooth[j]:.3f} T[{tidx}]" if tidx >= 0 else f"平滑={smooth[j]:.3f}")
            cd_s.append(tidx)
        smooth_line_shape = 'hv'
    else:
        xd = np.arange(len(scores)).tolist()
        xs = np.arange(ws - 1, len(scores)).tolist()
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
        smooth_line_shape = 'linear'
        word_starts = None

    discrete_name = '离散句' if is_sent else '离散词'
    fig = go.Figure()

    # 背景说话者离散点（每句话在主图 x 轴上插值定位）
    bg_turn_to_x = {}
    bg_spk = cache.get('bg_speaker', '')
    bg_results_c = cache.get('bg_results', [])
    bg_utts_c = cache.get('bg_utterances', [])
    if bg_spk and bg_results_c and bg_utts_c and len(bg_utts_c) >= ws:
        # bg 话语序列平滑（与主图同样方式）
        if is_sent:
            bg_utt_scores = np.array([r[dim] for r in bg_results_c])
        else:
            from collections import defaultdict as _dd
            _bg_grp = _dd(list)
            for r in bg_results_c:
                ti = r.get('turn_info')
                if ti: _bg_grp[ti['turn_index']].append(r[dim])
            bg_utt_scores = np.array([
                float(np.mean(_bg_grp[u['turn_index']])) if _bg_grp[u['turn_index']] else 0.0
                for u in bg_utts_c])
        bg_smooth_v = smooth_scores(bg_utt_scores, ws)

        # 按"紧随其后的主句"分组，聚集在该主句 x 起点
        main_turn_indices = [u['turn_index'] for u in utterances]
        from collections import defaultdict as _dd2
        bg_groups = _dd2(list)
        for j, bg_val in enumerate(bg_smooth_v):
            u_bg = bg_utts_c[j + ws - 1]
            T_bg = u_bg['turn_index']
            strat = u_bg.get('strategy') or 'Others'
            next_idx = next((i for i, T in enumerate(main_turn_indices) if T > T_bg),
                            len(main_turn_indices) - 1)
            bg_groups[next_idx].append((float(bg_val), T_bg, strat))

        from collections import defaultdict as _dd3
        strat_dots = _dd3(lambda: {'x': [], 'y': [], 'text': [], 'cd': []})
        for main_idx in sorted(bg_groups.keys()):
            if main_idx >= len(utterances): continue
            if is_sent:
                x_base = float(word_starts[main_idx])
            else:
                T_main = utterances[main_idx]['turn_index']
                x_base = next((float(i) for i, r in enumerate(results)
                               if r.get('turn_info') and r['turn_info']['turn_index'] == T_main), None)
                if x_base is None: continue
            for k, (bg_val, T_bg, strat) in enumerate(bg_groups[main_idx]):
                x_dot = x_base + k
                strat_dots[strat]['x'].append(x_dot)
                strat_dots[strat]['y'].append(bg_val)
                strat_dots[strat]['text'].append(f"T[{T_bg}] [{strat}] {dim[0].upper()}={bg_val:.3f}")
                strat_dots[strat]['cd'].append(T_bg)
                bg_turn_to_x[T_bg] = x_dot

        for strat, dots in strat_dots.items():
            sc = STRATEGY_COLORS.get(strat, '#BDBDBD')
            fig.add_trace(go.Scatter(
                x=dots['x'], y=dots['y'], mode='markers',
                name=strat,
                marker=dict(size=10, color=sc, opacity=0.75,
                            symbol='circle', line=dict(width=1, color='white')),
                text=dots['text'], hoverinfo='text',
                customdata=dots['cd']))

    fig.add_trace(go.Scatter(x=xd, y=scores.tolist(), mode='markers+lines', name=discrete_name,
        line=dict(dash='dot', color='rgba(150,150,150,0.5)'), marker=dict(size=6, color='gray'),
        text=ht_d, hoverinfo='text', customdata=cd_d))
    fig.add_trace(go.Scatter(x=xs, y=smooth.tolist(), mode='lines+markers',
        name=f'{dim.capitalize()} (W={ws})',
        line=dict(color=clr.get(dim, 'crimson'), width=3, shape=smooth_line_shape), marker=dict(size=4),
        text=ht_s, hoverinfo='text', customdata=cd_s))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
    # 拐点标记：竖线 + 可hover的散点（显示标签）
    if markers:
        y_max = 1.0
        mk_x, mk_y, mk_text, mk_color = [], [], [], []
        for m in markers:
            if m['speaker'] not in mf: continue
            mt = m['turn']; ms = m['speaker']; ml = m.get('label', '')
            if ms == bg_spk and mt in bg_turn_to_x:
                # 背景说话者：直接对齐到 bg 点的 x 位置
                mid_x = bg_turn_to_x[mt]
            elif is_sent and word_starts is not None:
                # 句粒度：用对应句的词中点定位
                sent_pos = [i for i, r in enumerate(results) if r.get('turn_info') and r['turn_info']['turn_index'] == mt]
                if not sent_pos:
                    all_t = [(i, r['turn_info']['turn_index']) for i, r in enumerate(results) if r.get('turn_info')]
                    if not all_t: continue
                    si = min(all_t, key=lambda t: abs(t[1] - mt))[0]
                else:
                    si = sent_pos[0]
                mid_x = word_starts[si] + (word_counts[si] - 1) // 2
            else:
                # 词粒度：用词索引定位
                x_pos = [i for i, r in enumerate(results) if r.get('turn_info') and r['turn_info']['turn_index'] == mt]
                if not x_pos:
                    all_t = [(i, r['turn_info']['turn_index']) for i, r in enumerate(results) if r.get('turn_info')]
                    if not all_t: continue
                    mid_x = min(all_t, key=lambda t: abs(t[1] - mt))[0]
                else:
                    mid_x = x_pos[len(x_pos) // 2]
            mc = MARKER_COLORS.get(ms, '#9c27b0')
            fig.add_vline(x=mid_x, line_dash='dash' if ms == 'seeker' else 'dot',
                line_color=mc, opacity=0.5, line_width=2)
            icon = MARKER_ICONS.get(ms, '📌')
            fig.add_annotation(x=mid_x, y=y_max, text=f"{icon}T{mt}", showarrow=False,
                font=dict(size=9, color=mc))
            mk_x.append(mid_x); mk_y.append(y_max)
            hover_label = f"<b>{icon} 轮次[{mt}] {ms}</b>"
            if ml: hover_label += f"<br>📝 {ml}"
            else: hover_label += "<br><i>(无标签)</i>"
            mk_text.append(hover_label); mk_color.append(mc)
        if mk_x:
            fig.add_trace(go.Scatter(x=mk_x, y=mk_y, mode='markers', name='拐点标记',
                marker=dict(size=12, color=mk_color, symbol='diamond', line=dict(width=1, color='white')),
                hovertext=mk_text, hoverinfo='text', showlegend=True))
    gran_tag = "句粒度" if is_sent else "词粒度"
    fig.update_layout(
        title=f"#{cache.get('conv_id','?')} | {cache.get('speaker','?')} | {dim.capitalize()} | {gran_tag}",
        xaxis_title="情感词索引", yaxis_title=f"{dim.capitalize()}",
        yaxis=dict(range=[-1, 1]),
        hovermode="closest", height=260, margin=dict(l=40, r=20, t=35, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    shown = len([m for m in (markers or []) if m['speaker'] in mf])
    return fig, f"✅ {len(results)}{unit_label} | W={ws} | 弧线{len(smooth)} | 拐点{shown}/{len(markers or [])} | {gran_tag}"


def _build_diff_figure(dim, ws, smooth_mode, cache):
    empty = go.Figure()
    empty.update_layout(title="需要选择单独说话者（Seeker 或 Supporter）",
                        height=220, margin=dict(l=40, r=20, t=35, b=25))
    if not cache or not cache.get('results') or not cache.get('bg_utterances'):
        return empty

    results = cache['results']
    utterances = cache.get('utterances', [])
    bg_utterances = cache.get('bg_utterances', [])
    gran = cache.get('granularity', 'word')
    is_sent = (gran == 'sentence')
    if not utterances or not bg_utterances:
        return empty

    # ---- seeker 每句平滑 VAD（与主图对齐）----
    scores = np.array([r[dim] for r in results])
    if smooth_mode == 'context' and is_sent and sent_predictor is not None:
        seeker_smooth_list = []
        for i in range(ws - 1, len(utterances)):
            combined = ' '.join(u['content'] for u in utterances[i - ws + 1: i + 1])
            preds = sent_predictor._predict_batch([combined])
            dim_idx = {'valence': 0, 'arousal': 1, 'dominance': 2}[dim]
            seeker_smooth_list.append((float(preds[0][dim_idx]) - 3.0) / 2.0)
        seeker_smooth = np.array(seeker_smooth_list)
    else:
        if is_sent:
            utt_scores = scores
        else:
            from collections import defaultdict as _dda
            grp = _dda(list)
            for r in results:
                ti = r.get('turn_info')
                if ti: grp[ti['turn_index']].append(r[dim])
            utt_scores = np.array([
                float(np.mean(grp[u['turn_index']])) if grp[u['turn_index']] else 0.0
                for u in utterances])
        if len(utt_scores) < ws:
            return empty
        seeker_smooth = smooth_scores(utt_scores, ws)
    valid_start = ws - 1

    # ---- x 坐标（与主图对齐）----
    if is_sent:
        wcs = [max(1, len(re.findall(r'\b[a-z]+\b', u['content'].lower()))) for u in utterances]
        ws_arr, cum = [], 0
        for c in wcs:
            ws_arr.append(cum); cum += c
        def xmid(i): return float(ws_arr[i] + (wcs[i] - 1) / 2)
        def xbg(i):  return float(ws_arr[i])   # 阶梯线从句子起点开始
    else:
        twr = {}
        for idx, r in enumerate(results):
            ti = r.get('turn_info')
            if ti:
                T = ti['turn_index']
                twr[T] = (twr[T][0] if T in twr else idx, idx)
        def xmid(i):
            T = utterances[i]['turn_index']
            lo, hi = twr.get(T, (i, i))
            return float((lo + hi) / 2)
        def xbg(i): return xmid(i)   # 词粒度无阶梯，中点即可

    # ---- 差值计算 ----
    seeker_turns = [u['turn_index'] for u in utterances]
    seeker_turn_set = set(seeker_turns)

    def block_vad(block):
        if not block: return None
        combined = ' '.join(u['content'] for u in block)
        if is_sent and sent_predictor is not None:
            preds = sent_predictor._predict_batch([combined])
            dim_idx = {'valence': 0, 'arousal': 1, 'dominance': 2}[dim]
            return (float(preds[0][dim_idx]) - 3.0) / 2.0
        vr = vad_extractor.extract(combined)
        return float(np.mean([r[dim] for r in vr])) if vr else None

    # 将 supporter 话语按"中间是否有 seeker 插话"切成连续块
    supp_blocks = []
    if bg_utterances:
        cur = [bg_utterances[0]]
        for u in bg_utterances[1:]:
            if any(cur[-1]['turn_index'] < st < u['turn_index'] for st in seeker_turn_set):
                supp_blocks.append(cur); cur = [u]
            else:
                cur.append(u)
        supp_blocks.append(cur)

    def nearest_prev_block(T):
        cands = [b for b in supp_blocks if b[-1]['turn_index'] < T]
        return max(cands, key=lambda b: b[-1]['turn_index']) if cands else None

    def nearest_next_block(T):
        cands = [b for b in supp_blocks if b[0]['turn_index'] > T]
        return min(cands, key=lambda b: b[0]['turn_index']) if cands else None

    dp_x, dp_y, dp_hover = [], [], []
    dn_x, dn_y, dn_hover = [], [], []
    bg_x, bg_y = [], []
    from collections import defaultdict as _ddb
    sp_dots = _ddb(lambda: {'x': [], 'y': []})
    sn_dots = _ddb(lambda: {'x': [], 'y': []})

    for j, sv in enumerate(seeker_smooth):
        i = j + valid_start
        if i >= len(utterances): continue
        T = seeker_turns[i]
        prev_blk = nearest_prev_block(T)
        next_blk = nearest_next_block(T)
        xm = xmid(i)
        bg_x.append(xbg(i)); bg_y.append(float(sv))

        pv = block_vad(prev_blk)
        if pv is not None:
            d = float(sv) - pv
            dp_x.append(xm); dp_y.append(d)
            strats = [u.get('strategy') or 'Others' for u in prev_blk]
            dp_hover.append(f"T[{T}] Δ{dim[0].upper()}={d:.3f}<br>策略: {', '.join(strats)}")
            for k, s in enumerate(dict.fromkeys(strats)):
                sp_dots[s]['x'].append(xm - (k + 1) * 1.0)
                sp_dots[s]['y'].append(d)

        nv = block_vad(next_blk)
        if nv is not None:
            d = float(sv) - nv
            dn_x.append(xm); dn_y.append(d)
            strats = [u.get('strategy') or 'Others' for u in next_blk]
            dn_hover.append(f"T[{T}] Δ{dim[0].upper()}={d:.3f}<br>策略: {', '.join(strats)}")
            for k, s in enumerate(dict.fromkeys(strats)):
                sn_dots[s]['x'].append(xm + (k + 1) * 1.0)
                sn_dots[s]['y'].append(d)

    # ---- 画图 ----
    clr = {'valence': 'crimson', 'arousal': 'darkorange', 'dominance': 'steelblue'}
    fig = go.Figure()
    # seeker 原始弧线（背景，浅色，双轴右侧）
    bg_shape = 'hv' if is_sent else 'linear'
    if bg_x:
        fig.add_trace(go.Scatter(x=bg_x, y=bg_y, mode='lines', name=f'{dim.capitalize()} (seeker)',
            line=dict(color=clr.get(dim, 'gray'), width=1.5, dash='dot', shape=bg_shape),
            opacity=0.3, yaxis='y2', hoverinfo='skip'))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
    if dp_x:
        fig.add_trace(go.Scatter(x=dp_x, y=dp_y, mode='lines+markers',
            name='seeker − prev_supp（被引导）',
            line=dict(color='#FF7043', width=2), marker=dict(size=5),
            text=dp_hover, hoverinfo='text'))
    if dn_x:
        fig.add_trace(go.Scatter(x=dn_x, y=dn_y, mode='lines+markers',
            name='seeker − next_supp（附和）',
            line=dict(color='#42A5F5', width=2), marker=dict(size=5),
            text=dn_hover, hoverinfo='text'))
    for s, d in sp_dots.items():
        fig.add_trace(go.Scatter(x=d['x'], y=d['y'], mode='markers', name=s,
            marker=dict(size=9, color=STRATEGY_COLORS.get(s, '#BDBDBD'),
                        symbol='triangle-left', opacity=0.85),
            hovertext=s, hoverinfo='text', showlegend=False))
    for s, d in sn_dots.items():
        fig.add_trace(go.Scatter(x=d['x'], y=d['y'], mode='markers', name=s,
            marker=dict(size=9, color=STRATEGY_COLORS.get(s, '#BDBDBD'),
                        symbol='triangle-right', opacity=0.85),
            hovertext=s, hoverinfo='text', showlegend=False))
    fig.update_layout(
        title=f"#{cache.get('conv_id','?')} | {dim.capitalize()} 差值分析",
        xaxis_title="情感词索引", yaxis_title=f"Δ{dim.capitalize()}",
        yaxis=dict(range=[-1, 1]),
        yaxis2=dict(range=[-1, 1], overlaying='y', side='right',
                    showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest", height=240, margin=dict(l=40, r=20, t=35, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


@app.callback(
    Output('graph-diff-valence', 'figure'),
    Output('graph-diff-arousal', 'figure'),
    Output('graph-diff-dominance', 'figure'),
    Input('window-slider', 'value'),
    Input('smooth-mode-radio', 'value'),
    Input('vad-cache-store', 'data'))
def update_diff_graphs(ws, smooth_mode, cache):
    figs = [_build_diff_figure(dim, ws, smooth_mode, cache)
            for dim in ['valence', 'arousal', 'dominance']]
    return figs[0], figs[1], figs[2]


@app.callback(
    Output('graph-valence', 'figure'),
    Output('graph-arousal', 'figure'),
    Output('graph-dominance', 'figure'),
    Output('status-info', 'children'),
    Input('window-slider', 'value'),
    Input('smooth-mode-radio', 'value'),
    Input('vad-cache-store', 'data'),
    Input('markers-store', 'data'),
    Input('marker-filter', 'value'))
def update_graphs(ws, smooth_mode, cache, markers, mf):
    figs, status = [], "暂无数据"
    for dim in ['valence', 'arousal', 'dominance']:
        fig, status = _build_figure(dim, ws, smooth_mode, cache, markers, mf)
        figs.append(fig)
    return figs[0], figs[1], figs[2], status


@app.callback(
    Output('dialog-panel', 'children'),
    Input('graph-valence', 'hoverData'),
    Input('graph-arousal', 'hoverData'),
    Input('graph-dominance', 'hoverData'),
    Input('vad-cache-store', 'data'),
    Input('markers-store', 'data'))
def on_hover_dialog(hoverV, hoverA, hoverD, cache, markers):
    hoverData = hoverV or hoverA or hoverD
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

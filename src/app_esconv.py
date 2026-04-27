import os, sys, json, re
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dash
from dash import html, dcc, Input, Output, State, ALL, ctx
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from scipy.stats import chi2, gaussian_kde
from src.vad_extractor import VADExtractor, SentenceVADPredictor
from src.emotion_smoothing import smooth_scores
from src.esconv_loader import ESConvLoader

LEXICON_PATH = "NRC-VAD-Lexicon-v2.1.txt"
ESCONV_PATH = "ESConv-strategy.json"
CACHE_DIR = "src/cache"
SYNC_RANGE_DIR = os.path.join(CACHE_DIR, "sync_ranges")
DEFAULT_DIFF_CACHE_DIR = os.path.join(CACHE_DIR, "default_diff_arcs")
MARKERS_FILE = os.path.join(CACHE_DIR, "markers.json")
AUTO_MARKERS_FILE = os.path.join(CACHE_DIR, "auto_markers.json")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SYNC_RANGE_DIR, exist_ok=True)
os.makedirs(DEFAULT_DIFF_CACHE_DIR, exist_ok=True)

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

MARKER_COLORS = {'seeker': '#4CAF50', 'supporter': '#64B5F6'}
MARKER_ICONS = {'seeker': '🟢', 'supporter': '🔵'}
DIFF_RELATION_COLORS = {'prev': '#FF7043', 'next': 'rgba(66,165,245,0.55)'}
DIM_COLORS = {'valence': 'crimson', 'arousal': 'darkorange', 'dominance': 'steelblue'}
DIM_SHORT = {'valence': 'V', 'arousal': 'A', 'dominance': 'D'}
RELATION_LABELS = {'prev': '上次supporter', 'next': '下次supporter'}
TREND_LABELS = {'rise': '升段起点', 'fall': '降段起点'}
SYNC_CONFIDENCE = 2.0 / 3.0
SYNC_PLOT_MAX_POINTS = 2500
SYNC_CACHE_VERSION = 5
SYNC_ELLIPSOID_RESOLUTION = 22
SYNC_DEFAULT_WINDOW_SIZE = 2
SYNC_DEFAULT_GRANULARITY = 'sentence'
SYNC_DEFAULT_SMOOTH_MODE = 'context'
AUTO_TREND_SKIP_POINTS = 2
AUTO_TREND_MIN_DELTA = 0.1
SYNC_KDE_GRID_SIZE = 55
SYNC_KDE_PAIRS = [
    ('valence', 'arousal', 0, 1, 'ΔValence', 'ΔArousal'),
    ('valence', 'dominance', 0, 2, 'ΔValence', 'ΔDominance'),
    ('arousal', 'dominance', 1, 2, 'ΔArousal', 'ΔDominance'),
]
SHOW_MODAL_STYLE = {
    'display': 'flex', 'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '100%',
    'backgroundColor': 'rgba(0,0,0,0.4)', 'zIndex': 9999,
    'justifyContent': 'center', 'alignItems': 'center'
}
HIDE_MODAL_STYLE = {'display': 'none'}
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

SYNC_RANGE_MEMORY_CACHE = {}
SYNC_VIEW_CACHE_MEMORY = {}
SYNC_CURRENT_POINTS_MEMORY_CACHE = {}
DEFAULT_DIFF_MEMORY_CACHE = {}
ALL_SYNC_RATE_MEMORY_CACHE = {}

# 存储格式: { "conv_0": [{"turn":3,"speaker":"seeker","label":"..."}, ...] }
def _ck(cid): return f"conv_{cid}"

def _load_json_dict(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_all_markers():
    return _load_json_dict(MARKERS_FILE)

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

def load_all_auto_markers():
    return _load_json_dict(AUTO_MARKERS_FILE)

def save_all_auto_markers(data):
    with open(AUTO_MARKERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_auto_marker_labels(cid):
    return load_all_auto_markers().get(_ck(cid), {})

def save_auto_marker_label(cid, marker_id, label):
    data = load_all_auto_markers()
    key = _ck(cid)
    labels = data.get(key, {})
    if label:
        labels[marker_id] = label
    else:
        labels.pop(marker_id, None)
    data[key] = labels
    save_all_auto_markers(data)

def delete_auto_marker_label(cid, marker_id):
    data = load_all_auto_markers()
    key = _ck(cid)
    labels = data.get(key, {})
    labels.pop(marker_id, None)
    data[key] = labels
    save_all_auto_markers(data)

def _build_meta_info(meta, conv_id):
    return html.Div([
        html.Span(f"🆔 {meta.get('id', conv_id)}", style={'marginRight': '15px'}),
        html.Span(f"😟 {meta.get('emotion_type', 'N/A')}", style={'marginRight': '15px'}),
        html.Span(f"📋 {meta.get('problem_type', 'N/A')}", style={'marginRight': '15px'}),
        html.Span(f"🔥 初始: {meta.get('initial_emotion_intensity', 'N/A')}",
                  style={'marginRight': '15px', 'color': '#d32f2f'}),
        html.Span(f"🌿 最终: {meta.get('final_emotion_intensity', 'N/A')}", style={'color': '#388e3c'}),
        html.Br(),
        html.Span(f"📝 {meta.get('situation', 'N/A')}", style={'fontStyle': 'italic'})
    ])

def _compute_vad_results(utterances, granularity):
    if granularity == 'sentence' and sent_predictor is not None:
        return sent_predictor.predict_utterances(utterances), 'sentence'
    text = esconv_loader.utterances_to_text(utterances)
    vad_r = vad_extractor.extract(text)
    tm = esconv_loader.build_turn_mapping(utterances, vad_r)
    for i, v in enumerate(vad_r):
        v['turn_info'] = tm[i] if i < len(tm) and tm[i] else None
    return vad_r, 'word'

def _vad_cache_path(conv_id, speaker, granularity):
    suffix = '_sent' if granularity == 'sentence' else ''
    return os.path.join(CACHE_DIR, f"vad_conv{conv_id}_{speaker}{suffix}.json")

def _load_vad_cache(conv_id, speaker, granularity):
    cache_path = _vad_cache_path(conv_id, speaker, granularity)
    if not os.path.exists(cache_path):
        return None
    try:
        results, metadata = vad_extractor.load_cache(cache_path)
    except Exception:
        return None
    cached_gran = metadata.get('granularity')
    if cached_gran and cached_gran != granularity:
        return None
    return results

def build_conversation_cache(conv_id, speaker, granularity, persist=True):
    conv = esconv_loader.get_conversation(conv_id)
    if not conv:
        return None, None
    meta = conv.get('meta', {})
    dialog = conv.get('dialog', [])
    utterances = esconv_loader.filter_utterances(dialog, speaker)
    if not utterances:
        return {
            'conv_id': conv_id, 'speaker': speaker, 'granularity': granularity,
            'results': [], 'dialog': dialog, 'utterances': []
        }, meta

    cached_results = _load_vad_cache(conv_id, speaker, granularity)
    if cached_results is not None:
        vad_results, actual_gran = cached_results, granularity
    else:
        vad_results, actual_gran = _compute_vad_results(utterances, granularity)
    if persist:
        cp = _vad_cache_path(conv_id, speaker, actual_gran)
        vad_extractor.save_cache(
            vad_results, cp,
            metadata={'conv_id': conv_id, 'speaker': speaker, 'granularity': actual_gran})

    cache = {
        'conv_id': conv_id, 'speaker': speaker, 'granularity': actual_gran,
        'results': vad_results, 'dialog': dialog, 'utterances': utterances
    }
    bg_spk = 'supporter' if speaker == 'seeker' else ('seeker' if speaker == 'supporter' else None)
    if bg_spk:
        bg_utts = esconv_loader.filter_utterances(dialog, bg_spk)
        if bg_utts:
            cached_bg_results = _load_vad_cache(conv_id, bg_spk, granularity)
            if cached_bg_results is not None:
                bg_results = cached_bg_results
            else:
                bg_results, _ = _compute_vad_results(bg_utts, granularity)
            cache['bg_speaker'] = bg_spk
            cache['bg_results'] = bg_results
            cache['bg_utterances'] = bg_utts
    return cache, meta

def _utterance_scores_from_results(results, utterances, dim, is_sent):
    if is_sent:
        return np.array([r[dim] for r in results], dtype=float)
    groups = defaultdict(list)
    for r in results:
        ti = r.get('turn_info')
        if ti:
            groups[ti['turn_index']].append(r[dim])
    return np.array([
        float(np.mean(groups[u['turn_index']])) if groups[u['turn_index']] else 0.0
        for u in utterances
    ], dtype=float)

def _score_text_block(text, dim, is_sent):
    if not text.strip():
        return None
    if is_sent and sent_predictor is not None:
        preds = sent_predictor._predict_batch([text])
        dim_idx = {'valence': 0, 'arousal': 1, 'dominance': 2}[dim]
        return (float(preds[0][dim_idx]) - 3.0) / 2.0
    wr = vad_extractor.extract(text)
    return float(np.mean([r[dim] for r in wr])) if wr else None

def _score_utterance_block(block, dim, is_sent):
    if not block:
        return None
    combined = ' '.join(u['content'] for u in block)
    return _score_text_block(combined, dim, is_sent)

def _compute_smoothed_utterance_curve(dim, ws, smooth_mode, cache):
    results = cache.get('results', [])
    utterances = cache.get('utterances', [])
    is_sent = (cache.get('granularity') == 'sentence')
    if not results or not utterances:
        return None, ws - 1
    use_context = (smooth_mode == 'context' and is_sent)
    if use_context:
        if len(utterances) < ws:
            return None, ws - 1
        ctx_scores = []
        for i in range(ws - 1, len(utterances)):
            combined = ' '.join(u['content'] for u in utterances[i - ws + 1:i + 1])
            val = _score_text_block(combined, dim, True)
            ctx_scores.append(0.0 if val is None else float(val))
        return np.array(ctx_scores, dtype=float), ws - 1

    utt_scores = _utterance_scores_from_results(results, utterances, dim, is_sent)
    if len(utt_scores) < ws:
        return None, ws - 1
    return smooth_scores(utt_scores, ws), ws - 1

def _compute_default_main_context_bundle(dim, cache):
    utterances = cache.get('utterances', [])
    results = cache.get('results', [])
    if not utterances or not results:
        return None
    word_counts = [max(1, len(re.findall(r'\b[a-z]+\b', u['content'].lower()))) for u in utterances]
    word_starts = []
    cum = 0
    for c in word_counts:
        word_starts.append(cum)
        cum += c

    scores = np.array([r[dim] for r in results], dtype=float)
    utt_turns = [r['turn_info']['turn_index'] if r.get('turn_info') else -1 for r in results]
    xd_utt = [word_starts[i] + (word_counts[i] - 1) // 2 for i in range(len(results))]

    ctx_scores, ctx_turns, xs_ctx = [], [], []
    for i in range(SYNC_DEFAULT_WINDOW_SIZE - 1, len(utterances)):
        combined = ' '.join(u['content'] for u in utterances[i - SYNC_DEFAULT_WINDOW_SIZE + 1:i + 1])
        val = _score_text_block(combined, dim, True)
        ctx_scores.append(0.0 if val is None else float(val))
        ctx_turns.append(utterances[i]['turn_index'])
        xs_ctx.append(word_starts[i])

    return {
        'utt_x': xd_utt,
        'utt_y': scores.tolist(),
        'utt_turns': utt_turns,
        'ctx_x': xs_ctx,
        'ctx_y': ctx_scores,
        'ctx_turns': ctx_turns,
    }

def _build_turn_x_helpers(results, utterances, is_sent):
    if is_sent:
        word_counts = [max(1, len(re.findall(r'\b[a-z]+\b', u['content'].lower()))) for u in utterances]
        word_starts, cum = [], 0
        for c in word_counts:
            word_starts.append(cum)
            cum += c

        def xmid(i): return float(word_starts[i] + (word_counts[i] - 1) / 2.0)
        def xbg(i): return float(word_starts[i])
        def xspan(i): return float(word_counts[i])

        return (
            xmid,
            xbg,
            xspan,
            {u['turn_index']: xmid(i) for i, u in enumerate(utterances)},
            {u['turn_index']: xspan(i) for i, u in enumerate(utterances)},
        )

    turn_word_ranges = {}
    for idx, r in enumerate(results):
        ti = r.get('turn_info')
        if ti:
            turn_idx = ti['turn_index']
            turn_word_ranges[turn_idx] = (
                turn_word_ranges[turn_idx][0] if turn_idx in turn_word_ranges else idx, idx)

    def xmid(i):
        turn_idx = utterances[i]['turn_index']
        lo, hi = turn_word_ranges.get(turn_idx, (i, i))
        return float((lo + hi) / 2.0)

    def xbg(i):
        turn_idx = utterances[i]['turn_index']
        lo, _ = turn_word_ranges.get(turn_idx, (i, i))
        return float(lo)
    def xspan(i):
        turn_idx = utterances[i]['turn_index']
        lo, hi = turn_word_ranges.get(turn_idx, (i, i))
        return float(max(1, hi - lo + 1))

    return (
        xmid,
        xbg,
        xspan,
        {u['turn_index']: xmid(i) for i, u in enumerate(utterances)},
        {u['turn_index']: xspan(i) for i, u in enumerate(utterances)},
    )

def _interpolate_turn_x(turn_to_x, turn_idx):
    if turn_idx in turn_to_x:
        return turn_to_x[turn_idx]
    keys = sorted(turn_to_x.keys())
    if not keys:
        return None
    lo = max((k for k in keys if k <= turn_idx), default=keys[0])
    hi = min((k for k in keys if k >= turn_idx), default=keys[-1])
    if lo == hi:
        return turn_to_x[lo]
    return turn_to_x[lo] + (turn_idx - lo) / (hi - lo) * (turn_to_x[hi] - turn_to_x[lo])

def _build_supporter_blocks(bg_utterances, seeker_turns):
    seeker_turn_set = set(seeker_turns)
    supp_blocks = []
    if not bg_utterances:
        return supp_blocks
    current = [bg_utterances[0]]
    for utt in bg_utterances[1:]:
        if any(current[-1]['turn_index'] < st < utt['turn_index'] for st in seeker_turn_set):
            supp_blocks.append(current)
            current = [utt]
        else:
            current.append(utt)
    supp_blocks.append(current)
    return supp_blocks

def _default_diff_cache_path(conv_id):
    return os.path.join(DEFAULT_DIFF_CACHE_DIR, f"default_diff_conv{conv_id}.json")

def _serialize_default_diff_bundle(bundle):
    return bundle

def _restore_default_diff_bundle(bundle):
    return bundle

def _is_default_diff_mode(cache, ws, smooth_mode):
    if not cache:
        return False
    return (
        cache.get('speaker') == 'seeker' and
        cache.get('granularity') == SYNC_DEFAULT_GRANULARITY and
        int(ws) == SYNC_DEFAULT_WINDOW_SIZE and
        smooth_mode == SYNC_DEFAULT_SMOOTH_MODE
    )

def _compute_default_diff_bundle(conv_id, persist=True):
    conv_cache, _ = build_conversation_cache(conv_id, 'seeker', SYNC_DEFAULT_GRANULARITY, persist=persist)
    if not conv_cache or not conv_cache.get('bg_utterances'):
        return None
    dims = {}
    main_context = {}
    for dim in ['valence', 'arousal', 'dominance']:
        series = _compute_diff_series(dim, SYNC_DEFAULT_WINDOW_SIZE, SYNC_DEFAULT_SMOOTH_MODE, conv_cache)
        if series is None:
            return None
        dims[dim] = series
        main_context[dim] = _compute_default_main_context_bundle(dim, conv_cache)

    current = _compute_current_sync_points_fresh(conv_cache, SYNC_DEFAULT_WINDOW_SIZE, SYNC_DEFAULT_SMOOTH_MODE)
    if current is None:
        return None

    return {
        'conv_id': conv_id,
        'speaker': 'seeker',
        'granularity': SYNC_DEFAULT_GRANULARITY,
        'smooth_mode': SYNC_DEFAULT_SMOOTH_MODE,
        'window_size': SYNC_DEFAULT_WINDOW_SIZE,
        'dims': dims,
        'main_context': main_context,
        'current_sync': {
            'points': current['points'].tolist(),
            'turns': list(current['turns']),
            'utterance_spans': current['utterance_spans'].tolist(),
        },
    }

def _load_default_diff_bundle(conv_id, compute_if_missing=True):
    if conv_id in DEFAULT_DIFF_MEMORY_CACHE:
        return DEFAULT_DIFF_MEMORY_CACHE[conv_id]
    cache_path = _default_diff_cache_path(conv_id)
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            bundle = _restore_default_diff_bundle(json.load(f))
        DEFAULT_DIFF_MEMORY_CACHE[conv_id] = bundle
        return bundle
    if not compute_if_missing:
        return None
    bundle = _compute_default_diff_bundle(conv_id, persist=True)
    if bundle is None:
        return None
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(_serialize_default_diff_bundle(bundle), f, ensure_ascii=False, indent=2)
    DEFAULT_DIFF_MEMORY_CACHE[conv_id] = bundle
    return bundle

def _compute_diff_series(dim, ws, smooth_mode, cache):
    empty = {'prev': None, 'next': None}
    if not cache or not cache.get('results') or not cache.get('bg_utterances'):
        return None

    utterances = cache.get('utterances', [])
    bg_utterances = cache.get('bg_utterances', [])
    results = cache.get('results', [])
    if not utterances or not bg_utterances:
        return None

    seeker_smooth, valid_start = _compute_smoothed_utterance_curve(dim, ws, smooth_mode, cache)
    if seeker_smooth is None or len(seeker_smooth) == 0:
        return None

    is_sent = (cache.get('granularity') == 'sentence')
    xmid, xbg, _xspan, turn_to_x, _turn_to_width = _build_turn_x_helpers(cache.get('results', []), utterances, is_sent)
    seeker_raw_scores = _utterance_scores_from_results(results, utterances, dim, is_sent)
    seeker_turns = [u['turn_index'] for u in utterances]
    supp_blocks = _build_supporter_blocks(bg_utterances, seeker_turns)

    def nearest_prev_block(turn_idx):
        cands = [b for b in supp_blocks if b[-1]['turn_index'] < turn_idx]
        return max(cands, key=lambda b: b[-1]['turn_index']) if cands else None

    def nearest_next_block(turn_idx):
        cands = [b for b in supp_blocks if b[0]['turn_index'] > turn_idx]
        return min(cands, key=lambda b: b[0]['turn_index']) if cands else None

    rel_data = {
        'prev': {'x': [], 'y': [], 'turns': [], 'hover': [], 'strategies': [], 'strategy_summary': []},
        'next': {'x': [], 'y': [], 'turns': [], 'hover': [], 'strategies': [], 'strategy_summary': []},
    }
    strategy_dots = {
        'prev': defaultdict(lambda: {'x': [], 'y': [], 'hover': [], 'turns': []}),
        'next': defaultdict(lambda: {'x': [], 'y': [], 'hover': [], 'turns': []}),
    }
    seeker_curve = {'x': [], 'y': [], 'turns': []}
    for utt_idx, utt in enumerate(utterances):
        if utt_idx >= len(seeker_raw_scores):
            break
        seeker_curve['x'].append(xbg(utt_idx))
        seeker_curve['y'].append(float(seeker_raw_scores[utt_idx]))
        seeker_curve['turns'].append(utt['turn_index'])

    for j, sv in enumerate(seeker_smooth):
        utt_idx = j + valid_start
        if utt_idx >= len(utterances):
            continue
        seeker_turn = seeker_turns[utt_idx]
        prev_blk = nearest_prev_block(seeker_turn)
        next_blk = nearest_next_block(seeker_turn)
        xm = xmid(utt_idx)

        prev_turn = seeker_turns[utt_idx - 1] if utt_idx > 0 else -1
        next_turn = seeker_turns[utt_idx + 1] if utt_idx < len(seeker_turns) - 1 else float('inf')

        for relation, block, immediate in [
            ('prev', prev_blk, prev_blk is not None and all(prev_turn < u['turn_index'] < seeker_turn for u in prev_blk)),
            ('next', next_blk, next_blk is not None and all(seeker_turn < u['turn_index'] < next_turn for u in next_blk)),
        ]:
            other_score = _score_utterance_block(block, dim, is_sent)
            if other_score is None:
                continue
            delta = float(sv) - other_score
            strategies = [u.get('strategy') or 'Others' for u in block]
            strategy_summary = ', '.join(dict.fromkeys(strategies)) if strategies else 'Others'
            span = f"T[{block[0]['turn_index']}~{block[-1]['turn_index']}]"
            rel_data[relation]['x'].append(xm)
            rel_data[relation]['y'].append(delta)
            rel_data[relation]['turns'].append(seeker_turn)
            rel_data[relation]['strategies'].append(strategies)
            rel_data[relation]['strategy_summary'].append(strategy_summary)
            rel_data[relation]['hover'].append(
                f"T[{seeker_turn}] Δ{DIM_SHORT[dim]}={delta:.3f}<br>"
                f"{RELATION_LABELS[relation]}: {strategy_summary}<br>supporter段: {span}")
            if immediate:
                for offset_idx, strategy in enumerate(dict.fromkeys(strategies)):
                    x_offset = xm - (offset_idx + 1) * 1.0 if relation == 'prev' else xm + (offset_idx + 1) * 1.0
                    strategy_dots[relation][strategy]['x'].append(x_offset)
                    strategy_dots[relation][strategy]['y'].append(delta)
                    strategy_dots[relation][strategy]['hover'].append(
                        f"T[{seeker_turn}] | {RELATION_LABELS[relation]}策略: {strategy}")
                    strategy_dots[relation][strategy]['turns'].append(seeker_turn)

    return {
        'is_sent': is_sent,
        'turn_to_x': turn_to_x,
        'seeker_curve': seeker_curve,
        'prev': rel_data['prev'],
        'next': rel_data['next'],
        'strategy_prev': strategy_dots['prev'],
        'strategy_next': strategy_dots['next'],
    }

def _build_auto_diff_markers(dim, relation, relation_data, conv_id):
    if not relation_data or len(relation_data.get('y', [])) < 2:
        return []
    labels = get_auto_marker_labels(conv_id)
    xs = relation_data['x']
    ys = relation_data['y']
    turns = relation_data['turns']
    ys_arr = np.asarray(ys, dtype=float)
    markers = []
    start_idx = AUTO_TREND_SKIP_POINTS + 1
    end_exclusive = max(start_idx, len(ys_arr) - 2)
    prev_dir = 0
    for idx in range(1, min(start_idx, len(ys_arr))):
        diff = ys_arr[idx] - ys_arr[idx - 1]
        if abs(diff) < AUTO_TREND_MIN_DELTA:
            continue
        if diff > 0:
            prev_dir = 1
        elif diff < 0:
            prev_dir = -1

    for marker_idx in range(start_idx, end_exclusive):
        diff = ys_arr[marker_idx] - ys_arr[marker_idx - 1]
        if abs(diff) < AUTO_TREND_MIN_DELTA:
            continue
        cur_dir = 1 if diff > 0 else -1
        if prev_dir == 0:
            prev_dir = cur_dir
            continue
        if cur_dir == prev_dir:
            prev_dir = cur_dir
            continue

        trend = 'rise' if cur_dir > 0 else 'fall'
        strategy_summary = (
            relation_data['strategy_summary'][marker_idx]
            if marker_idx < len(relation_data['strategy_summary']) else '')
        default_label = f"{DIM_SHORT[dim]} {TREND_LABELS[trend]}"
        if strategy_summary:
            default_label += f" | {RELATION_LABELS[relation]}: {strategy_summary}"
        marker_id = f"{relation}:{dim}:{trend}:{marker_idx}"
        label = labels.get(marker_id) or default_label
        markers.append({
            'marker_id': marker_id,
            'relation': relation,
            'dim': dim,
            'trend': trend,
            'point_index': marker_idx,
            'x': xs[marker_idx],
            'y': ys[marker_idx],
            'turn': turns[marker_idx],
            'label': label,
            'default_label': default_label,
            'slope': float(diff),
        })
        prev_dir = cur_dir
    return markers

def _parse_turn_from_customdata(customdata):
    if isinstance(customdata, dict):
        return customdata.get('turn')
    if isinstance(customdata, (int, np.integer)):
        return int(customdata)
    if isinstance(customdata, float) and customdata >= 0 and customdata.is_integer():
        return int(customdata)
    return None

def _sample_points(points, max_points=SYNC_PLOT_MAX_POINTS):
    if points is None or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]

def _sync_cache_key(tail_ratio, ws, smooth_mode, granularity):
    pct = int(round(float(tail_ratio) * 100))
    return f"v{SYNC_CACHE_VERSION}_tail{pct}_ws{int(ws)}_{smooth_mode}_{granularity}"

def _sync_cache_path(tail_ratio, ws, smooth_mode, granularity):
    return os.path.join(SYNC_RANGE_DIR, f"{_sync_cache_key(tail_ratio, ws, smooth_mode, granularity)}.json")

def _serialize_sync_dataset(result):
    serializable = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif key == 'kde_data' and isinstance(value, dict):
            serializable[key] = {}
            for pair_key, pair_data in value.items():
                serializable[key][pair_key] = {}
                for sub_key, sub_value in pair_data.items():
                    serializable[key][pair_key][sub_key] = (
                        sub_value.tolist() if isinstance(sub_value, np.ndarray) else sub_value)
        else:
            serializable[key] = value
    return serializable

def _restore_sync_dataset(data):
    restored = dict(data)
    for key in ['sample_points', 'mean', 'cov', 'low', 'high']:
        if key in restored and restored[key] is not None:
            restored[key] = np.asarray(restored[key], dtype=float)
    if 'kde_data' in restored and isinstance(restored['kde_data'], dict):
        restored_kde = {}
        for pair_key, pair_data in restored['kde_data'].items():
            restored_kde[pair_key] = {}
            for sub_key, sub_value in pair_data.items():
                if sub_key in {'xs', 'ys', 'z'} and sub_value is not None:
                    restored_kde[pair_key][sub_key] = np.asarray(sub_value, dtype=float)
                else:
                    restored_kde[pair_key][sub_key] = sub_value
        restored['kde_data'] = restored_kde
    return restored

def _load_sync_dataset_from_disk(tail_ratio, ws, smooth_mode, granularity):
    cache_path = _sync_cache_path(tail_ratio, ws, smooth_mode, granularity)
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'r', encoding='utf-8') as f:
        return _restore_sync_dataset(json.load(f))

def _save_sync_dataset_to_disk(tail_ratio, ws, smooth_mode, granularity, result):
    cache_path = _sync_cache_path(tail_ratio, ws, smooth_mode, granularity)
    serializable = _serialize_sync_dataset(result)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

def _build_gaussian_ellipsoid(mean, cov, chi2_threshold, confidence_ratio):
    cov = np.asarray(cov, dtype=float)
    mean = np.asarray(mean, dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    radii = np.sqrt(vals * chi2_threshold)

    u = np.linspace(0.0, 2.0 * np.pi, SYNC_ELLIPSOID_RESOLUTION)
    v = np.linspace(0.0, np.pi, SYNC_ELLIPSOID_RESOLUTION)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([xs, ys, zs], axis=-1)
    transform = vecs @ np.diag(radii)
    ellipsoid = np.einsum('...i,ij->...j', sphere, transform.T) + mean
    return go.Surface(
        x=ellipsoid[:, :, 0], y=ellipsoid[:, :, 1], z=ellipsoid[:, :, 2],
        name=f'{int(round(confidence_ratio * 100))}% 高斯椭球',
        opacity=0.18, showscale=False,
        colorscale=[[0.0, '#66BB6A'], [1.0, '#2E7D32']],
        hoverinfo='skip')

def _mahalanobis_inside(points, mean, cov, chi2_threshold):
    centered = np.asarray(points, dtype=float) - np.asarray(mean, dtype=float)
    cov_inv = np.linalg.pinv(np.asarray(cov, dtype=float))
    dist2 = np.einsum('...i,ij,...j->...', centered, cov_inv, centered)
    return dist2 <= chi2_threshold, dist2

def _build_sync_kde_cache(sample_points):
    sampled = np.asarray(sample_points, dtype=float)
    kde_data = {}
    for key_x, key_y, x_idx, y_idx, x_label, y_label in SYNC_KDE_PAIRS:
        pair_key = f'{key_x}_{key_y}'
        xs = np.linspace(-1.0, 1.0, SYNC_KDE_GRID_SIZE)
        ys = np.linspace(-1.0, 1.0, SYNC_KDE_GRID_SIZE)
        grid_x, grid_y = np.meshgrid(xs, ys)
        density = None
        try:
            proj = sampled[:, [x_idx, y_idx]].T
            kde = gaussian_kde(proj)
            density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
        except Exception:
            density = None
        kde_data[pair_key] = {
            'xs': xs,
            'ys': ys,
            'z': density,
            'x_idx': x_idx,
            'y_idx': y_idx,
            'x_label': x_label,
            'y_label': y_label,
        }
    return kde_data

def _build_sync_kde_figure(kde_data, current_points, inside, turns, cache, tail_pct):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{x_label} vs {y_label}' for _, _, _, _, x_label, y_label in SYNC_KDE_PAIRS],
        horizontal_spacing=0.08)

    current = np.asarray(current_points, dtype=float)
    inside = np.asarray(inside, dtype=bool)

    for col, (key_x, key_y, x_idx, y_idx, x_label, y_label) in enumerate(SYNC_KDE_PAIRS, start=1):
        pair_key = f'{key_x}_{key_y}'
        pair = kde_data.get(pair_key, {}) if isinstance(kde_data, dict) else {}
        density = pair.get('z')
        xs = pair.get('xs')
        ys = pair.get('ys')
        if density is not None and xs is not None and ys is not None:
            fig.add_trace(go.Contour(
                x=xs, y=ys, z=density,
                colorscale='YlGnBu',
                contours=dict(showlabels=False),
                line=dict(width=0.5),
                opacity=0.85,
                showscale=False,
                hovertemplate=f'{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>density=%{{z:.4f}}<extra></extra>',
                name=f'{x_label}/{y_label} KDE'),
                row=1, col=col)

        fig.add_trace(go.Scatter(
            x=current[:, x_idx], y=current[:, y_idx],
            mode='lines', name='当前轨迹',
            line=dict(color='#455A64', width=2),
            hovertext=[
                f"T[{turns[i]}] {x_label}={current[i, x_idx]:.3f}<br>{y_label}={current[i, y_idx]:.3f}"
                for i in range(len(current))
            ],
            hoverinfo='text',
            showlegend=(col == 1)),
            row=1, col=col)
        if np.any(inside):
            fig.add_trace(go.Scatter(
                x=current[inside, x_idx], y=current[inside, y_idx],
                mode='markers', name='同步区内',
                marker=dict(size=7, color='#2E7D32', opacity=0.95),
                hovertext=[f"T[{turns[i]}] 同步区内" for i in np.where(inside)[0]],
                hoverinfo='text',
                showlegend=(col == 1)),
                row=1, col=col)
        if np.any(~inside):
            fig.add_trace(go.Scatter(
                x=current[~inside, x_idx], y=current[~inside, y_idx],
                mode='markers', name='同步区外',
                marker=dict(size=7, color='#C62828', opacity=0.9),
                hovertext=[f"T[{turns[i]}] 同步区外" for i in np.where(~inside)[0]],
                hoverinfo='text',
                showlegend=(col == 1)),
                row=1, col=col)

        fig.update_xaxes(title_text=x_label, range=[-1, 1], row=1, col=col)
        fig.update_yaxes(title_text=y_label, range=[-1, 1], row=1, col=col)

    fig.update_layout(
        title=f"#{cache.get('conv_id', '?')} | 同步范围 KDE 投影 | 尾段 {tail_pct}%",
        height=420,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0))
    return fig

def _compute_sync_dataset(tail_ratio, ws, smooth_mode, granularity, compute_if_missing=True):
    actual_mode = 'context' if smooth_mode == 'context' and granularity == 'sentence' else 'avg'
    cache_key = (round(float(tail_ratio), 4), int(ws), actual_mode, granularity)
    if cache_key in SYNC_RANGE_MEMORY_CACHE:
        return SYNC_RANGE_MEMORY_CACHE[cache_key]
    disk_cached = _load_sync_dataset_from_disk(tail_ratio, ws, actual_mode, granularity)
    if disk_cached is not None:
        SYNC_RANGE_MEMORY_CACHE[cache_key] = disk_cached
        return disk_cached
    if not compute_if_missing:
        return None

    dataset_points = []
    used_conversations = 0
    for conv_id in conv_ids:
        conv_cache, _ = build_conversation_cache(conv_id, 'seeker', granularity, persist=False)
        if not conv_cache or not conv_cache.get('bg_utterances'):
            continue
        series = {
            dim: _compute_diff_series(dim, ws, actual_mode, conv_cache)
            for dim in ['valence', 'arousal', 'dominance']
        }
        if any(series[dim] is None or len(series[dim]['prev']['y']) == 0 for dim in series):
            continue
        sizes = [len(series[dim]['prev']['y']) for dim in series]
        size = min(sizes)
        if size <= 0:
            continue
        if tail_ratio <= 0:
            start_idx = 0
        else:
            tail_count = max(1, int(np.ceil(size * tail_ratio)))
            start_idx = max(0, size - tail_count)
        stacked = np.column_stack([
            np.asarray(series['valence']['prev']['y'][:size], dtype=float)[start_idx:],
            np.asarray(series['arousal']['prev']['y'][:size], dtype=float)[start_idx:],
            np.asarray(series['dominance']['prev']['y'][:size], dtype=float)[start_idx:],
        ])
        if len(stacked) == 0:
            continue
        dataset_points.append(stacked)
        used_conversations += 1

    if not dataset_points:
        return None

    points = np.vstack(dataset_points)
    alpha = (1.0 - SYNC_CONFIDENCE) / 2.0
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False) if len(points) > 1 else np.eye(3) * 1e-6
    cov = np.asarray(cov, dtype=float) + np.eye(3) * 1e-6
    chi2_threshold = float(chi2.ppf(SYNC_CONFIDENCE, df=3))
    result = {
        'sample_points': _sample_points(points),
        'mean': mean,
        'cov': cov,
        'chi2_threshold': chi2_threshold,
        'low': np.quantile(points, alpha, axis=0),
        'high': np.quantile(points, 1.0 - alpha, axis=0),
        'used_conversations': used_conversations,
        'point_count': int(len(points)),
        'tail_ratio': tail_ratio,
        'confidence': SYNC_CONFIDENCE,
        'granularity': granularity,
        'smooth_mode': actual_mode,
    }
    result['kde_data'] = _build_sync_kde_cache(result['sample_points'])
    SYNC_RANGE_MEMORY_CACHE[cache_key] = result
    _save_sync_dataset_to_disk(tail_ratio, ws, actual_mode, granularity, result)
    return result

def _load_sync_dataset_cached_only(tail_ratio, ws, smooth_mode, granularity):
    return _compute_sync_dataset(tail_ratio, ws, smooth_mode, granularity, compute_if_missing=False)

def _get_sync_defaults():
    if sent_predictor is None:
        return SYNC_DEFAULT_WINDOW_SIZE, 'avg', 'word'
    return SYNC_DEFAULT_WINDOW_SIZE, SYNC_DEFAULT_SMOOTH_MODE, SYNC_DEFAULT_GRANULARITY

def _get_sync_view_cache(base_cache):
    if not base_cache:
        return None
    conv_id = base_cache.get('conv_id')
    if conv_id is None:
        return None
    sync_ws, sync_mode, sync_granularity = _get_sync_defaults()
    if (
        base_cache.get('speaker') == 'seeker' and
        base_cache.get('granularity') == sync_granularity
    ):
        return base_cache
    cache_key = (conv_id, sync_granularity)
    if cache_key in SYNC_VIEW_CACHE_MEMORY:
        return SYNC_VIEW_CACHE_MEMORY[cache_key]
    sync_cache, _ = build_conversation_cache(conv_id, 'seeker', sync_granularity, persist=False)
    if sync_cache is not None:
        SYNC_VIEW_CACHE_MEMORY[cache_key] = sync_cache
    return sync_cache

def _compute_current_sync_points_fresh(cache, ws, smooth_mode):
    if not cache or cache.get('speaker') != 'seeker':
        return None
    conv_id = cache.get('conv_id')
    actual_mode = 'context' if smooth_mode == 'context' and cache.get('granularity') == 'sentence' else 'avg'
    cache_key = (conv_id, int(ws), actual_mode, cache.get('granularity'))
    if cache_key in SYNC_CURRENT_POINTS_MEMORY_CACHE:
        return SYNC_CURRENT_POINTS_MEMORY_CACHE[cache_key]
    utterances = cache.get('utterances', [])
    results = cache.get('results', [])
    is_sent = (cache.get('granularity') == 'sentence')
    _xmid, _xbg, _xspan, _turn_to_x, turn_to_width = _build_turn_x_helpers(results, utterances, is_sent)
    series = {
        dim: _compute_diff_series(dim, ws, actual_mode, cache)
        for dim in ['valence', 'arousal', 'dominance']
    }
    if any(series[dim] is None or len(series[dim]['prev']['y']) == 0 for dim in series):
        return None
    sizes = [len(series[dim]['prev']['y']) for dim in series]
    size = min(sizes)
    if size <= 0:
        return None
    current = {
        'points': np.column_stack([
            np.asarray(series['valence']['prev']['y'][:size], dtype=float),
            np.asarray(series['arousal']['prev']['y'][:size], dtype=float),
            np.asarray(series['dominance']['prev']['y'][:size], dtype=float),
        ]),
        'turns': series['valence']['prev']['turns'][:size],
        'utterance_spans': np.asarray([
            float(turn_to_width.get(turn_idx, 1.0))
            for turn_idx in series['valence']['prev']['turns'][:size]
        ], dtype=float),
    }
    SYNC_CURRENT_POINTS_MEMORY_CACHE[cache_key] = current
    return current

def _compute_current_sync_points(cache, ws, smooth_mode):
    if _is_default_diff_mode(cache, ws, smooth_mode):
        bundle = _load_default_diff_bundle(cache.get('conv_id'), compute_if_missing=True)
        if bundle and bundle.get('current_sync'):
            current_sync = bundle['current_sync']
            return {
                'points': np.asarray(current_sync['points'], dtype=float),
                'turns': list(current_sync['turns']),
                'utterance_spans': np.asarray(current_sync['utterance_spans'], dtype=float),
            }
    return _compute_current_sync_points_fresh(cache, ws, smooth_mode)

def _compute_sync_rate_from_current(current, mean, cov, confidence_ratio):
    points = np.asarray(current['points'], dtype=float)
    spans = np.asarray(current['utterance_spans'], dtype=float)
    chi2_threshold = float(chi2.ppf(confidence_ratio, df=3))
    inside, dist2 = _mahalanobis_inside(points, mean, cov, chi2_threshold)
    total_span = float(np.sum(spans))
    inside_span = float(np.sum(spans * inside.astype(float)))
    sync_rate = (inside_span / total_span) if total_span > 0 else 0.0
    return {
        'inside': inside,
        'dist2': dist2,
        'total_span': total_span,
        'inside_span': inside_span,
        'sync_rate': sync_rate,
        'chi2_threshold': chi2_threshold,
    }

def _compute_all_sync_rates(tail_pct, confidence_pct):
    tail_pct = int(tail_pct or 25)
    confidence_pct = int(confidence_pct or round(SYNC_CONFIDENCE * 100))
    cache_key = (tail_pct, confidence_pct)
    if cache_key in ALL_SYNC_RATE_MEMORY_CACHE:
        return ALL_SYNC_RATE_MEMORY_CACHE[cache_key]

    tail_ratio = tail_pct / 100.0
    confidence_ratio = confidence_pct / 100.0
    sync_ws, sync_mode, sync_granularity = _get_sync_defaults()
    dataset = _load_sync_dataset_cached_only(tail_ratio, sync_ws, sync_mode, sync_granularity)
    if dataset is None:
        return None
    mean = np.asarray(dataset['mean'], dtype=float)
    cov = np.asarray(dataset['cov'], dtype=float)

    rows = []
    for conv_id in conv_ids:
        bundle = _load_default_diff_bundle(conv_id, compute_if_missing=True)
        if not bundle or not bundle.get('current_sync'):
            continue
        current_sync = bundle['current_sync']
        current = {
            'points': np.asarray(current_sync['points'], dtype=float),
            'turns': list(current_sync['turns']),
            'utterance_spans': np.asarray(current_sync['utterance_spans'], dtype=float),
        }
        if len(current['points']) == 0:
            continue
        metrics = _compute_sync_rate_from_current(current, mean, cov, confidence_ratio)
        rows.append({
            'conv_id': conv_id,
            'sync_rate': metrics['sync_rate'],
            'inside_span': metrics['inside_span'],
            'total_span': metrics['total_span'],
            'total_points': int(len(current['points'])),
            'inside_points': int(np.sum(metrics['inside'])),
            'mean_dist2': float(np.mean(metrics['dist2'])) if len(metrics['dist2']) else 0.0,
        })

    rows.sort(key=lambda row: row['sync_rate'])
    result = {
        'tail_pct': tail_pct,
        'confidence_pct': confidence_pct,
        'rows': rows,
    }
    ALL_SYNC_RATE_MEMORY_CACHE[cache_key] = result
    return result

def _cluster_sync_rates(rows, k):
    if not rows:
        return [], np.asarray([], dtype=float)
    k = max(1, min(int(k), len(rows)))
    values = np.asarray([row['sync_rate'] for row in rows], dtype=float)
    if k == 1:
        return np.zeros(len(values), dtype=int), np.asarray([float(np.mean(values))], dtype=float)

    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
    centers = np.quantile(values, quantiles)
    centers = np.asarray(centers, dtype=float)
    for _ in range(50):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            members = values[labels == idx]
            if len(members) > 0:
                new_centers[idx] = float(np.mean(members))
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order.tolist())}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=int)
    centers = centers[order]
    return labels, centers

def _build_sync_cluster_figure(cluster_data, k, selected_conv_id=None):
    fig = go.Figure()
    rows = cluster_data.get('rows', []) if cluster_data else []
    if not rows:
        fig.update_layout(
            title="同步率聚类",
            height=340, margin=dict(l=40, r=20, t=40, b=40))
        return fig, "暂无可聚类的同步率数据。"

    labels, centers = _cluster_sync_rates(rows, k)
    palette = ['#1565C0', '#2E7D32', '#EF6C00', '#6A1B9A', '#C62828', '#00838F', '#5D4037', '#AD1457']
    xs = np.asarray([row['sync_rate'] for row in rows], dtype=float)
    ys = labels.astype(float)
    colors = [palette[label % len(palette)] for label in labels]
    hover = [
        f"#{rows[i]['conv_id']}<br>同步率={rows[i]['sync_rate']:.2%}<br>"
        f"同步跨度={rows[i]['inside_span']:.1f}/{rows[i]['total_span']:.1f}<br>"
        f"cluster={labels[i]}"
        for i in range(len(rows))
    ]
    symbols = ['diamond' if selected_conv_id is not None and rows[i]['conv_id'] == selected_conv_id else 'circle'
               for i in range(len(rows))]
    sizes = [11 if selected_conv_id is not None and rows[i]['conv_id'] == selected_conv_id else 8
             for i in range(len(rows))]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers+text',
        marker=dict(color=colors, size=sizes, symbol=symbols, line=dict(width=1, color='white')),
        text=[f"#{row['conv_id']}" for row in rows],
        textposition='top center',
        textfont=dict(size=9, color='#37474F'),
        customdata=[row['conv_id'] for row in rows],
        hovertext=hover, hoverinfo='text',
        name='对话'))
    for idx, center in enumerate(centers):
        fig.add_vline(x=float(center), line_dash='dot', line_color=palette[idx % len(palette)], opacity=0.6)
    fig.update_layout(
        title=f"全数据同步率聚类 | k={k}",
        xaxis_title='同步率',
        yaxis_title='簇编号',
        yaxis=dict(tickmode='array', tickvals=list(range(len(centers))), range=[-0.5, len(centers) - 0.5]),
        height=340, margin=dict(l=40, r=20, t=40, b=40),
        hovermode='closest')
    summary = ' | '.join([
        f"簇{idx}: 中心 {centers[idx]:.2%} ({int(np.sum(labels == idx))} 个对话)"
        for idx in range(len(centers))
    ])
    return fig, summary

def _build_sync_cluster_distribution_figure(cluster_data, k, selected_conv_id=None):
    fig = go.Figure()
    rows = cluster_data.get('rows', []) if cluster_data else []
    if not rows:
        fig.update_layout(
            title="同步率分布",
            height=260, margin=dict(l=40, r=20, t=40, b=40))
        return fig

    labels, centers = _cluster_sync_rates(rows, k)
    palette = ['#1565C0', '#2E7D32', '#EF6C00', '#6A1B9A', '#C62828', '#00838F', '#5D4037', '#AD1457']
    xs = np.asarray([row['sync_rate'] for row in rows], dtype=float)
    fig.add_trace(go.Histogram(
        x=xs,
        nbinsx=24,
        marker=dict(color='rgba(120,144,156,0.55)', line=dict(color='white', width=1)),
        name='同步率分布',
        hovertemplate='同步率区间中心=%{x:.2%}<br>对话数=%{y}<extra></extra>'))

    for idx, center in enumerate(centers):
        fig.add_vline(
            x=float(center),
            line_dash='dot',
            line_color=palette[idx % len(palette)],
            opacity=0.75,
            annotation_text=f'簇{idx}: {center:.1%}',
            annotation_position='top')

    if selected_conv_id is not None:
        selected = next((row for row in rows if row['conv_id'] == selected_conv_id), None)
        if selected is not None:
            fig.add_vline(
                x=float(selected['sync_rate']),
                line_dash='solid',
                line_color='#C62828',
                line_width=2,
                opacity=0.9,
                annotation_text=f"#{selected_conv_id}: {selected['sync_rate']:.1%}",
                annotation_position='bottom')

    fig.update_layout(
        title=f"同步率分布 | k={k}",
        xaxis_title='同步率',
        yaxis_title='对话数',
        bargap=0.06,
        height=260,
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode='x')
    return fig

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
                value='sentence' if sent_predictor is not None else 'word',
                labelStyle={'display': 'inline-block', 'marginRight': '8px'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '8px'}),
    html.Div([
        html.Div([
            html.Label("窗口大小:", style={'fontWeight': 'bold'}),
            dcc.Slider(id='window-slider', min=2, max=20, step=1, value=2,
                marks={i: str(i) for i in range(2, 21, 2)},
                tooltip={"placement": "bottom", "always_visible": True})
        ], style={'width': '55%', 'display': 'inline-block', 'verticalAlign': 'middle'}),
        html.Div([
            html.Label("平滑模式:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(id='smooth-mode-radio',
                options=[{'label': '标准均值', 'value': 'avg'},
                         {'label': '上文窗口', 'value': 'context'}],
                value='context' if sent_predictor is not None else 'avg',
                labelStyle={'display': 'inline-block', 'marginRight': '10px'})
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
    html.Div([
        html.Div("🧭 同步范围（基于 seeker − 上次supporter 的三维状态差值）",
                 style={'fontSize': '13px', 'fontWeight': 'bold', 'padding': '8px 12px 4px',
                        'color': '#555'}),
        html.Div([
            html.Div([
                html.Label("尾段范围:", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='sync-tail-slider', min=0, max=50, step=5, value=25,
                    marks={i: f"{i}%" for i in range(0, 55, 10)},
                    tooltip={"placement": "bottom", "always_visible": True})
            ], style={'width': '55%', 'display': 'inline-block', 'verticalAlign': 'middle'}),
            html.Div([
                html.Label("椭圆置信度:", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='sync-confidence-slider', min=40, max=90, step=1,
                    value=int(round(SYNC_CONFIDENCE * 100)),
                    marks={40: '40%', 50: '50%', 67: '67%', 75: '75%', 90: '90%'},
                    tooltip={"placement": "bottom", "always_visible": True})
            ], style={'width': '38%', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginLeft': '4%'}),
        ], style={'padding': '8px 12px'}),
        html.Div([
            html.Div(
                "按状态差值图横坐标的最后 X% 截取；0% 表示不截尾，使用整段差值序列。"
                "同步范围固定按默认模式计算：句粒度 / 上下文窗口 / W=2。",
                style={'fontSize': '12px', 'color': '#666'})
        ], style={'padding': '0 12px 8px'}),
        dcc.Loading(type='circle', color='#666', children=[
            dcc.Graph(id='graph-sync-3d'),
            dcc.Graph(id='graph-sync-kde'),
            html.Div(id='sync-info', style={
                'padding': '6px 12px', 'fontSize': '12px', 'color': '#555',
                'backgroundColor': '#f8f9fa', 'borderRadius': '6px', 'marginTop': '4px'})
        ])
    ], style={'marginTop': '8px', 'borderTop': '1px solid #ddd', 'paddingTop': '4px'}),
    html.Div([
        html.Div("🧩 全数据同步率聚类（与上方同步范围参数对齐）",
                 style={'fontSize': '13px', 'fontWeight': 'bold', 'padding': '8px 12px 4px',
                        'color': '#555'}),
        html.Div([
            html.Label("类别数 k:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='sync-cluster-k-slider', min=2, max=8, step=1, value=4,
                marks={i: str(i) for i in range(2, 9)},
                tooltip={"placement": "bottom", "always_visible": True})
        ], style={'padding': '8px 12px'}),
        dcc.Loading(type='circle', color='#666', children=[
            dcc.Graph(id='graph-sync-clusters'),
            dcc.Graph(id='graph-sync-cluster-dist'),
            html.Div(id='sync-cluster-info', style={
                'padding': '6px 12px', 'fontSize': '12px', 'color': '#555',
                'backgroundColor': '#f8f9fa', 'borderRadius': '6px', 'marginTop': '4px'})
        ])
    ], style={'marginTop': '8px', 'borderTop': '1px solid #ddd', 'paddingTop': '4px'}),
    # 标签编辑弹窗
    dcc.Store(id='editing-target', data=None),
    dcc.Store(id='auto-marker-revision', data=0),
    dcc.Store(id='sync-dataset-store', data=None),
    dcc.Store(id='sync-current-store', data=None),
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
    if conv_id is None:
        return None, "请选择对话", []
    cache, meta = build_conversation_cache(conv_id, speaker, granularity, persist=True)
    if cache is None:
        return None, f"未找到 #{conv_id}", []
    return cache, _build_meta_info(meta or {}, conv_id), get_conv_markers(conv_id)


# 点击轮次：未标记→新增；已标记→打开编辑弹窗
@app.callback(
    Output('markers-store', 'data', allow_duplicate=True),
    Output('editing-target', 'data'),
    Output('label-input', 'value'),
    Output('label-modal', 'style'),
    Output('modal-title', 'children'),
    Input({'type': 'turn-div', 'index': ALL}, 'n_clicks'),
    State('conv-id-dropdown', 'value'),
    State('vad-cache-store', 'data'),
    State('markers-store', 'data'),
    prevent_initial_call=True)
def on_turn_click(all_clicks, conv_id, cache, markers):
    if not ctx.triggered_id or conv_id is None or not cache:
        return dash.no_update, dash.no_update, dash.no_update, HIDE_MODAL_STYLE, ""
    turn_idx = ctx.triggered_id['index']
    if not all_clicks or sum(c for c in all_clicks if c) == 0:
        return dash.no_update, dash.no_update, dash.no_update, HIDE_MODAL_STYLE, ""
    markers = markers or []
    existing = [m for m in markers if m['turn'] == turn_idx]
    if existing:
        lbl = existing[0].get('label', '')
        spk = existing[0].get('speaker', '')
        icon = MARKER_ICONS.get(spk, '📌')
        return (
            dash.no_update,
            {'type': 'manual', 'turn': turn_idx},
            lbl,
            SHOW_MODAL_STYLE,
            f"编辑拐点标签 {icon} 轮次[{turn_idx}]")
    else:
        dialog = cache.get('dialog', [])
        spk = dialog[turn_idx].get('speaker', 'seeker') if turn_idx < len(dialog) else 'seeker'
        new_markers = add_marker(conv_id, turn_idx, spk)
        return new_markers, dash.no_update, dash.no_update, HIDE_MODAL_STYLE, ""


@app.callback(
    Output('editing-target', 'data', allow_duplicate=True),
    Output('label-input', 'value', allow_duplicate=True),
    Output('label-modal', 'style', allow_duplicate=True),
    Output('modal-title', 'children', allow_duplicate=True),
    Input('graph-diff-valence', 'clickData'),
    Input('graph-diff-arousal', 'clickData'),
    Input('graph-diff-dominance', 'clickData'),
    State('conv-id-dropdown', 'value'),
    prevent_initial_call=True)
def on_diff_marker_click(click_v, click_a, click_d, conv_id):
    click_map = {
        'graph-diff-valence': click_v,
        'graph-diff-arousal': click_a,
        'graph-diff-dominance': click_d,
    }
    click_data = click_map.get(ctx.triggered_id) or click_v or click_a or click_d
    if conv_id is None or not click_data or 'points' not in click_data or not click_data['points']:
        return dash.no_update, dash.no_update, HIDE_MODAL_STYLE, dash.no_update
    customdata = click_data['points'][0].get('customdata')
    if not isinstance(customdata, dict) or customdata.get('kind') != 'auto-marker':
        return dash.no_update, dash.no_update, HIDE_MODAL_STYLE, dash.no_update
    marker_id = customdata.get('marker_id')
    label = get_auto_marker_labels(conv_id).get(marker_id, customdata.get('default_label', ''))
    title = f"编辑自动标记 {customdata.get('display', '')}（删除=恢复默认）"
    return customdata, label, SHOW_MODAL_STYLE, title


# 弹窗按钮操作
@app.callback(
    Output('markers-store', 'data', allow_duplicate=True),
    Output('label-modal', 'style', allow_duplicate=True),
    Output('editing-target', 'data', allow_duplicate=True),
    Output('auto-marker-revision', 'data'),
    Input('btn-save-label', 'n_clicks'),
    Input('btn-delete-marker', 'n_clicks'),
    Input('btn-cancel-modal', 'n_clicks'),
    State('editing-target', 'data'),
    State('label-input', 'value'),
    State('conv-id-dropdown', 'value'),
    State('auto-marker-revision', 'data'),
    prevent_initial_call=True)
def on_modal_action(save_c, del_c, cancel_c, editing_target, label_val, conv_id, auto_rev):
    if not ctx.triggered_id or editing_target is None or conv_id is None:
        return dash.no_update, HIDE_MODAL_STYLE, None, dash.no_update

    target_type = editing_target.get('type')
    trigger = ctx.triggered_id
    if trigger == 'btn-cancel-modal':
        return dash.no_update, HIDE_MODAL_STYLE, None, dash.no_update

    if target_type == 'manual':
        turn_idx = editing_target.get('turn')
        if trigger == 'btn-save-label':
            return update_label(conv_id, turn_idx, label_val or ''), HIDE_MODAL_STYLE, None, dash.no_update
        if trigger == 'btn-delete-marker':
            return remove_marker(conv_id, turn_idx), HIDE_MODAL_STYLE, None, dash.no_update
    elif target_type == 'auto-marker':
        marker_id = editing_target.get('marker_id')
        if trigger == 'btn-save-label':
            save_auto_marker_label(conv_id, marker_id, (label_val or '').strip())
            return dash.no_update, HIDE_MODAL_STYLE, None, (auto_rev or 0) + 1
        if trigger == 'btn-delete-marker':
            delete_auto_marker_label(conv_id, marker_id)
            return dash.no_update, HIDE_MODAL_STYLE, None, (auto_rev or 0) + 1

    return dash.no_update, HIDE_MODAL_STYLE, None, dash.no_update


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


def _build_diff_figure(dim, ws, smooth_mode, cache, markers=None, mf=None, sync_current_data=None):
    empty = go.Figure()
    empty.update_layout(title="需要选择单独说话者（Seeker 或 Supporter）",
                        height=220, margin=dict(l=40, r=20, t=35, b=25))
    if not cache or not cache.get('results') or not cache.get('bg_utterances'):
        return empty
    diff_series = None
    if _is_default_diff_mode(cache, ws, smooth_mode):
        bundle = _load_default_diff_bundle(cache.get('conv_id'), compute_if_missing=True)
        if bundle:
            diff_series = bundle.get('dims', {}).get(dim)
    if diff_series is None:
        diff_series = _compute_diff_series(dim, ws, smooth_mode, cache)
    if diff_series is None:
        return empty

    utterances = cache.get('utterances', [])
    prev_series = diff_series['prev']
    next_series = diff_series['next']
    bg_series = diff_series['seeker_curve']
    is_sent = diff_series['is_sent']
    fig = go.Figure()
    bg_shape = 'hv'
    if bg_series['x']:
        fig.add_trace(go.Scatter(
            x=bg_series['x'], y=bg_series['y'], mode='lines', name=f'{dim.capitalize()} (seeker)',
            line=dict(color=DIM_COLORS.get(dim, 'gray'), width=1.5, dash='dot', shape=bg_shape),
            opacity=0.3, yaxis='y2', hoverinfo='skip'))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)

    if prev_series['x']:
        fig.add_trace(go.Scatter(
            x=prev_series['x'], y=prev_series['y'], mode='lines+markers',
            name='seeker − prev_supp（被引导）',
            line=dict(color=DIFF_RELATION_COLORS['prev'], width=2), marker=dict(size=5),
            text=prev_series['hover'], hoverinfo='text', customdata=prev_series['turns']))
        outside_turns = set()
        if (
            sync_current_data and
            sync_current_data.get('conv_id') == cache.get('conv_id') and
            isinstance(sync_current_data.get('outside_turns'), list)
        ):
            outside_turns = set(sync_current_data['outside_turns'])
        outside_x = []
        outside_y = []
        outside_hover = []
        for x_val, y_val, turn in zip(prev_series['x'], prev_series['y'], prev_series['turns']):
            if turn in outside_turns:
                outside_x.append(x_val)
                outside_y.append(y_val)
                outside_hover.append(f"T[{turn}] 同步范围外")
        if outside_x:
            fig.add_trace(go.Scatter(
                x=outside_x, y=outside_y, mode='markers',
                name='同步范围外（prev）',
                marker=dict(size=8, color='#C62828', symbol='circle',
                            line=dict(width=1, color='white')),
                hovertext=outside_hover, hoverinfo='text', showlegend=False))
    if next_series['x']:
        fig.add_trace(go.Scatter(
            x=next_series['x'], y=next_series['y'], mode='lines+markers',
            name='seeker − next_supp（附和）',
            line=dict(color=DIFF_RELATION_COLORS['next'], width=2), marker=dict(size=4, color='#64B5F6'),
            text=next_series['hover'], hoverinfo='text', customdata=next_series['turns']))

    auto_markers = (
        _build_auto_diff_markers(dim, 'prev', prev_series, cache.get('conv_id')) +
        _build_auto_diff_markers(dim, 'next', next_series, cache.get('conv_id')))
    inside_turns = set()
    if (
        sync_current_data and
        sync_current_data.get('conv_id') == cache.get('conv_id') and
        isinstance(sync_current_data.get('inside_turns'), list)
    ):
        inside_turns = set(sync_current_data['inside_turns'])
    for marker in auto_markers:
        if marker['relation'] == 'prev' and inside_turns:
            point_index = marker.get('point_index')
            if point_index is not None and point_index > 0:
                current_turn = prev_series['turns'][point_index]
                prev_turn = prev_series['turns'][point_index - 1]
                if current_turn in inside_turns and prev_turn in inside_turns:
                    continue
        trend_symbol = 'triangle-up' if marker['trend'] == 'rise' else 'triangle-down'
        marker_color = '#BF360C' if marker['relation'] == 'prev' else 'rgba(100,181,246,0.9)'
        marker_x = marker['x'] if marker['relation'] == 'prev' else marker['x'] + 0.35
        fig.add_trace(go.Scatter(
            x=[marker_x], y=[marker['y']], mode='markers',
            marker=dict(size=14, color=marker_color, symbol=trend_symbol,
                        line=dict(width=1, color='white')),
            hovertext=[
                f"T[{marker['turn']}] {marker['label']}<br>"
                f"下一跳变化: {marker['slope']:+.3f}<br>点击可编辑"
            ],
            hoverinfo='text',
            customdata=[{
                'kind': 'auto-marker',
                'type': 'auto-marker',
                'marker_id': marker['marker_id'],
                'turn': marker['turn'],
                'display': f"{dim.capitalize()} | {RELATION_LABELS[marker['relation']]} | {TREND_LABELS[marker['trend']]}",
                'default_label': marker['default_label']
            }],
            name=f"{RELATION_LABELS[marker['relation']]}{TREND_LABELS[marker['trend']]}",
            showlegend=False))

    if markers and mf:
        mk_x, mk_y, mk_text, mk_color = [], [], [], []
        for m in markers:
            if m['speaker'] not in mf: continue
            mt = m['turn']; ms = m['speaker']; ml = m.get('label', '')
            mid_x = diff_series['turn_to_x'].get(mt)
            if ms == cache.get('bg_speaker') or mid_x is None:
                mid_x = _interpolate_turn_x(diff_series['turn_to_x'], mt)
            if mid_x is None:
                continue
            mc = MARKER_COLORS.get(ms, '#9c27b0')
            fig.add_vline(x=mid_x, line_dash='dash' if ms == 'seeker' else 'dot',
                          line_color=mc, opacity=0.5, line_width=2)
            icon = MARKER_ICONS.get(ms, '📌')
            fig.add_annotation(x=mid_x, y=1.0, text=f"{icon}T{mt}", showarrow=False,
                               font=dict(size=9, color=mc))
            mk_x.append(mid_x); mk_y.append(1.0)
            lbl = f"<b>{icon} 轮次[{mt}] {ms}</b>"
            if ml: lbl += f"<br>📝 {ml}"
            mk_text.append(lbl); mk_color.append(mc)
        if mk_x:
            fig.add_trace(go.Scatter(x=mk_x, y=mk_y, mode='markers', name='拐点标记',
                marker=dict(size=10, color=mk_color, symbol='diamond',
                            line=dict(width=1, color='white')),
                hovertext=mk_text, hoverinfo='text', showlegend=True))

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
    Input('vad-cache-store', 'data'),
    Input('markers-store', 'data'),
    Input('marker-filter', 'value'),
    Input('auto-marker-revision', 'data'),
    Input('sync-current-store', 'data'))
def update_diff_graphs(ws, smooth_mode, cache, markers, mf, _auto_rev, sync_current_data):
    figs = [_build_diff_figure(dim, ws, smooth_mode, cache, markers, mf, sync_current_data)
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
    Output('sync-dataset-store', 'data'),
    Input('sync-tail-slider', 'value'))
def load_sync_dataset_store(tail_pct):
    tail_pct = tail_pct or 25
    tail_ratio = tail_pct / 100.0
    sync_ws, sync_mode, sync_granularity = _get_sync_defaults()
    dataset = _load_sync_dataset_cached_only(tail_ratio, sync_ws, sync_mode, sync_granularity)
    if dataset is None:
        return {
            'available': False,
            'tail_pct': tail_pct,
            'window_size': sync_ws,
            'smooth_mode': sync_mode,
            'granularity': sync_granularity,
            'cache_key': _sync_cache_key(tail_ratio, sync_ws, sync_mode, sync_granularity),
        }
    serializable = _serialize_sync_dataset(dataset)
    serializable.update({
        'available': True,
        'tail_pct': tail_pct,
        'window_size': sync_ws,
        'smooth_mode': sync_mode,
        'granularity': sync_granularity,
        'cache_key': _sync_cache_key(tail_ratio, sync_ws, sync_mode, sync_granularity),
    })
    return serializable


@app.callback(
    Output('graph-sync-3d', 'figure'),
    Output('graph-sync-kde', 'figure'),
    Output('sync-info', 'children'),
    Output('sync-current-store', 'data'),
    Input('sync-dataset-store', 'data'),
    Input('sync-tail-slider', 'value'),
    Input('sync-confidence-slider', 'value'),
    Input('vad-cache-store', 'data'))
def update_sync_view(sync_dataset_data, tail_pct, confidence_pct, cache):
    empty = go.Figure()
    empty.update_layout(
        title="同步范围三维分布",
        scene=dict(xaxis_title='ΔValence', yaxis_title='ΔArousal', zaxis_title='ΔDominance'),
        height=520, margin=dict(l=10, r=10, t=40, b=10))
    empty_kde = go.Figure()
    empty_kde.update_layout(
        title="同步范围 KDE 投影",
        height=420, margin=dict(l=30, r=20, t=50, b=30))

    sync_ws, sync_mode, sync_granularity = _get_sync_defaults()
    sync_cache = _get_sync_view_cache(cache)
    if not sync_cache:
        empty.add_annotation(
            text="请先选择一个对话后再查看同步范围。",
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14))
        empty_kde.add_annotation(
            text="请先选择一个对话后再查看 KDE 投影。",
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14))
        return empty, empty_kde, "当前没有可用对话，无法显示同步范围。", None

    tail_pct = tail_pct or 25
    confidence_pct = confidence_pct or int(round(SYNC_CONFIDENCE * 100))
    confidence_ratio = float(confidence_pct) / 100.0
    tail_ratio = tail_pct / 100.0
    cache_key = _sync_cache_key(tail_ratio, sync_ws, sync_mode, sync_granularity)

    dataset = _load_sync_dataset_cached_only(tail_ratio, sync_ws, sync_mode, sync_granularity)
    if dataset is None and sync_dataset_data and sync_dataset_data.get('available'):
        sync_key = sync_dataset_data.get('cache_key')
        if sync_key == cache_key:
            dataset = _restore_sync_dataset(sync_dataset_data)
    if dataset is None:
        empty.add_annotation(
            text="未找到同步范围缓存，请先运行预计算脚本。",
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14))
        empty_kde.add_annotation(
            text="未找到 KDE 缓存，请先运行预计算脚本。",
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14))
        return empty, empty_kde, f"未命中缓存：`{cache_key}`。请先预计算同步范围缓存。", None

    current = _compute_current_sync_points(sync_cache, sync_ws, sync_mode)
    if dataset is None or current is None:
        empty.add_annotation(
            text="当前参数下没有足够的差值点可用于同步范围统计。",
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14))
        empty_kde.add_annotation(
            text="当前参数下没有足够的差值点可用于 KDE 投影。",
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14))
        return empty, empty_kde, "差值点不足，无法计算同步范围。", None

    mean = np.asarray(dataset['mean'], dtype=float)
    cov = np.asarray(dataset['cov'], dtype=float)
    sampled = np.asarray(dataset['sample_points'], dtype=float)
    alpha = (1.0 - confidence_ratio) / 2.0
    low = np.quantile(sampled, alpha, axis=0)
    high = np.quantile(sampled, 1.0 - alpha, axis=0)
    chi2_threshold = float(chi2.ppf(confidence_ratio, df=3))
    points = current['points']
    turns = current['turns']
    utterance_spans = np.asarray(current['utterance_spans'], dtype=float)
    inside, dist2 = _mahalanobis_inside(points, mean, cov, chi2_threshold)
    total_span = float(np.sum(utterance_spans))
    inside_span = float(np.sum(utterance_spans * inside.astype(float)))
    sync_rate = (inside_span / total_span) if total_span > 0 else 0.0

    display_tail_pct = int(round(float(dataset.get('tail_ratio', tail_ratio)) * 100))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=sampled[:, 0], y=sampled[:, 1], z=sampled[:, 2],
        mode='markers', name='全数据尾段分布',
        marker=dict(size=3, color='rgba(120,120,120,0.30)'),
        hoverinfo='skip'))
    fig.add_trace(_build_gaussian_ellipsoid(mean, cov, chi2_threshold, confidence_ratio))
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='lines', name='当前对话轨迹',
        line=dict(color='#546E7A', width=5),
        hovertext=[
            f"T[{turns[i]}] ΔV={points[i,0]:.3f}<br>ΔA={points[i,1]:.3f}<br>ΔD={points[i,2]:.3f}"
            for i in range(len(points))
        ],
        hoverinfo='text'))
    if np.any(inside):
        fig.add_trace(go.Scatter3d(
            x=points[inside, 0], y=points[inside, 1], z=points[inside, 2],
            mode='markers+text', name='同步区内',
            text=[f"T[{turns[i]}]" for i in np.where(inside)[0]],
            textposition='top center',
            textfont=dict(size=9, color='#1B5E20'),
            marker=dict(size=6, color='#2E7D32', opacity=0.95),
            hovertext=[
                f"T[{turns[i]}] 在同步范围内<br>Mahalanobis^2={dist2[i]:.3f}"
                for i in np.where(inside)[0]
            ],
            hoverinfo='text'))
    if np.any(~inside):
        fig.add_trace(go.Scatter3d(
            x=points[~inside, 0], y=points[~inside, 1], z=points[~inside, 2],
            mode='markers+text', name='同步区外',
            text=[f"T[{turns[i]}]" for i in np.where(~inside)[0]],
            textposition='top center',
            textfont=dict(size=9, color='#8E0000'),
            marker=dict(size=6, color='#C62828', opacity=0.9),
            hovertext=[
                f"T[{turns[i]}] 在同步范围外<br>Mahalanobis^2={dist2[i]:.3f}"
                for i in np.where(~inside)[0]
            ],
            hoverinfo='text'))

    fig.update_layout(
        title=(
            f"#{sync_cache.get('conv_id', '?')} | 同步范围 3D | 尾段 {display_tail_pct}% | "
            f"同步率 {sync_rate:.1%}"
        ),
        scene=dict(
            xaxis_title='ΔValence',
            yaxis_title='ΔArousal',
            zaxis_title='ΔDominance',
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            camera=dict(eye=dict(x=1.35, y=1.35, z=1.1))
        ),
        height=520, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=0.98, xanchor='left', x=0))

    kde_fig = _build_sync_kde_figure(
        dataset.get('kde_data', {}), points, inside, turns, sync_cache, display_tail_pct)
    sync_current_data = {
        'conv_id': sync_cache.get('conv_id'),
        'tail_pct': display_tail_pct,
        'confidence_pct': confidence_pct,
        'inside_turns': [int(turns[i]) for i in np.where(inside)[0]],
        'outside_turns': [int(turns[i]) for i in np.where(~inside)[0]],
    }

    info = html.Div([
        html.Span(
            f"同步率: {sync_rate:.1%}（按 seeker 话语在背景阶梯图上的真实跨度加权；"
            f"同步跨度 {inside_span:.1f} / 总跨度 {total_span:.1f}）",
            style={'marginRight': '18px', 'fontWeight': 'bold', 'color': '#2E7D32'}),
        html.Span(
            f"正态近似中心: μ=({mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f})",
            style={'marginRight': '18px'}),
        html.Span(
            f"全数据统计: {dataset['used_conversations']} 个对话，尾段 {display_tail_pct}% ，"
            f"{int(round(confidence_ratio * 100))}% 高斯椭球；"
            f"外围分位参考 V[{low[0]:.3f}, {high[0]:.3f}] "
            f"A[{low[1]:.3f}, {high[1]:.3f}] "
            f"D[{low[2]:.3f}, {high[2]:.3f}]；"
            f"固定默认模式：{sync_granularity} / {sync_mode} / W={sync_ws}；"
            f"缓存键 {dataset.get('cache_key', 'n/a')}",
            style={'color': '#666'})
    ])
    return fig, kde_fig, info, sync_current_data


@app.callback(
    Output('graph-sync-clusters', 'figure'),
    Output('graph-sync-cluster-dist', 'figure'),
    Output('sync-cluster-info', 'children'),
    Input('sync-tail-slider', 'value'),
    Input('sync-confidence-slider', 'value'),
    Input('sync-cluster-k-slider', 'value'),
    Input('conv-id-dropdown', 'value'))
def update_sync_clusters(tail_pct, confidence_pct, k, selected_conv_id):
    cluster_data = _compute_all_sync_rates(tail_pct, confidence_pct)
    if cluster_data is None:
        fig = go.Figure()
        fig.update_layout(title="同步率聚类", height=340, margin=dict(l=40, r=20, t=40, b=40))
        fig.add_annotation(
            text="缺少默认差值缓存或同步范围缓存，请先完成预计算。",
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14))
        dist_fig = go.Figure()
        dist_fig.update_layout(title="同步率分布", height=260, margin=dict(l=40, r=20, t=40, b=40))
        dist_fig.add_annotation(
            text="缺少默认差值缓存或同步范围缓存。",
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14))
        return fig, dist_fig, "当前无法计算全数据同步率聚类。"
    fig, summary = _build_sync_cluster_figure(cluster_data, k or 4, selected_conv_id)
    dist_fig = _build_sync_cluster_distribution_figure(cluster_data, k or 4, selected_conv_id)
    info = html.Div([
        html.Span(
            f"参数对齐：尾段 {cluster_data['tail_pct']}% | 椭圆置信度 {cluster_data['confidence_pct']}% | k={int(k or 4)}",
            style={'marginRight': '18px', 'fontWeight': 'bold'}),
        html.Span(summary, style={'color': '#666'})
    ])
    return fig, dist_fig, info


@app.callback(
    Output('dialog-panel', 'children'),
    Input('graph-valence', 'hoverData'),
    Input('graph-arousal', 'hoverData'),
    Input('graph-dominance', 'hoverData'),
    Input('vad-cache-store', 'data'),
    Input('markers-store', 'data'))
def on_hover_dialog(hoverV, hoverA, hoverD, cache, markers):
    hover_map = {
        'graph-valence': hoverV,
        'graph-arousal': hoverA,
        'graph-dominance': hoverD,
    }
    hoverData = hover_map.get(ctx.triggered_id) or hoverV or hoverA or hoverD
    if not cache or not cache.get('dialog'): return []
    dialog = cache['dialog']; results = cache.get('results', [])
    markers = markers or []; mm = {m['turn']: m for m in markers}
    hl = None
    if hoverData and 'points' in hoverData and hoverData['points']:
        pt = hoverData['points'][0]; cd = pt.get('customdata')
        parsed_turn = _parse_turn_from_customdata(cd)
        if parsed_turn is not None and parsed_turn >= 0:
            hl = parsed_turn
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

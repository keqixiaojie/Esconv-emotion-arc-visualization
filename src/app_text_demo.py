import re
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from src.vad_extractor import VADExtractor
from src.emotion_smoothing import smooth_scores


def split_sentences(text: str):
    """将文本按句子边界切分，过滤空串。"""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]

# 初始化VAD提取器
lexicon_path = "NRC-VAD-Lexicon-v2.1.txt"  # 请根据实际词典路径调整
vad_extractor = VADExtractor(lexicon_path)

app = dash.Dash(__name__)
app.title = "纯文本情感弧线交互示例"

default_text = """
I was so happy and thrilled today! It started a bit slow, actually quite boring and sad. 
But then a bunch of good things happened. I am thrilled and totally amazed. 
A battery of tests proved I am healthy. What a fantastic victory.
I felt so good, everything is happy and bright. But later, a sad news made me feel boring again.
"""

app.layout = html.Div([
    html.H2("纯文本情感弧线动态展示"),
    dcc.Textarea(
        id='input-text',
        value=default_text,
        style={'width': '100%', 'height': 150}
    ),
    html.Br(),
    html.Div([
        html.Label("选择情感维度:"),
        dcc.RadioItems(
            id='dimension-radio',
            options=[
                {'label': 'Valence', 'value': 'valence'},
                {'label': 'Arousal', 'value': 'arousal'},
                {'label': 'Dominance', 'value': 'dominance'}
            ],
            value='valence',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        )
    ]),
    html.Br(),
    html.Div([
        html.Div([
            html.Label("滑动窗口大小:"),
            dcc.Slider(
                id='window-slider',
                min=2,
                max=10,
                step=1,
                value=4,
                marks={i: str(i) for i in range(2, 11)}
            )
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'middle'}),
        html.Div([
            html.Label("平滑模式:"),
            dcc.RadioItems(
                id='smooth-mode-radio',
                options=[{'label': '标准均值', 'value': 'avg'},
                         {'label': '上文窗口', 'value': 'context'}],
                value='avg',
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'marginLeft': '30px'}),
    ]),
    html.Br(),
    dcc.Graph(id='emotion-graph'),
    html.Div(id='info-output')
])

@app.callback(
    Output('emotion-graph', 'figure'),
    Output('info-output', 'children'),
    Input('input-text', 'value'),
    Input('dimension-radio', 'value'),
    Input('window-slider', 'value'),
    Input('smooth-mode-radio', 'value')
)
def update_emotion_graph(text, dimension, window_size, smooth_mode):
    if smooth_mode == 'context':
        # 上文窗口模式：按句切分，每位置 i 取前 window_size 句拼接整体打分
        sentences = split_sentences(text)
        if len(sentences) < window_size:
            return go.Figure(), f"句子数 ({len(sentences)}) 小于窗口大小 ({window_size})，无法计算上文窗口。"

        # 离散：每句单独打分（NRC 词均值）
        disc_scores = []
        for s in sentences:
            r = vad_extractor.extract(s)
            disc_scores.append(float(np.mean([x[dimension] for x in r])) if r else 0.0)
        disc_scores = np.array(disc_scores)

        # 上文窗口：前 window_size 句拼接整体打分
        ctx_scores = []
        for i in range(window_size - 1, len(sentences)):
            combined = ' '.join(sentences[i - window_size + 1 : i + 1])
            r = vad_extractor.extract(combined)
            ctx_scores.append(float(np.mean([x[dimension] for x in r])) if r else 0.0)
        ctx_scores = np.array(ctx_scores)

        x_disc = np.arange(len(sentences))
        x_ctx = np.arange(window_size - 1, len(sentences))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_disc, y=disc_scores,
                                 mode='markers+lines', name='离散句子',
                                 line=dict(dash='dot', color='gray'),
                                 marker=dict(size=8, color='gray')))
        fig.add_trace(go.Scatter(x=x_ctx, y=ctx_scores,
                                 mode='lines+markers',
                                 name=f'上文窗口弧线 (W={window_size})',
                                 line=dict(color='firebrick', width=3),
                                 marker=dict(size=4)))
        fig.update_layout(
            title="情感弧线动态变化（上文窗口模式）",
            xaxis_title="句子索引",
            yaxis_title=f"{dimension.capitalize()} 分数",
            yaxis=dict(range=[-1, 1]),
            hovermode="x unified", height=500)
        return fig, f"共 {len(sentences)} 句，窗口大小：{window_size}，上文窗口弧线 {len(ctx_scores)} 点"

    # ---- 标准均值模式 ----
    results = vad_extractor.extract(text)
    if not results:
        return go.Figure(), "未从文本中提取到任何情感词汇。"

    scores = vad_extractor.get_scores_array(results, dimension)
    if len(scores) < window_size:
        return go.Figure(), f"有效情感词数 ({len(scores)}) 小于窗口大小 ({window_size})，无法平滑。"

    smooth = smooth_scores(scores, window_size)
    x_discrete = np.arange(len(scores))
    x_smooth = np.arange(window_size - 1, len(scores))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_discrete, y=scores,
                             mode='markers+lines',
                             name='离散情感词',
                             line=dict(dash='dot', color='gray'),
                             marker=dict(size=8, color='gray')))
    fig.add_trace(go.Scatter(x=x_smooth, y=smooth,
                             mode='lines',
                             name=f'平滑情感弧线 (窗口={window_size})',
                             line=dict(color='firebrick', width=3)))
    fig.update_layout(
        title="情感弧线动态变化",
        xaxis_title="情感词索引",
        yaxis_title=f"{dimension.capitalize()} 分数",
        yaxis=dict(range=[-1, 1]),
        hovermode="x unified", height=500)

    return fig, f"提取到 {len(results)} 个情感词，窗口大小：{window_size}"

if __name__ == '__main__':
    app.run_server(debug=True)
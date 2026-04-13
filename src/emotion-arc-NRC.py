import numpy as np
import re
import matplotlib.pyplot as plt

def get_emotion_arc_with_vis(raw_text, lexicon_path, dimension='valence', window_size=4):
    print("="*50)
    print(f"🚀 开始提取情感弧线 (支持 V2 词组匹配) | 维度: {dimension.upper()} | 窗口: {window_size}")
    print("="*50)

    # ==========================================
    # 步骤 1：构建哈希字典，并动态获取最大词组长度
    # ==========================================
    dim_map = {'valence': 1, 'arousal': 2, 'dominance': 3}
    dim_idx = dim_map.get(dimension.lower())
    
    lexicon_dict = {}
    max_ngram = 1  # 记录词典中最长词组包含的单词数
    
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            next(f) 
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    term = parts[0].lower()
                    score = float(parts[dim_idx])
                    lexicon_dict[term] = score
                    
                    # 计算该 term 包含几个单词 (空格数 + 1)
                    word_count = term.count(' ') + 1
                    if word_count > max_ngram:
                        max_ngram = word_count
                        
        print(f"✅ 步骤 1: 成功加载词表。共 {len(lexicon_dict)} 词。最大词组长度: {max_ngram} 词。")
    except FileNotFoundError:
        print("⚠️ 未找到文件，使用内置 Mock 词表 (包含词组)..")
        lexicon_dict = {'happy': 0.8, 'sad': -0.7, 'a bit': -0.2, 'a bunch': 0.1, 
                        'a battery': -0.3, 'victory': 0.8, 'boring': -0.5, 'out of control': -0.9}
        max_ngram = 3

    # ==========================================
    # 步骤 2：文本分词与【贪心词组匹配】
    # ==========================================
    # 先把纯文本转成有序的单字列表
    tokens = re.findall(r'\b[a-z]+\b', raw_text.lower())
    
    matched_terms = []
    discrete_scores = []
    
    i = 0
    while i < len(tokens):
        match_found = False
        # 从最大可能长度开始，往下递减尝试匹配 (比如先试 4个词，再试 3个...)
        for n in range(max_ngram, 0, -1):
            if i + n <= len(tokens):
                # 拼装候选词组
                candidate = " ".join(tokens[i:i+n])
                if candidate in lexicon_dict:
                    matched_terms.append(candidate)
                    discrete_scores.append(lexicon_dict[candidate])
                    i += n  # 匹配成功，指针直接跳过被匹配的 n 个词！
                    match_found = True
                    break
                    
        # 如果从长到短（甚至单字）都没在词典里找到，说明是中性废词，直接跳过 1 步
        if not match_found:
            i += 1

    scores_array = np.array(discrete_scores)
    print(f"\n🔍 步骤 2: 贪心匹配完成。")
    print(f"   -> 提取到的有效【情感词/词组】总数: {len(matched_terms)}")
    print(f"   -> 匹配明细 (原文按顺序出现的有效情感):")
    for term, score in zip(matched_terms, discrete_scores):
        print(f"      [{term}] -> {score}")
    
    if len(scores_array) < window_size:
        print(f"\n❌ 错误: 有效情感词 ({len(scores_array)}) 小于窗口大小 ({window_size})，无法计算平均。")
        return None, None

    # ==========================================
    # 步骤 3：一维卷积生成平滑弧线
    # ==========================================
    kernel = np.ones(window_size) / window_size
    emotion_arc = np.convolve(scores_array, kernel, mode='valid')
    
    print(f"\n📈 步骤 3: 卷积计算完成。弧线长: {len(emotion_arc)}")
    print("="*50)
    
    return scores_array, emotion_arc

def plot_emotion_dynamics(discrete_scores, emotion_arc, window_size):
    """
    使用 Matplotlib 绘制情感动力学图表
    """
    if discrete_scores is None or emotion_arc is None:
        return

    plt.figure(figsize=(12, 6))
    
    # 1. 绘制底层的“离散情感脉冲”
    x_discrete = np.arange(len(discrete_scores))
    plt.plot(x_discrete, discrete_scores, marker='o', linestyle=':', color='gray', 
             alpha=0.5, label='Discrete Emotion Words (Raw)')
    
    # 2. 绘制平滑后的“情感弧线”
    # 物理意义对齐：Pandas 的 rolling.mean 默认是 trailing window（向后看）
    # 即第一个有效均值产生在索引 W-1 处。我们把弧线的 X 坐标对齐到对应的结束时间点。
    x_arc = np.arange(window_size - 1, len(discrete_scores))
    plt.plot(x_arc, emotion_arc, color='crimson', linewidth=3, 
             label=f'Emotion Arc (Rolling Mean, W={window_size})')
    
    # 3. 图表美化
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3) # 零点中枢线
    plt.title("Utterance Emotion Dynamics: From Discrete Words to Continuous Arc", fontsize=14, pad=15)
    plt.xlabel("Timeline (Index of Emotion Words)", fontsize=12)
    plt.ylabel("Valence Score", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 动态调整 Y 轴范围，留出视觉空间
    plt.ylim(min(discrete_scores)-0.2, max(discrete_scores)+0.2)
    
    plt.tight_layout()
    plt.show()

# 测试运行
if __name__ == "__main__":
    # 注意这句话里有 v2 的词组：'a battery', 'a bit', 'a bunch'
    test_text = """
    I was so happy and thrilled today! It started a bit slow, actually quite boring and sad. 
    But then a bunch of good things happened. I am thrilled and totally amazed. 
    A battery of tests proved I am healthy. What a fantastic victory.
    I felt so good, everything is happy and bright. But later, a sad news made me feel boring again.
    """
    
    lexicon_file = "NRC-VAD-Lexicon-v2.1.txt" 
    W = 4
    
    raw_scores, arc = get_emotion_arc_with_vis(test_text, lexicon_file, dimension='valence', window_size=W)
    # 渲染图表
    plot_emotion_dynamics(raw_scores, arc, window_size=W)
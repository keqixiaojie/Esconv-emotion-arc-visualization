import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional

class VADExtractor:
    """VAD词语提取模块 - 支持多维度（Valence, Arousal, Dominance）"""
    
    def __init__(self, lexicon_path: str):
        """
        初始化VAD提取器
        
        Args:
            lexicon_path: NRC-VAD词典文件路径
        """
        self.lexicon_path = lexicon_path
        self.lexicon_dict = {}
        self.max_ngram = 1
        self.dim_map = {'valence': 1, 'arousal': 2, 'dominance': 3}
        self._load_lexicon()
    
    def _load_lexicon(self):
        """加载VAD词典"""
        try:
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                next(f)  # 跳过header
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        term = parts[0].lower()
                        try:
                            valence = float(parts[1])
                            arousal = float(parts[2])
                            dominance = float(parts[3])
                            self.lexicon_dict[term] = {
                                'valence': valence,
                                'arousal': arousal,
                                'dominance': dominance
                            }
                            word_count = term.count(' ') + 1
                            if word_count > self.max_ngram:
                                self.max_ngram = word_count
                        except ValueError:
                            continue
            print(f"✅ 加载词表成功: {len(self.lexicon_dict)} 词，最大词组长度: {self.max_ngram}")
        except FileNotFoundError:
            print(f"⚠️ 未找到词表文件: {self.lexicon_path}，使用Mock词表")
            self.lexicon_dict = {
                'happy': {'valence': 0.8, 'arousal': 0.6, 'dominance': 0.5},
                'sad': {'valence': -0.7, 'arousal': -0.5, 'dominance': -0.4},
                'angry': {'valence': -0.6, 'arousal': 0.8, 'dominance': 0.7},
                'excited': {'valence': 0.7, 'arousal': 0.9, 'dominance': 0.6},
            }
            self.max_ngram = 1
    
    def extract(self, text: str) -> List[Dict]:
        """
        从文本中提取VAD词语
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict]: 每个元素包含 {term, valence, arousal, dominance, position}
        """
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        results = []
        i = 0
        position = 0
        
        while i < len(tokens):
            match_found = False
            # 从最大可能长度开始贪心匹配
            for n in range(self.max_ngram, 0, -1):
                if i + n <= len(tokens):
                    candidate = " ".join(tokens[i:i+n])
                    if candidate in self.lexicon_dict:
                        vad = self.lexicon_dict[candidate]
                        results.append({
                            'term': candidate,
                            'valence': vad['valence'],
                            'arousal': vad['arousal'],
                            'dominance': vad['dominance'],
                            'position': position
                        })
                        i += n
                        position += 1
                        match_found = True
                        break
            
            if not match_found:
                i += 1
        
        return results
    
    def save_cache(self, results: List[Dict], output_path: str, metadata: Dict = None):
        """保存提取结果到缓存文件"""
        cache_data = {
            'metadata': metadata or {},
            'results': results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 缓存已保存: {output_path}")
    
    @staticmethod
    def load_cache(cache_path: str) -> Tuple[List[Dict], Dict]:
        """从缓存文件加载提取结果"""
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data['results'], cache_data.get('metadata', {})
    
    def get_scores_array(self, results: List[Dict], dimension: str = 'valence') -> np.ndarray:
        """
        从提取结果获取指定维度的分数数组
        
        Args:
            results: 提取结果列表
            dimension: 'valence', 'arousal', 或 'dominance'
            
        Returns:
            np.ndarray: 分数数组
        """
        if dimension not in ['valence', 'arousal', 'dominance']:
            raise ValueError(f"Invalid dimension: {dimension}")
        
        scores = [r[dimension] for r in results]
        return np.array(scores)


# ---------------------------------------------------------------------------
# 句粒度 VAD 预测（基于 RoBERTa 微调模型）
# ---------------------------------------------------------------------------

import os
import re as _re
import string as _string

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer


_ROBERTA_MODEL_NAME = "roberta-large"

def _resolve_pretrained(cache_root: str) -> str:
    """
    优先使用本地 HF 缓存（snapshots 目录），
    若不存在则返回模型名称让 from_pretrained 联网下载到 cache_root。
    """
    snapshots_dir = os.path.join(cache_root, f"models--{_ROBERTA_MODEL_NAME}", "snapshots")
    if os.path.isdir(snapshots_dir):
        snaps = os.listdir(snapshots_dir)
        if snaps:
            return os.path.join(snapshots_dir, snaps[0])
    print(f"⬇️ 本地未找到 {_ROBERTA_MODEL_NAME} 缓存，将从 HuggingFace 下载到 {cache_root} ...")
    return _ROBERTA_MODEL_NAME


class _VADRegressionModel(nn.Module):
    """与训练端 PretrainedLMModel(task=vad-regression) 对齐的推理模型"""

    def __init__(self, config, label_num):
        super().__init__()
        self.label_num = label_num
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.projection_lm = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.head = nn.Linear(config.hidden_size, label_num * 3)
        self.activation = nn.Sigmoid()
        self.v_head = nn.Linear(label_num, 1, bias=False)
        self.a_head = nn.Linear(label_num, 1, bias=False)
        self.d_head = nn.Linear(label_num, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, return_dict=False)
        hidden_states, pooled_output = outputs
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)
        v_logit, a_logit, d_logit = torch.split(logits, self.label_num, dim=1)
        v_out = self.v_head(self.activation(v_logit))
        a_out = self.a_head(self.activation(a_logit))
        d_out = self.d_head(self.activation(d_logit))
        return torch.cat([v_out, a_out, d_out], dim=1)


def _preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text == text else ""
    text = text.strip('"').strip("'").strip()
    text = _re.sub(r"([{}])".format(_re.escape(_string.punctuation)), r" \1 ", text)
    text = _re.sub(r"\s{2,}", " ", text)
    return text.strip()


class SentenceVADPredictor:
    """句粒度 VAD 预测器 - 使用 RoBERTa 微调模型对每个 utterance 预测 V/A/D"""

    def __init__(self, ckpt_dir: str, config_cache: str, vocab_cache: str,
                 epoch: int = 15, ckpt_prefix: str = "emobank-vad-regression",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 定位 checkpoint
        ckpt_path = None
        for f in os.listdir(ckpt_dir):
            if f.startswith(ckpt_prefix) and f.endswith(f"-{epoch}.ckpt"):
                ckpt_path = os.path.join(ckpt_dir, f)
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"未找到 epoch={epoch} 的 checkpoint (prefix={ckpt_prefix}) in {ckpt_dir}")

        # 加载 tokenizer（本地缓存命中则离线，否则联网下载）
        vocab_path = _resolve_pretrained(vocab_cache)
        self.tokenizer = RobertaTokenizer.from_pretrained(
            vocab_path,
            cache_dir=vocab_cache if vocab_path == _ROBERTA_MODEL_NAME else None)

        # 加载模型 config
        config_path = _resolve_pretrained(config_cache)
        config = RobertaConfig.from_pretrained(
            config_path,
            cache_dir=config_cache if config_path == _ROBERTA_MODEL_NAME else None)
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        label_num = state_dict["head.weight"].shape[0] // 3

        self.model = _VADRegressionModel(config, label_num)
        new_sd = {k.replace("pre_trained_lm.", "roberta."): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_sd, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ 句粒度模型已加载: {ckpt_path}  device={self.device}  label_num={label_num}")
        # warmup: 跑一次空推理，让 CUDA kernel 预编译
        self._warmup()

    def _warmup(self):
        """首次推理 warmup，避免回调超时"""
        dummy = self.tokenizer(["warmup"], max_length=32, padding="max_length",
                                truncation=True, return_tensors="pt")
        with torch.no_grad():
            self.model(dummy["input_ids"].to(self.device),
                       attention_mask=dummy["attention_mask"].to(self.device))
        print("✅ 模型 warmup 完成")

    def _predict_batch(self, texts: List[str], max_len: int = 256, batch_size: int = 32) -> np.ndarray:
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch = [_preprocess_text(t) for t in texts[i:i + batch_size]]
            enc = self.tokenizer(batch, max_length=max_len, padding="max_length",
                                 truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attn_mask)
                preds = F.relu(logits).cpu().numpy()
            all_preds.append(preds)
        return np.concatenate(all_preds, axis=0)

    def predict_utterances(self, utterances: List[Dict]) -> List[Dict]:
        """
        对 utterance 列表做句级 VAD 预测，返回与词粒度 extract() 格式一致的结果。

        Args:
            utterances: esconv_loader.filter_utterances() 的输出

        Returns:
            List[Dict]: 每个元素 {term, valence, arousal, dominance, position, turn_info}
        """
        if not utterances:
            return []
        texts = [u['content'] for u in utterances]
        preds = self._predict_batch(texts)  # 模型输出范围 1-5 (EmoBank)
        results = []
        for i, (u, vad) in enumerate(zip(utterances, preds)):
            content = u['content']
            # 将 EmoBank 1-5 范围映射到 -1~1: (x - 3) / 2
            v_mapped = (float(vad[0]) - 3.0) / 2.0
            a_mapped = (float(vad[1]) - 3.0) / 2.0
            d_mapped = (float(vad[2]) - 3.0) / 2.0
            results.append({
                'term': content[:50] + ('...' if len(content) > 50 else ''),
                'valence': v_mapped,
                'arousal': a_mapped,
                'dominance': d_mapped,
                'position': i,
                'turn_info': {
                    'turn_index': u['turn_index'],
                    'speaker': u['speaker'],
                    'content': u['content'],
                    'strategy': u.get('strategy'),
                },
            })
        return results
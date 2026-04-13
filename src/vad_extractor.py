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
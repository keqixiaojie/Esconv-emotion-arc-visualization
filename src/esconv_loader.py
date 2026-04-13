import json
from typing import List, Dict, Optional, Tuple

class ESConvLoader:
    """ESConv对话加载与处理模块"""
    
    def __init__(self, json_path: str):
        """
        初始化ESConv加载器
        
        Args:
            json_path: ESConv-strategy.json 文件路径
        """
        self.json_path = json_path
        self.conversations = []
        self._load_conversations()
    
    def _load_conversations(self):
        """加载对话数据"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.conversations = json.load(f)
            print(f"✅ 加载对话成功: {len(self.conversations)} 个对话")
        except FileNotFoundError:
            print(f"❌ 文件不存在: {self.json_path}")
            self.conversations = []
    
    def get_conversation_ids(self) -> List[int]:
        """获取所有对话ID"""
        return [conv.get('meta', {}).get('id', i) for i, conv in enumerate(self.conversations)]
    
    def get_conversation(self, conv_id: int) -> Optional[Dict]:
        """
        按ID获取单个对话
        
        Args:
            conv_id: 对话ID
            
        Returns:
            对话数据字典，包含 meta 和 dialog
        """
        for conv in self.conversations:
            if conv.get('meta', {}).get('id') == conv_id:
                return conv
        return None
    
    def filter_utterances(self, dialog: List[Dict], speaker: str = 'seeker') -> List[Dict]:
        """
        从对话中过滤指定说话者的话语
        
        Args:
            dialog: 对话轮次列表
            speaker: 'seeker', 'supporter', 或 'both'
            
        Returns:
            List[Dict]: 每个元素包含 {turn_index, speaker, content, strategy}
        """
        results = []
        
        for turn_idx, turn in enumerate(dialog):
            turn_speaker = turn.get('speaker', '').lower()
            
            if speaker == 'both':
                results.append({
                    'turn_index': turn_idx,
                    'speaker': turn_speaker,
                    'content': turn.get('content', ''),
                    'strategy': turn.get('strategy')
                })
            elif speaker.lower() == turn_speaker:
                results.append({
                    'turn_index': turn_idx,
                    'speaker': turn_speaker,
                    'content': turn.get('content', ''),
                    'strategy': turn.get('strategy')
                })
        
        return results
    
    def utterances_to_text(self, utterances: List[Dict]) -> str:
        """
        将话语列表拼接为纯文本
        
        Args:
            utterances: 话语列表
            
        Returns:
            拼接后的文本
        """
        return ' '.join([u['content'] for u in utterances])
    
    def build_turn_mapping(self, utterances: List[Dict], vad_results: List[Dict]) -> List[Dict]:
        """
        建立VAD匹配词 → 对话轮次的映射
        
        通过统计每个话语贡献的token数量，将VAD结果中的position
        映射回对应的对话轮次。
        
        Args:
            utterances: 话语列表
            vad_results: VAD提取结果
            
        Returns:
            List[Dict]: 与vad_results等长，每个元素包含对应轮次信息
        """
        import re
        
        # 统计每个话语的token数量，构建累积边界
        token_boundaries = []  # (cumulative_start, cumulative_end, utterance_index)
        cumulative = 0
        for idx, utterance in enumerate(utterances):
            tokens = re.findall(r'\b[a-z]+\b', utterance['content'].lower())
            token_count = len(tokens)
            token_boundaries.append((cumulative, cumulative + token_count, idx))
            cumulative += token_count
        
        # 为每个VAD结果，根据其在全文token流中的原始位置找到对应话语
        # vad_results中的position是匹配词的序号，但我们需要原始token位置
        # 重新计算：遍历全文token流，记录每个匹配词消耗的原始token位置
        full_text = self.utterances_to_text(utterances)
        all_tokens = re.findall(r'\b[a-z]+\b', full_text.lower())
        
        # 重放贪心匹配过程，记录每个匹配词的原始token起始位置
        from src.vad_extractor import VADExtractor
        # 避免循环导入，这里直接用简单方法
        
        mapping = []
        for vad_item in vad_results:
            # 用term的词数来估算它在原始token流中的大致位置
            mapping.append(None)
        
        # 更精确的方法：在提取时就记录原始token索引
        # 这里用回退方案：按顺序分配
        token_pos = 0  # 当前在全文token流中的位置
        result_mapping = []
        
        for vad_item in vad_results:
            term = vad_item['term']
            term_token_count = term.count(' ') + 1
            
            # 在token流中搜索该term的位置
            found = False
            while token_pos + term_token_count <= len(all_tokens):
                candidate = ' '.join(all_tokens[token_pos:token_pos + term_token_count])
                if candidate == term:
                    # 找到了，确定属于哪个utterance
                    mid_pos = token_pos + term_token_count // 2
                    utt_info = None
                    for start, end, utt_idx in token_boundaries:
                        if start <= mid_pos < end:
                            u = utterances[utt_idx]
                            utt_info = {
                                'turn_index': u['turn_index'],
                                'speaker': u['speaker'],
                                'content': u['content'],
                                'strategy': u.get('strategy')
                            }
                            break
                    result_mapping.append(utt_info)
                    token_pos += term_token_count
                    found = True
                    break
                token_pos += 1
            
            if not found:
                result_mapping.append(None)
        
        return result_mapping
    
    def get_conversation_summary(self, conv_id: int) -> Optional[Dict]:
        """获取对话的元信息摘要"""
        conv = self.get_conversation(conv_id)
        if conv:
            return conv.get('meta', {})
        return None
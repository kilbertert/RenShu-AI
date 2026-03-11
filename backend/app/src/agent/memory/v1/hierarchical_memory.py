# """
# 分层记忆系统
#
# 实现三层记忆架构：
# - Working Memory: 当前对话窗口
# - Episodic Memory: 会话内摘要
# - Semantic Memory: 跨会话的长期知识
# """
#
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass, field
# from enum import Enum
# from datetime import datetime
# import json
#
#
# class MemoryLevel(Enum):
#     """记忆层级"""
#     WORKING = "working"      # 工作记忆：当前对话窗口
#     EPISODIC = "episodic"    # 情景记忆：会话摘要
#     SEMANTIC = "semantic"    # 语义记忆：长期知识
#
#
# @dataclass
# class MemoryEntry:
#     """记忆条目"""
#     content: str
#     level: MemoryLevel
#     timestamp: datetime = field(default_factory=datetime.now)
#     metadata: Dict[str, Any] = field(default_factory=dict)
#     importance: float = 0.5  # 重要性分数 0-1
#     access_count: int = 0
#     last_accessed: Optional[datetime] = None
#
#
# @dataclass
# class PatientProfile:
#     """患者档案（语义记忆）"""
#     patient_id: str
#     chief_complaints: List[str] = field(default_factory=list)
#     symptoms_history: List[str] = field(default_factory=list)
#     syndromes: List[str] = field(default_factory=list)
#     prescriptions: List[str] = field(default_factory=list)
#     allergies: List[str] = field(default_factory=list)
#     notes: str = ""
#     created_at: datetime = field(default_factory=datetime.now)
#     updated_at: datetime = field(default_factory=datetime.now)
#
#
# class HierarchicalMemory:
#     """
#     分层记忆系统
#
#     实现三层记忆架构，支持：
#     - 自动提升（Working → Episodic → Semantic）
#     - 基于重要性的保留策略
#     - TCM 特定的信息提取
#     """
#
#     # 各层容量限制
#     WORKING_CAPACITY = 10      # 最近 10 轮对话
#     EPISODIC_CAPACITY = 50     # 最多 50 条摘要
#     SEMANTIC_CAPACITY = 100    # 最多 100 条长期知识
#
#     # 重要性阈值
#     PROMOTION_THRESHOLD = 0.7  # 超过此阈值可提升到上层
#
#     # TCM 关键信息类型
#     TCM_INFO_TYPES = {
#         "症状": ["症状", "不适", "疼痛", "失眠", "乏力"],
#         "体征": ["舌象", "脉象", "面色", "体态"],
#         "诊断": ["辨证", "证型", "诊断"],
#         "治疗": ["方剂", "处方", "用药", "治法"],
#     }
#
#     def __init__(
#         self,
#         working_capacity: int = 10,
#         episodic_capacity: int = 50,
#         semantic_capacity: int = 100,
#     ):
#         """
#         初始化记忆系统
#
#         Args:
#             working_capacity: 工作记忆容量
#             episodic_capacity: 情景记忆容量
#             semantic_capacity: 语义记忆容量
#         """
#         self.working_capacity = working_capacity
#         self.episodic_capacity = episodic_capacity
#         self.semantic_capacity = semantic_capacity
#
#         # 三层记忆存储
#         self._working: List[MemoryEntry] = []
#         self._episodic: List[MemoryEntry] = []
#         self._semantic: List[MemoryEntry] = []
#
#         # 患者档案（语义记忆的结构化部分）
#         self._patient_profiles: Dict[str, PatientProfile] = {}
#
#     def add_to_working(
#         self,
#         content: str,
#         metadata: Optional[Dict[str, Any]] = None,
#         importance: float = 0.5,
#     ) -> MemoryEntry:
#         """
#         添加到工作记忆
#
#         Args:
#             content: 记忆内容
#             metadata: 元数据
#             importance: 重要性分数
#
#         Returns:
#             添加的记忆条目
#         """
#         entry = MemoryEntry(
#             content=content,
#             level=MemoryLevel.WORKING,
#             metadata=metadata or {},
#             importance=importance,
#         )
#
#         self._working.append(entry)
#
#         # 超容量时触发整理
#         if len(self._working) > self.working_capacity:
#             self._consolidate_working()
#
#         return entry
#
#     def _consolidate_working(self):
#         """整理工作记忆"""
#         if len(self._working) <= self.working_capacity:
#             return
#
#         # 按重要性排序
#         sorted_entries = sorted(
#             self._working,
#             key=lambda e: (e.importance, e.timestamp),
#             reverse=True
#         )
#
#         # 保留重要的
#         to_keep = sorted_entries[:self.working_capacity]
#         to_promote = sorted_entries[self.working_capacity:]
#
#         # 提升重要条目到情景记忆
#         for entry in to_promote:
#             if entry.importance >= self.PROMOTION_THRESHOLD:
#                 self._promote_to_episodic(entry)
#
#         self._working = to_keep
#
#     def _promote_to_episodic(self, entry: MemoryEntry):
#         """提升到情景记忆"""
#         promoted = MemoryEntry(
#             content=entry.content,
#             level=MemoryLevel.EPISODIC,
#             timestamp=entry.timestamp,
#             metadata=entry.metadata,
#             importance=entry.importance,
#         )
#
#         self._episodic.append(promoted)
#
#         # 超容量时整理情景记忆
#         if len(self._episodic) > self.episodic_capacity:
#             self._consolidate_episodic()
#
#     def _consolidate_episodic(self):
#         """整理情景记忆"""
#         if len(self._episodic) <= self.episodic_capacity:
#             return
#
#         # 合并相似条目
#         merged = self._merge_similar_entries(self._episodic)
#
#         # 按重要性和访问频率排序
#         sorted_entries = sorted(
#             merged,
#             key=lambda e: (e.importance * 0.6 + (e.access_count / 10) * 0.4),
#             reverse=True
#         )
#
#         # 保留重要的
#         to_keep = sorted_entries[:self.episodic_capacity]
#         to_promote = sorted_entries[self.episodic_capacity:]
#
#         # 提升到语义记忆
#         for entry in to_promote:
#             if entry.importance >= self.PROMOTION_THRESHOLD:
#                 self._promote_to_semantic(entry)
#
#         self._episodic = to_keep
#
#     def _promote_to_semantic(self, entry: MemoryEntry):
#         """提升到语义记忆"""
#         # 提取结构化信息
#         tcm_info = self._extract_tcm_info(entry.content)
#
#         promoted = MemoryEntry(
#             content=entry.content,
#             level=MemoryLevel.SEMANTIC,
#             timestamp=entry.timestamp,
#             metadata={**entry.metadata, "tcm_info": tcm_info},
#             importance=entry.importance,
#         )
#
#         self._semantic.append(promoted)
#
#         # 语义记忆容量控制
#         if len(self._semantic) > self.semantic_capacity:
#             # 移除最旧且最不重要的
#             self._semantic.sort(
#                 key=lambda e: (e.importance, e.access_count),
#                 reverse=True
#             )
#             self._semantic = self._semantic[:self.semantic_capacity]
#
#     def _merge_similar_entries(
#         self,
#         entries: List[MemoryEntry]
#     ) -> List[MemoryEntry]:
#         """合并相似条目"""
#         # 简单实现：基于关键词重叠度
#         merged = []
#         used = set()
#
#         for i, entry in enumerate(entries):
#             if i in used:
#                 continue
#
#             similar_group = [entry]
#             entry_keywords = set(self._extract_keywords(entry.content))
#
#             for j, other in enumerate(entries[i + 1:], i + 1):
#                 if j in used:
#                     continue
#
#                 other_keywords = set(self._extract_keywords(other.content))
#                 overlap = len(entry_keywords & other_keywords)
#
#                 if overlap > 3:  # 超过 3 个关键词重叠
#                     similar_group.append(other)
#                     used.add(j)
#
#             # 合并组内条目
#             if len(similar_group) > 1:
#                 merged_entry = self._merge_group(similar_group)
#                 merged.append(merged_entry)
#             else:
#                 merged.append(entry)
#
#             used.add(i)
#
#         return merged
#
#     def _merge_group(self, group: List[MemoryEntry]) -> MemoryEntry:
#         """合并一组相似条目"""
#         # 取重要性最高的作为基础
#         base = max(group, key=lambda e: e.importance)
#
#         # 合并内容（简单拼接）
#         contents = [e.content for e in group]
#         merged_content = "\n---\n".join(contents)
#
#         # 合并元数据
#         merged_metadata = {}
#         for e in group:
#             merged_metadata.update(e.metadata)
#
#         return MemoryEntry(
#             content=merged_content,
#             level=base.level,
#             timestamp=max(e.timestamp for e in group),
#             metadata=merged_metadata,
#             importance=max(e.importance for e in group),
#             access_count=sum(e.access_count for e in group),
#         )
#
#     def _extract_keywords(self, text: str) -> List[str]:
#         """提取关键词"""
#         # 简单实现：提取中医关键词
#         keywords = []
#         for category, terms in self.TCM_INFO_TYPES.items():
#             for term in terms:
#                 if term in text:
#                     keywords.append(term)
#
#         # 提取名词短语（简化版）
#         # 这里可以接入分词器
#         return keywords
#
#     def _extract_tcm_info(self, text: str) -> Dict[str, List[str]]:
#         """提取 TCM 结构化信息"""
#         result = {}
#
#         for category, terms in self.TCM_INFO_TYPES.items():
#             found = []
#             for term in terms:
#                 if term in text:
#                     # 提取包含该词的句子
#                     sentences = text.split("。")
#                     for sent in sentences:
#                         if term in sent and sent.strip():
#                             found.append(sent.strip())
#                             break
#
#             if found:
#                 result[category] = found
#
#         return result
#
#     def retrieve(
#         self,
#         query: str,
#         levels: Optional[List[MemoryLevel]] = None,
#         limit: int = 5,
#     ) -> List[MemoryEntry]:
#         """
#         检索记忆
#
#         Args:
#             query: 查询内容
#             levels: 要检索的层级（默认全部）
#             limit: 返回数量限制
#
#         Returns:
#             匹配的记忆条目
#         """
#         levels = levels or [MemoryLevel.WORKING, MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC]
#
#         candidates = []
#
#         if MemoryLevel.WORKING in levels:
#             candidates.extend(self._working)
#         if MemoryLevel.EPISODIC in levels:
#             candidates.extend(self._episodic)
#         if MemoryLevel.SEMANTIC in levels:
#             candidates.extend(self._semantic)
#
#         # 简单相关性评分
#         query_keywords = set(self._extract_keywords(query))
#
#         scored = []
#         for entry in candidates:
#             entry_keywords = set(self._extract_keywords(entry.content))
#             overlap = len(query_keywords & entry_keywords)
#             # 考虑文本包含
#             text_match = 1 if query in entry.content else 0
#
#             score = overlap * 0.5 + text_match * 0.3 + entry.importance * 0.2
#             scored.append((score, entry))
#
#         # 排序并返回
#         scored.sort(key=lambda x: x[0], reverse=True)
#
#         results = []
#         for score, entry in scored[:limit]:
#             # 更新访问信息
#             entry.access_count += 1
#             entry.last_accessed = datetime.now()
#             results.append(entry)
#
#         return results
#
#     def get_context_for_prompt(
#         self,
#         current_query: str,
#         max_tokens: int = 500,
#     ) -> str:
#         """
#         获取用于提示词的上下文
#
#         Args:
#             current_query: 当前查询
#             max_tokens: 最大 token 数
#
#         Returns:
#             格式化的上下文字符串
#         """
#         # 检索相关记忆
#         relevant = self.retrieve(current_query, limit=10)
#
#         if not relevant:
#             return ""
#
#         # 按层级组织
#         by_level = {
#             MemoryLevel.SEMANTIC: [],
#             MemoryLevel.EPISODIC: [],
#             MemoryLevel.WORKING: [],
#         }
#
#         for entry in relevant:
#             by_level[entry.level].append(entry)
#
#         # 构建上下文
#         parts = []
#
#         if by_level[MemoryLevel.SEMANTIC]:
#             parts.append("【长期记录】")
#             for entry in by_level[MemoryLevel.SEMANTIC][:2]:
#                 parts.append(f"- {entry.content[:100]}...")
#
#         if by_level[MemoryLevel.EPISODIC]:
#             parts.append("【历史摘要】")
#             for entry in by_level[MemoryLevel.EPISODIC][:3]:
#                 parts.append(f"- {entry.content[:80]}...")
#
#         if by_level[MemoryLevel.WORKING]:
#             parts.append("【近期对话】")
#             for entry in by_level[MemoryLevel.WORKING][:3]:
#                 parts.append(f"- {entry.content[:60]}...")
#
#         context = "\n".join(parts)
#
#         # Token 限制
#         estimated_tokens = self._estimate_tokens(context)
#         if estimated_tokens > max_tokens:
#             # 简单截断
#             ratio = max_tokens / estimated_tokens
#             context = context[:int(len(context) * ratio)]
#
#         return context
#
#     def _estimate_tokens(self, text: str) -> int:
#         """估算 token 数量"""
#         if not text:
#             return 0
#         chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
#         other_chars = len(text) - chinese_chars
#         return int(chinese_chars / 1.5 + other_chars / 4)
#
#     def update_patient_profile(
#         self,
#         patient_id: str,
#         **updates
#     ) -> PatientProfile:
#         """
#         更新患者档案
#
#         Args:
#             patient_id: 患者ID
#             **updates: 更新字段
#
#         Returns:
#             更新后的档案
#         """
#         if patient_id not in self._patient_profiles:
#             self._patient_profiles[patient_id] = PatientProfile(
#                 patient_id=patient_id
#             )
#
#         profile = self._patient_profiles[patient_id]
#
#         for key, value in updates.items():
#             if hasattr(profile, key):
#                 current = getattr(profile, key)
#                 if isinstance(current, list) and isinstance(value, str):
#                     current.append(value)
#                 elif isinstance(current, list) and isinstance(value, list):
#                     current.extend(value)
#                 else:
#                     setattr(profile, key, value)
#
#         profile.updated_at = datetime.now()
#         return profile
#
#     def get_patient_profile(self, patient_id: str) -> Optional[PatientProfile]:
#         """获取患者档案"""
#         return self._patient_profiles.get(patient_id)
#
#     def get_stats(self) -> Dict[str, Any]:
#         """获取记忆系统统计"""
#         return {
#             "working_count": len(self._working),
#             "working_capacity": self.working_capacity,
#             "episodic_count": len(self._episodic),
#             "episodic_capacity": self.episodic_capacity,
#             "semantic_count": len(self._semantic),
#             "semantic_capacity": self.semantic_capacity,
#             "patient_profiles": len(self._patient_profiles),
#         }
#
#     def clear_working(self):
#         """清空工作记忆"""
#         self._working = []
#
#     def export_to_dict(self) -> Dict[str, Any]:
#         """导出记忆到字典"""
#         return {
#             "working": [
#                 {
#                     "content": e.content,
#                     "importance": e.importance,
#                     "timestamp": e.timestamp.isoformat(),
#                     "metadata": e.metadata,
#                 }
#                 for e in self._working
#             ],
#             "episodic": [
#                 {
#                     "content": e.content,
#                     "importance": e.importance,
#                     "timestamp": e.timestamp.isoformat(),
#                     "metadata": e.metadata,
#                 }
#                 for e in self._episodic
#             ],
#             "semantic": [
#                 {
#                     "content": e.content,
#                     "importance": e.importance,
#                     "timestamp": e.timestamp.isoformat(),
#                     "metadata": e.metadata,
#                 }
#                 for e in self._semantic
#             ],
#         }

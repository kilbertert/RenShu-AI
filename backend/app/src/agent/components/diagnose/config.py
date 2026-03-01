"""
诊断子图配置参数
"""

from typing import List


class DiagnoseConfig:
    """诊断子图配置"""

    # === 追问控制 ===
    MAX_FOLLOW_UP_ROUNDS: int = 4          # 最大追问轮数（从5改为4）
    MIN_REQUIRED_CATEGORIES: int = 4       # 最少需要的信息类别数

    # === 复杂度阈值 ===
    SIMPLE_THRESHOLD: int = 3              # 简单阈值（≤3分）
    MODERATE_THRESHOLD: int = 6            # 中等阈值（≤6分）
    # > 6分为复杂

    # === 多模态 ===
    ENABLE_TONGUE_REQUEST: bool = True     # 是否启用舌像请求
    ENABLE_REPORT_REQUEST: bool = True     # 是否启用报告请求
    TONGUE_REQUEST_PROBABILITY: float = 0.7  # 请求舌像的概率（满足条件时）

    # === LLM 参数 ===
    COLLECTION_TEMPERATURE: float = 0.7    # 信息收集阶段温度
    DIAGNOSIS_TEMPERATURE: float = 0.3     # 辨证阶段温度（更确定性）
    MAX_TOKENS: int = 4096                 # 最大token数（避免长文本截断）

    # === DeepSearch ===
    DEEPSEARCH_MAX_ITERATIONS: int = 3     # DeepSearch 最大迭代次数
    DEEPSEARCH_SOURCES: List[str] = [      # DeepSearch 数据源
        "knowledge_graph",
        "vector_store",
        "tcm_classics"
    ]

    # === 错误响应 ===
    ERROR_RESPONSES = {
        "collection_failed": "抱歉，我没有完全理解您的描述。能否再详细说明一下您的症状？",
        "tongue_analysis_failed": "舌像分析暂时出现问题，我们继续通过问诊了解您的情况。",
        "diagnosis_failed": "抱歉，根据目前的信息，我无法给出确切的判断。建议您前往医院进行详细检查。",
        "timeout": "分析时间较长，请稍候。如果长时间没有响应，请刷新重试。",
    }


# 全局配置实例
diagnose_config = DiagnoseConfig()

"""
语言模型模块

提供模型实体、供应商适配器等功能。
"""


from .model_stats import ModelStatsManager, model_stats_manager

__all__ = [
    # 实体类

    # 统计管理
    'ModelStatsManager',
    'model_stats_manager',
    # 默认配置

]

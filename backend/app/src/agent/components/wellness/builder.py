from langgraph.graph import StateGraph, START, END
from .states import WellnessInputState, WellnessOverallState, WellnessOutputState
from .router import wellness_router_node, route_wellness
from .handlers import (
    handle_wellness_seasonal,
    handle_wellness_daily,
    handle_wellness_constitution,
    handle_wellness_general
)

def create_wellness_graph():
    """创建养生子图"""
    workflow = StateGraph(WellnessOverallState, input=WellnessInputState, output=WellnessOutputState)
    
    # 添加节点
    workflow.add_node("wellness_router_node", wellness_router_node)
    workflow.add_node("handle_wellness_seasonal", handle_wellness_seasonal)
    workflow.add_node("handle_wellness_daily", handle_wellness_daily)
    workflow.add_node("handle_wellness_constitution", handle_wellness_constitution)
    workflow.add_node("handle_wellness_general", handle_wellness_general)
    
    # 添加边
    workflow.add_edge(START, "wellness_router_node")
    workflow.add_conditional_edges(
        "wellness_router_node",
        route_wellness,
        {
            "handle_wellness_seasonal": "handle_wellness_seasonal",
            "handle_wellness_daily": "handle_wellness_daily",
            "handle_wellness_constitution": "handle_wellness_constitution",
            "handle_wellness_general": "handle_wellness_general",
        }
    )
    
    # 汇合到结束
    workflow.add_edge("handle_wellness_seasonal", END)
    workflow.add_edge("handle_wellness_daily", END)
    workflow.add_edge("handle_wellness_constitution", END)
    workflow.add_edge("handle_wellness_general", END)
    
    return workflow.compile()

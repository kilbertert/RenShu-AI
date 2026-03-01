from .states import WellnessOverallState

async def wellness_router_node(state: WellnessOverallState) -> dict:
    """养生流二级路由节点"""
    sub_type = state.get("sub_type", "")
    
    if sub_type == "seasonal":
        return {"steps": ["养生流: 进入 [季节养生] 分支"]}
    elif sub_type == "constitution":
        return {"steps": ["养生流: 进入 [体质调理] 分支"]}
    elif sub_type == "daily":
        return {"steps": ["养生流: 进入 [日常科普] 分支"]}
    else:
        return {"steps": ["养生流: 进入 [复杂/通用养生] 分支"]}

def route_wellness(state: WellnessOverallState) -> str:
    """养生流二级路由逻辑"""
    sub_type = state.get("sub_type", "")
    if sub_type == "seasonal":
        return "handle_wellness_seasonal"
    elif sub_type == "daily":
        return "handle_wellness_daily"
    elif sub_type == "constitution":
        return "handle_wellness_constitution"
    else:
        return "handle_wellness_general"

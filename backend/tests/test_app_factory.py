"""FastAPI 应用工厂的基础回归测试。"""

from main import app


def test_api_routes_are_registered_when_app_is_created() -> None:
    """业务路由应在应用创建时存在，而不是等数据库启动后才动态注册。"""
    paths = {getattr(route, "path", "") for route in app.routes}

    assert "/api/v1/chat/generate" in paths
    assert "/api/v1/cases" in paths
    assert "/api/v1/users/login" in paths

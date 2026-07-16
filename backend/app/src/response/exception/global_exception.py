"""
全局异常处理器

注意：此处不再注册 @app.middleware("http") 形式的 BaseHTTPMiddleware。
BaseHTTPMiddleware 在异常路径下会拦截响应并导致 CORS 头丢失。
request_id / client_ip 已由 ResponseMiddleware (纯 ASGI) 统一处理。
"""

import uuid

from fastapi import FastAPI
from starlette.responses import JSONResponse

from fastapi import FastAPI, Request

from app.src.response.exception.exceptions import APIException, InternalServerException
from app.src.response.response_factory import response_factory
from app.src.response.response_middleware import request_context
from app.src.utils import get_logger


class GlobalReOrExHandler:
    """
    全局异常处理器
    """

    def __init__(self, app: FastAPI):
        self.app = app
        self.register_error_handler()
        self.logger = get_logger("app")

    def register_error_handler(self):
        """
        注册不同类型的异常处理函数到FasstAPi中
        """

        # 全局异常处理器
        @self.app.exception_handler(APIException)
        async def api_exception_handler(request: Request, exc: APIException):
            """处理API异常"""
            self.logger.error(f"API异常: {exc.message}")
            response = response_factory.from_exception(
                exc,
                request_id=request_context.get_request_id(),
                host_id=request.headers.get("host", "localhost")
            )

            return JSONResponse(
                status_code=exc.http_status if hasattr(exc, 'http_status') else 400,
                content=response.dict()
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """处理通用异常"""

            self.logger.critical(f"未处理的异常: {str(exc)}")
            # 包装为内部服务器异常
            wrapped_exc = InternalServerException(
                message="服务器内部错误",
                details={"original_error": str(exc)}
            )

            response = response_factory.from_exception(
                wrapped_exc,
                request_id=request_context.get_request_id(),
                host_id=request.headers.get("host", "localhost")
            )

            return JSONResponse(
                status_code=500,
                content=response.dict()
            )

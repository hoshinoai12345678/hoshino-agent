"""认证 API（注册/登录/登出/当前用户）

接口：
- POST /api/auth/register  注册
- POST /api/auth/login     登录（设置签名 Cookie）
- POST /api/auth/logout    登出（清除 Cookie）
- GET  /api/auth/me        获取当前登录用户
"""
import os

from fastapi import APIRouter, Cookie, HTTPException, Response, status
from pydantic import BaseModel

from config import AUTH_COOKIE_MAX_AGE, AUTH_COOKIE_NAME, DEFAULT_SESSION_ID
from core.auth import (authenticate, create_session_token, register, session_id_from_username,
                       verify_session_token)
from core.logger import get_logger

logger = get_logger(__name__)

# Cookie secure 标志：HTTPS 部署时设 COOKIE_SECURE=1，本地 HTTP 调试保持 0
_COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0").lower() == "1"

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def do_register(req: RegisterRequest):
    """注册新用户"""
    success, msg = register(req.username, req.password)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return {"status": "ok", "message": msg, "username": req.username}


@router.post("/login")
async def do_login(req: LoginRequest, response: Response):
    """登录并设置签名 Cookie"""
    success, msg = authenticate(req.username, req.password)
    if not success:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)

    token = create_session_token(req.username)
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=AUTH_COOKIE_MAX_AGE,
        httponly=True,         # 防止 JS 读取（XSS 防护）
        samesite="lax",        # 防 CSRF
        secure=_COOKIE_SECURE,  # HTTPS 部署设 COOKIE_SECURE=1
        path="/",
    )
    logger.info(f"用户登录: {req.username}")
    return {
        "status": "ok",
        "message": msg,
        "username": req.username,
        "session_id": session_id_from_username(req.username),
    }


@router.post("/logout")
async def do_logout(response: Response):
    """登出，清除 Cookie"""
    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")
    return {"status": "ok", "message": "已登出"}


@router.get("/me")
async def get_current_user(session: str = Cookie(default=None, alias=AUTH_COOKIE_NAME)):
    """获取当前登录用户

    Returns:
        {"username": "...", "session_id": "web_..."} 或 401
    """
    username = verify_session_token(session)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录或登录已过期")
    return {
        "username": username,
        "session_id": session_id_from_username(username),
    }


def get_session_id_from_cookie(session: str = None) -> str:
    """从 Cookie 提取 session_id（供 chat API 调用）

    未登录时 fallback 到 DEFAULT_SESSION_ID，保证向后兼容。
    """
    username = verify_session_token(session)
    if username:
        return session_id_from_username(username)
    return DEFAULT_SESSION_ID

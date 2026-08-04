"""用户认证模块

简单的账号密码认证，用于 Web 通道多用户隔离。
- 用户数据存 JSON 文件（data/users.json）
- 密码用 PBKDF2-HMAC-SHA256 + 随机 salt 哈希
- 登录态用 itsdangerous 签名 Cookie（无需服务端 session 存储）

session_id 复用现有的多用户隔离机制：
- 登录后 session_id = "web_" + username
- 未登录 fallback 到 DEFAULT_SESSION_ID
"""
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from config import (AUTH_COOKIE_MAX_AGE, AUTH_SECRET_KEY, DATA_DIR, USERS_FILE)
from core.logger import get_logger

logger = get_logger(__name__)

# 用户名白名单字符（字母数字下划线横线，与 session_id 校验保持一致）
# 长度 1-32，允许简单用户名如 "1"、"a" 等
_USERNAME_MIN_LEN = 1
_USERNAME_MAX_LEN = 32
_PASSWORD_MIN_LEN = 1

# 签名 Cookie 序列化器（全局单例）
_serializer: Optional[URLSafeTimedSerializer] = None


def _get_serializer() -> URLSafeTimedSerializer:
    global _serializer
    if _serializer is None:
        _serializer = URLSafeTimedSerializer(AUTH_SECRET_KEY, salt="hoshino-auth")
    return _serializer


def validate_username(username: str) -> tuple[bool, str]:
    """校验用户名合法性（防注入，不限制复杂度）"""
    if not username:
        return False, "用户名不能为空"
    if len(username) < _USERNAME_MIN_LEN:
        return False, f"用户名至少 {_USERNAME_MIN_LEN} 个字符"
    if len(username) > _USERNAME_MAX_LEN:
        return False, f"用户名最多 {_USERNAME_MAX_LEN} 个字符"
    # 白名单字符校验，防止 session_id 表名注入
    if not all(c.isalnum() or c in "_-" for c in username):
        return False, "用户名仅允许字母数字下划线横线"
    return True, ""


def validate_password(password: str) -> tuple[bool, str]:
    """校验密码合法性（不限制复杂度，允许简单密码如 '1'）"""
    if not password:
        return False, "密码不能为空"
    if len(password) < _PASSWORD_MIN_LEN:
        return False, f"密码至少 {_PASSWORD_MIN_LEN} 个字符"
    return True, ""


def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """PBKDF2-HMAC-SHA256 哈希密码

    Args:
        password: 明文密码
        salt: 盐（None 时自动生成）
    Returns:
        (hash_hex, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    # 100000 次迭代，符合 OWASP 2023 推荐
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return dk.hex(), salt


def _verify_password(password: str, hash_hex: str, salt: str) -> bool:
    """验证密码"""
    computed_hash, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed_hash, hash_hex)


def _load_users() -> dict:
    """加载用户数据（JSON 文件）

    Returns:
        {"username": {"password_hash": "...", "salt": "...", "created_at": ...}}
    """
    try:
        if not os.path.exists(USERS_FILE):
            return {}
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载用户数据失败: {e}")
        return {}


def _save_users(users: dict) -> bool:
    """保存用户数据到 JSON 文件"""
    try:
        # 确保目录存在
        Path(USERS_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"保存用户数据失败: {e}")
        return False


def register(username: str, password: str) -> tuple[bool, str]:
    """注册新用户

    Returns:
        (success, message)
    """
    ok, msg = validate_username(username)
    if not ok:
        return False, msg
    ok, msg = validate_password(password)
    if not ok:
        return False, msg

    users = _load_users()
    if username in users:
        return False, "用户名已存在"

    hash_hex, salt = _hash_password(password)
    users[username] = {
        "password_hash": hash_hex,
        "salt": salt,
        "created_at": time.time(),
    }
    if _save_users(users):
        logger.info(f"新用户注册: {username}")
        return True, "注册成功"
    return False, "保存用户数据失败"


def authenticate(username: str, password: str) -> tuple[bool, str]:
    """验证用户登录

    Returns:
        (success, message)
    """
    users = _load_users()
    user = users.get(username)
    if not user:
        return False, "用户名或密码错误"

    if not _verify_password(password, user["password_hash"], user["salt"]):
        return False, "用户名或密码错误"

    return True, "登录成功"


def create_session_token(username: str) -> str:
    """签发签名 Cookie token

    token 内容为 {"username": "xxx"}，带时间戳签名。
    验证时自动检查有效期（max_age）。
    """
    serializer = _get_serializer()
    return serializer.dumps({"username": username})


def verify_session_token(token: str) -> Optional[str]:
    """验证签名 Cookie token，返回 username（失败返回 None）

    自动检查签名有效性和过期时间。
    """
    if not token:
        return None
    serializer = _get_serializer()
    try:
        data = serializer.loads(token, max_age=AUTH_COOKIE_MAX_AGE)
        return data.get("username")
    except SignatureExpired:
        logger.warning("Cookie 已过期")
        return None
    except BadSignature:
        logger.warning("Cookie 签名无效")
        return None
    except Exception as e:
        logger.warning(f"Cookie 验证失败: {e}")
        return None


def session_id_from_username(username: str) -> str:
    """用户名转 session_id（复用现有隔离机制）

    格式：web_{username}，与 Telegram 的 tg_{chat_id} 隔离
    """
    return f"web_{username}"

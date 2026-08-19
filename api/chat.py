"""聊天 API（SSE 流式响应，多角色支持）

接口：
- GET  /api/characters    列出可用角色
- POST /api/chat          流式聊天（可指定角色）
- GET  /api/state         获取 Agent 状态
- GET  /api/history       获取工作记忆中的对话历史（用于刷新页面后恢复聊天框）
- POST /api/reset         重置 Agent（清空工作记忆和情绪，保留长期记忆）
- POST /api/forget        忘记用户（清空长期记忆，保留角色知识库）
- GET  /api/memories      获取历史记忆

角色隔离设计：
- Agent 实例按 (session_id, character_id) 双维度缓存，不同用户 × 不同角色完全独立。
- character_id 通过 query 参数或 ChatRequest.body 传入，默认 DEFAULT_CHARACTER_ID。
- session_id 仍由 Cookie 登录态解析，与 character_id 正交。
"""
import asyncio
import json
from fastapi import APIRouter, Cookie, Query
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from config import AUTH_COOKIE_NAME, DEFAULT_SESSION_ID, DEFAULT_CHARACTER_ID, list_available_characters
from agent.hoshino_agent import HoshinoAgent
from core.persona import Persona
from api.auth import get_session_id_from_cookie

router = APIRouter(prefix="/api", tags=["chat"])

# Agent 实例缓存：key = (session_id, character_id)，实现 用户×角色 双维度隔离
_agents: dict[tuple[str, str], HoshinoAgent] = {}
_agents_lock = asyncio.Lock()


async def get_agent(session_id: str, character_id: str = DEFAULT_CHARACTER_ID) -> HoshinoAgent:
    """获取或创建指定 (session, character) 的 Agent（协程安全）"""
    key = (session_id, character_id)
    async with _agents_lock:
        if key not in _agents:
            _agents[key] = HoshinoAgent(session_id=session_id, character_id=character_id)
        return _agents[key]


def resolve_session_id(session: str = None, session_id_query: str = None) -> str:
    """解析 session_id：优先 Cookie 登录态，其次 query 参数，最后 default

    session: Cookie 中的签名 token
    session_id_query: URL query 参数（保留向后兼容，如 Telegram 通道不用走 Cookie）
    """
    # 1. 优先从 Cookie 解析登录用户
    if session:
        sid = get_session_id_from_cookie(session)
        if sid != DEFAULT_SESSION_ID:
            return sid
    # 2. fallback 到 query 参数
    if session_id_query:
        return session_id_query
    # 3. 默认
    return DEFAULT_SESSION_ID


def resolve_character_id(body_char: str = None, query_char: str = None) -> str:
    """解析 character_id：优先 body，其次 query，最后默认角色

    body_char: ChatRequest.body 中的 character_id（前端切换角色时传入）
    query_char: URL query 参数（兼容非 body 场景，如 GET 请求）
    """
    if body_char:
        return body_char
    if query_char:
        return query_char
    return DEFAULT_CHARACTER_ID


@router.get("/characters")
async def list_characters():
    """列出所有可用角色（前端角色选择器用）

    Returns:
        characters: 角色列表（按 character_id 排序）
        default: 默认角色 ID
    """
    character_ids = list_available_characters()
    characters = []
    for cid in character_ids:
        try:
            persona = Persona(character_id=cid)
            characters.append({
                "id": cid,
                "name": persona.name,
                "source": persona.source,
                "occupation": persona.occupation,
            })
        except Exception:
            # 跳过损坏的 persona.json
            continue
    return {"characters": characters, "default": DEFAULT_CHARACTER_ID}


class ChatRequest(BaseModel):
    message: str
    dev_mode: bool = False  # 是否开启开发者模式（输出思考链），由前端开关控制
    character_id: str = None  # 角色ID（可选，默认走 query 或 DEFAULT_CHARACTER_ID）


@router.post("/chat")
async def chat(
    req: ChatRequest,
    session: str = Cookie(default=None, alias=AUTH_COOKIE_NAME),
    session_id: str = Query(default=None),
    character_id: str = Query(default=None),
):
    """流式聊天接口（SSE）

    角色选择优先级：body.character_id > query.character_id > DEFAULT_CHARACTER_ID
    """
    sid = resolve_session_id(session, session_id)
    cid = resolve_character_id(req.character_id, character_id)
    agent = await get_agent(sid, cid)

    async def event_generator():
        try:
            async for chunk in agent.chat_stream(req.message, dev_mode=req.dev_mode):
                yield {"event": "message", "data": json.dumps(chunk, ensure_ascii=False)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps(
                {"type": "error", "content": str(e)}, ensure_ascii=False
            )}

    return EventSourceResponse(event_generator())


@router.get("/state")
async def get_state(
    session: str = Cookie(default=None, alias=AUTH_COOKIE_NAME),
    session_id: str = Query(default=None),
    character_id: str = Query(default=None),
):
    """获取 Agent 当前状态"""
    sid = resolve_session_id(session, session_id)
    cid = resolve_character_id(query_char=character_id)
    agent = await get_agent(sid, cid)
    return agent.get_state()


@router.get("/history")
async def get_history(
    session: str = Cookie(default=None, alias=AUTH_COOKIE_NAME),
    session_id: str = Query(default=None),
    character_id: str = Query(default=None),
):
    """获取工作记忆中的对话历史

    用于刷新页面后恢复前端聊天框显示。返回最近 WORKING_MEMORY_SIZE 轮对话。
    """
    sid = resolve_session_id(session, session_id)
    cid = resolve_character_id(query_char=character_id)
    agent = await get_agent(sid, cid)
    messages = agent.working.get_messages()
    return {
        "messages": [
            {
                "role": m["role"],
                "content": m["content"],
                "timestamp": m.get("timestamp", 0),
            }
            for m in messages
        ],
        "count": len(messages),
    }


@router.post("/reset")
async def reset_agent(
    session: str = Cookie(default=None, alias=AUTH_COOKIE_NAME),
    session_id: str = Query(default=None),
    character_id: str = Query(default=None),
):
    """重置 Agent（清空工作记忆和情绪，保留长期记忆）"""
    sid = resolve_session_id(session, session_id)
    cid = resolve_character_id(query_char=character_id)
    agent = await get_agent(sid, cid)
    agent.reset()
    return {"status": "ok", "message": "Agent 已重置"}


@router.post("/forget")
async def forget_user(
    session: str = Cookie(default=None, alias=AUTH_COOKIE_NAME),
    session_id: str = Query(default=None),
    character_id: str = Query(default=None),
):
    """忘记用户（清空长期记忆 + 情绪好感，保留角色知识库和工作记忆）

    清空当前 (session, character) 的：
    - 情景记忆（episodic）：用户说过的具体事件
    - 语义记忆（semantic）：用户画像
    - 情绪状态（emotion）：P/A/D 和好感度重置到角色 baseline（对陌生人的初见态度）
    不影响：其他 session/角色、角色知识库索引、工作记忆。
    """
    sid = resolve_session_id(session, session_id)
    cid = resolve_character_id(query_char=character_id)
    agent = await get_agent(sid, cid)
    ep_deleted = agent.episodic.clear()
    sem_deleted = agent.semantic.clear()
    agent.emotion.reset()
    return {
        "status": "ok",
        "message": "长期记忆与情绪好感已清空",
        "episodic_deleted": ep_deleted,
        "semantic_deleted": sem_deleted,
    }


@router.get("/memories")
async def get_memories(
    limit: int = 20,
    offset: int = 0,
    session: str = Cookie(default=None, alias=AUTH_COOKIE_NAME),
    session_id: str = Query(default=None),
    character_id: str = Query(default=None),
):
    """获取情景记忆列表（支持分页）

    Args:
        limit: 返回条数
        offset: 偏移量（按时间倒序计）
    Returns:
        episodic: 情景记忆列表（分页）
        semantic: 用户画像（全量，通常条数不多）
        episodic_count: 情景记忆总数（用于判断是否还有更多）
        semantic_count: 语义记忆总数
    """
    sid = resolve_session_id(session, session_id)
    cid = resolve_character_id(query_char=character_id)
    agent = await get_agent(sid, cid)
    memories = agent.episodic.get_recent(limit=limit, offset=offset)
    profile = agent.semantic.get_all()
    return {
        "episodic": memories,
        "semantic": [dict(p) for p in profile],
        "episodic_count": agent.episodic.count(),
        "semantic_count": agent.semantic.count(),
        "offset": offset,
        "limit": limit,
    }

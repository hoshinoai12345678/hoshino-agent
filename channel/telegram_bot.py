"""Telegram Bot 通道

复用 HoshinoAgent 内核，通过 Telegram Bot API 对外提供聊天服务。
- session_id = tg_{chat_id}，与 Web 通道隔离
- 支持 /start /reset /forget 命令
- 支持 HTTPS_PROXY 环境变量（国内访问 Telegram API 必需）
- 复用 Web 通道的 session 管理逻辑，保证 agent 实例协程安全

启动方式：
    set TG_BOT_TOKEN=8864641604:AAHyHJ-adKpic50JRvuW15OAtfStaLuIP4U
    set HTTPS_PROXY=http://127.0.0.1:7890
    python -m channel.telegram_bot

国内需配代理：
    
"""
import asyncio
import os
import sys

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# 确保项目根目录在 sys.path 中（支持 python -m channel.telegram_bot 启动）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.hoshino_agent import HoshinoAgent
from rag.indexer import KnowledgeIndexer
from config import TG_BOT_TOKEN, TG_HTTPS_PROXY
from core.logger import get_logger

logger = get_logger(__name__)

# session 管理器：TG 通道独立维护，session_id = tg_{chat_id}
_agents: dict[str, HoshinoAgent] = {}
_agents_lock = asyncio.Lock()


async def get_agent(session_id: str) -> HoshinoAgent:
    """获取或创建指定 session 的 Agent（协程安全）"""
    async with _agents_lock:
        if session_id not in _agents:
            _agents[session_id] = HoshinoAgent(session_id=session_id)
        return _agents[session_id]


def tg_session_id(chat_id: int) -> str:
    """TG chat_id 转 session_id（校验合法性，仅含数字）"""
    return f"tg_{chat_id}"


# ---- 命令处理 ----

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/start - 介绍与欢迎"""
    await update.message.reply_text(
        "嗨~我是星野爱！♪✨\n"
        "直接跟我聊天就好啦~\n\n"
        "命令：\n"
        "/reset - 重新开始（清空短期记忆和情绪）\n"
        "/forget - 忘记你（清空长期记忆）\n"
        "/state - 查看当前状态"
    )


async def reset_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/reset - 清空工作记忆和情绪"""
    chat_id = update.effective_chat.id
    agent = await get_agent(tg_session_id(chat_id))
    agent.reset()
    await update.message.reply_text("爱重新回来啦！♪ 让我们重新开始吧~")


async def forget_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/forget - 清空长期记忆"""
    chat_id = update.effective_chat.id
    agent = await get_agent(tg_session_id(chat_id))
    ep_deleted = agent.episodic.clear()
    sem_deleted = agent.semantic.clear()
    await update.message.reply_text(
        f"咦……爱好像忘记了什么~♪\n"
        f"（已清空情景记忆 {ep_deleted} 条，用户画像 {sem_deleted} 条）"
    )


async def state_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/state - 查看状态"""
    chat_id = update.effective_chat.id
    agent = await get_agent(tg_session_id(chat_id))
    state = agent.get_state()
    emo = state["emotion"]
    await update.message.reply_text(
        f"【当前状态】\n"
        f"情绪：{emo.get('emotion_label', '')}\n"
        f"P/A/D：{emo['pleasure']:.2f} / {emo['arousal']:.2f} / {emo['dominance']:.2f}\n"
        f"好感度：{emo['favorability']}（{emo.get('favorability_level', '')}）\n"
        f"工作记忆：{state['working_memory_size']} 条\n"
        f"情景记忆：{state['episodic_count']} 条\n"
        f"用户画像：{state['semantic_count']} 条"
    )


async def chat_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """普通消息处理：调用 agent.chat 聚合后回复"""
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    session_id = tg_session_id(chat_id)
    agent = await get_agent(session_id)

    # 发送"正在输入"状态
    await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        # 调用非流式 chat（聚合完整回复，TG 不需要 SSE）
        result = await agent.chat(user_text)
        reply = result.get("reply", "")
        if not reply:
            reply = "诶……爱好像没听清呢，能再说一次吗？♪"

        # TG 单条消息上限 4096 字符，超长截断
        if len(reply) > 4000:
            reply = reply[:4000] + "\n\n（消息过长，已截断）"

        await update.message.reply_text(reply)

        # 记录工具调用和记忆操作（开发者可观察）
        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            tool_summary = "、".join(f"{t['name']}" for t in tool_calls)
            logger.info(f"TG session={session_id} 工具调用: {tool_summary}")

    except Exception as e:
        logger.error(f"TG 处理失败 session={session_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "呜……爱好像有点不舒服，能稍等一下再跟爱说话吗？💦"
        )


def build_application() -> Application:
    """构建 TG Bot Application（支持代理）"""
    if not TG_BOT_TOKEN:
        raise RuntimeError(
            "未设置 TG_BOT_TOKEN 环境变量。"
            "请从 @BotFather 获取 token 后设置：export TG_BOT_TOKEN=your_token"
        )

    builder = ApplicationBuilder().token(TG_BOT_TOKEN)

    # 代理配置（国内访问 Telegram API 必需）
    if TG_HTTPS_PROXY:
        logger.info(f"使用代理: {TG_HTTPS_PROXY}")
        builder = builder.proxy(TG_HTTPS_PROXY).get_updates_proxy(TG_HTTPS_PROXY)

    app = builder.build()

    # 注册命令
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("forget", forget_cmd))
    app.add_handler(CommandHandler("state", state_cmd))
    # 普通文本消息
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))

    return app


async def post_init(app: Application):
    """Bot 初始化后：构建知识索引"""
    indexer = KnowledgeIndexer()
    count = indexer.build_index()
    logger.info(f"角色知识索引就绪，共 {count} 个分块")
    me = await app.bot.get_me()
    logger.info(f"TG Bot 已启动：@{me.username} ({me.first_name})")


def main():
    """启动 TG Bot"""
    logger.info(f"TG_BOT_TOKEN: {'已设置' if TG_BOT_TOKEN else '未设置'}")
    logger.info(f"HTTPS_PROXY: {TG_HTTPS_PROXY or '未设置'}")

    app = build_application()
    app.post_init = post_init

    # pollup 模式，无需公网 IP
    logger.info("启动 Telegram Bot（long polling 模式）...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

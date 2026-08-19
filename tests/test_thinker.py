"""测试 Thinker 思考者 Agent

覆盖：
- 预筛逻辑：短消息 → 返回空（跳过 LLM）
- to_prompt_context：空思考返回空，有 inner_monologue 返回角色化独白
- 降级路径：LLM 没返回 inner_monologue 时用 strategy 拼
- tool_plan 规范化：过滤非法步骤、补全字段
- LLM 不可用降级
"""
import pytest

from agent.thinker import Thinker


def test_to_prompt_context_empty():
    """空思考或无内容返回空字符串"""
    t = Thinker()
    assert t.to_prompt_context({}) == ""
    # 无 inner_monologue 且无 strategy 也返回空
    assert t.to_prompt_context({"intent": "问候", "strategy": ""}) == ""


def test_to_prompt_context_with_monologue():
    """有 inner_monologue 时直接返回角色化独白"""
    t = Thinker()
    thought = {
        "intent": "用户想了解爱的口头禅",
        "user_emotion": "好奇",
        "needs_tools": True,
        "needs_plan": False,
        "tool_plan": [
            {"action": "search_knowledge", "target": "口头禅", "reason": "用户询问"},
            {"action": "reply", "target": "综合回答", "reason": "最终回复"},
        ],
        "strategy": "先检索知识再用可爱语气回答",
        "has_complex_signal": False,
        "inner_monologue": "哇，用户想了解爱的口头禅呀~让爱想想，先去翻翻自己的资料库，再用可爱的语气回答吧♪",
    }
    ctx = t.to_prompt_context(thought)
    assert "口头禅" in ctx
    assert "爱" in ctx
    # 不应出现机械标签
    assert "【内心思考】" not in ctx
    assert "【执行计划】" not in ctx


def test_to_prompt_context_multi_step_monologue():
    """多任务请求：inner_monologue 自然提到先做什么再做什么"""
    t = Thinker()
    thought = {
        "intent": "自我介绍+询问口头禅",
        "user_emotion": "友好",
        "needs_tools": True,
        "needs_plan": True,
        "tool_plan": [
            {"action": "save_memory", "target": "用户名小枫", "reason": "自我介绍"},
            {"action": "save_memory", "target": "喜欢草莓", "reason": "用户透露偏好"},
            {"action": "search_knowledge", "target": "口头禅", "reason": "用户询问"},
            {"action": "reply", "target": "综合回答", "reason": "最终回复"},
        ],
        "strategy": "先记住信息再回答",
        "has_complex_signal": True,
        "inner_monologue": "用户叫小枫，还喜欢草莓~爱要先记住小枫的名字和喜好，然后查查自己的口头禅，最后综合回答给小枫♪",
    }
    ctx = t.to_prompt_context(thought)
    assert "小枫" in ctx
    assert "爱" in ctx
    # 不应出现机械标签
    assert "【执行计划】" not in ctx
    assert "请按计划依次调用工具" not in ctx


def test_to_prompt_context_fallback_no_monologue():
    """LLM 没返回 inner_monologue 时，用 intent + strategy 降级拼独白"""
    t = Thinker(character_name="星野爱")
    thought = {
        "intent": "闲聊",
        "strategy": "轻松回应",
    }
    ctx = t.to_prompt_context(thought)
    assert "爱" in ctx
    assert "轻松回应" in ctx
    assert "闲聊" in ctx


@pytest.mark.asyncio
async def test_think_short_message_skip():
    """短消息（<10字）直接返回空，不调用 LLM"""
    t = Thinker()
    thought = await t.think("你好", "", "", "")
    assert thought == {}


@pytest.mark.asyncio
async def test_think_no_llm_fallback():
    """LLM 不可用时降级返回角色化独白（保证思考链总有内容）"""
    t = Thinker(character_name="星野爱")
    t._client = None
    thought = await t.think("请帮我分析今天的心情并且总结一下", "", "", "")
    # 降级路径：返回 _fallback_thought，含角色化 inner_monologue
    assert thought["inner_monologue"] != ""
    assert "爱" in thought["inner_monologue"]
    # 命中多任务关键词（"分析"+"总结"），独白应提到"好几步"
    assert thought["has_complex_signal"] is True
    assert "好几步" in thought["inner_monologue"]


@pytest.mark.asyncio
async def test_think_normalizes_tool_plan():
    """think 方法规范化 tool_plan：过滤非法步骤、补全字段"""
    t = Thinker()

    # mock LLM 返回带非法步骤的 JSON
    raw = ('{"intent":"测试","user_emotion":"中性","needs_tools":true,'
           '"tool_plan":[{"action":"save_memory","target":"名字","reason":"自我介绍"},'
           '{"bad":"step"},"not_dict"],"strategy":"记住后回答",'
           '"inner_monologue":"爱要记住用户的名字然后回答~"}')

    class FakeMsg:
        content = raw
    class FakeChoice:
        message = FakeMsg()
    class FakeResp:
        choices = [FakeChoice()]

    async def fake_create(*args, **kwargs):
        return FakeResp()

    class FakeCompletions:
        create = fake_create
    class FakeChat:
        completions = FakeCompletions()
    class FakeClient:
        chat = FakeChat()

    t._client = FakeClient()

    thought = await t.think("我叫小枫，请记住我的名字然后告诉我你的口头禅", "", "", "")
    assert thought["intent"] == "测试"
    assert thought["needs_tools"] is True
    assert thought["inner_monologue"] == "爱要记住用户的名字然后回答~"
    # 非法步骤被过滤，只保留 1 个合法步骤
    assert len(thought["tool_plan"]) == 1
    assert thought["tool_plan"][0]["action"] == "save_memory"
    assert thought["tool_plan"][0]["target"] == "名字"

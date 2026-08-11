"""LangChain 对比实现

用 LangChain 框架重写简化版 Agent，与手写版（agent/）形成对比。
仅实现 ReAct + 工具调用 + 基本对话，不含 Thinker/Reflector/情绪/三层记忆。

用途：验证 LangChain 框架的适用性，作为技术选型对比的实证。
主实现仍为手写版（agent/hoshino_agent.py），原因见 README.md。
"""

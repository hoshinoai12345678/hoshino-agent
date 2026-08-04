"""对外通道适配层

每个通道复用同一套 HoshinoAgent 内核，通过 session_id 前缀隔离：
- web: 默认 session_id（或用户指定）
- tg: tg_{chat_id}
"""

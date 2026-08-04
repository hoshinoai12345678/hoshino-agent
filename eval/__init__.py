"""星野爱 Agent 评估体系

包含角色一致性、记忆召回、工具调用、情绪响应四类评估指标，
以及测试用例集和可独立运行的评估脚本。

用法：
    python -m eval.run_eval
    或
    python eval/run_eval.py
"""
from eval.metrics import (
    persona_consistency_rate,
    memory_recall_accuracy,
    tool_call_success_rate,
    emotion_response_appropriateness,
)

__all__ = [
    "persona_consistency_rate",
    "memory_recall_accuracy",
    "tool_call_success_rate",
    "emotion_response_appropriateness",
]

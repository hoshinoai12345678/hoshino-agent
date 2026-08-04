"""统一日志配置

所有模块通过 get_logger(__name__) 获取 logger，确保日志格式统一、级别可控。
支持环境变量 LOG_LEVEL 控制输出级别（DEBUG/INFO/WARNING/ERROR）。
"""
import logging
import os
import sys

_LOG_CONFIGURED = False


def _configure_logging():
    """配置根 logger（只执行一次）"""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    _LOG_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取统一配置的 logger"""
    _configure_logging()
    return logging.getLogger(name)

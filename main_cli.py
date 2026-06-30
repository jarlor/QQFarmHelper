# -*- coding: utf-8 -*-
"""
main_cli.py
QQFarmHelper 终端入口。

运行：
    python main_cli.py
    python main_cli.py --cycles 1
    python main_cli.py --dry-run --max-steps 3
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Any, Dict

from state_machine import FarmStateMachine, load_runtime_config


LOGGER = logging.getLogger("qq_farm_helper.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QQ经典农场终端自动辅助工具")
    parser.add_argument(
        "--config",
        default="config_runtime.json",
        help="运行配置文件路径，默认 config_runtime.json",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="处理多少个好友循环后停止；默认使用配置文件，0 表示一直运行",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="最多执行多少次状态机 step；适合 dry-run 调试，默认不限制",
    )
    parser.add_argument(
        "--no-actionable-sleep",
        type=float,
        default=None,
        help="好友列表没有可操作好友时，关闭列表后等待多少秒再重试",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印点击坐标，不真实点击",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别，默认 INFO",
    )
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_runtime_config(args.config)

    if args.cycles is not None:
        config.setdefault("behavior", {})["stop_after_cycles"] = max(0, int(args.cycles))

    if args.max_steps is not None:
        config.setdefault("behavior", {})["max_steps"] = max(0, int(args.max_steps))

    if args.no_actionable_sleep is not None:
        config.setdefault("timing", {})["no_actionable_friends_sleep"] = max(
            0.0,
            float(args.no_actionable_sleep),
        )

    if args.dry_run:
        config.setdefault("clicker", {})["dry_run"] = True

    return config


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    should_stop = False

    def request_stop(_signum, _frame) -> None:
        nonlocal should_stop
        should_stop = True
        LOGGER.info("收到停止信号，当前步骤结束后退出")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    config = build_config(args)
    bot = FarmStateMachine(config=config)

    try:
        LOGGER.info("启动终端模式，dry_run=%s", bot.config["clicker"].get("dry_run", False))
        bot.run_loop(
            should_pause=lambda: False,
            should_stop=lambda: should_stop,
        )
        LOGGER.info("运行结束：%s", bot.stats)
        return 0
    except KeyboardInterrupt:
        LOGGER.info("用户中断")
        return 130
    finally:
        bot.close()


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
main.py
QQ经典农场辅助工具入口。

运行：
    python main.py

测试不点击：
    python main.py --dry-run

热键：
    F8   开始/暂停
    F9   停止
    Esc  停止

打包 exe 示例：
    pyinstaller -F -w main.py --name QQFarmHelper

如果你要把 templates 和 config 一起打包：
    pyinstaller -F -w main.py --name QQFarmHelper ^
      --add-data "templates;templates" ^
      --add-data "config_runtime.json;."
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

from state_machine import FarmStateMachine, load_runtime_config


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="QQ经典农场辅助工具")
    parser.add_argument(
        "--config",
        default="config_runtime.json",
        help="配置文件路径。默认 config_runtime.json；不存在时会尝试 config_v3.json。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只识别和打印，不真正点击鼠标。",
    )
    parser.add_argument(
        "--start-now",
        action="store_true",
        help="启动后立即开始，不等待 F8。",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    logger = logging.getLogger("qq_farm_helper.main")
    config = load_runtime_config(args.config)

    if args.dry_run:
        config.setdefault("clicker", {})["dry_run"] = True

    paused = {"value": not args.start_now}
    stopped = {"value": False}

    try:
        import keyboard

        def toggle_pause():
            paused["value"] = not paused["value"]
            logger.info("当前状态：%s", "暂停" if paused["value"] else "运行")

        def stop():
            stopped["value"] = True
            logger.info("收到停止热键")

        keyboard.add_hotkey("F8", toggle_pause)
        keyboard.add_hotkey("F9", stop)
        keyboard.add_hotkey("esc", stop)

        print("\nQQFarmHelper 已启动")
        print("F8：开始/暂停")
        print("F9 或 Esc：停止")
        print("日志：logs/farm_helper.log")
        if paused["value"]:
            print("当前为暂停状态，按 F8 开始。")
        else:
            print("已按 --start-now 立即开始。")

    except Exception as e:
        logger.warning("keyboard 热键初始化失败：%r", e)
        logger.warning("将直接开始运行。按 Ctrl+C 停止。")
        paused["value"] = False

    bot = FarmStateMachine(config=config)

    try:
        bot.run_loop(
            should_pause=lambda: paused["value"],
            should_stop=lambda: stopped["value"],
        )
    except KeyboardInterrupt:
        logger.info("Ctrl+C 停止")
    finally:
        logger.info("程序退出")


if __name__ == "__main__":
    main()

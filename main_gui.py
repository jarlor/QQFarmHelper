# -*- coding: utf-8 -*-
"""
main_gui.py
QQFarmHelper 2.0 图形界面入口。

功能：
    - 弹出一个可操作窗口
    - 开始 / 暂停 / 结束退出
    - 显示当前运行状态和统计信息
    - 不再依赖 F8/F9 热键
    - 使用后台线程运行 FarmStateMachine，避免 GUI 卡死

运行：
    python main_gui.py

打包：
    pyinstaller --clean --noconfirm --onefile --windowed `
      --name QQFarmHelper `
      --hidden-import cv2 `
      --collect-all cv2 `
      --add-data "templates;templates" `
      --add-data "config_runtime.json;." `
      main_gui.py
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from state_machine import FarmStateMachine, load_runtime_config


APP_NAME = "QQFarmHelper 2.0"


class QQFarmHelperGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("420x260")
        self.root.resizable(False, False)

        self.paused = True
        self.stopped = False
        self.worker: threading.Thread | None = None
        self.bot: FarmStateMachine | None = None
        self.last_error = ""
        self.started_once = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.root.after(500, self.refresh_ui)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text=APP_NAME, font=("Microsoft YaHei UI", 16, "bold"))
        title.pack(anchor="center", pady=(0, 12))

        self.status_var = tk.StringVar(value="状态：未启动")
        ttk.Label(container, textvariable=self.status_var, font=("Microsoft YaHei UI", 11)).pack(anchor="w", pady=(0, 8))

        self.stats_var = tk.StringVar(value="访问：0    摘取：0    跳过务农：0    无可摘：0")
        ttk.Label(container, textvariable=self.stats_var, font=("Microsoft YaHei UI", 10)).pack(anchor="w", pady=(0, 8))

        self.tip_var = tk.StringVar(value="点击“开始”后会自动寻找 QQ经典农场窗口。")
        ttk.Label(container, textvariable=self.tip_var, font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(0, 16))

        button_frame = ttk.Frame(container)
        button_frame.pack(fill=tk.X, pady=(4, 10))

        ttk.Button(button_frame, text="开始", command=self.on_start).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 8))
        ttk.Button(button_frame, text="暂停", command=self.on_pause).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 8))
        ttk.Button(button_frame, text="结束退出", command=self.on_exit).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Separator(container, orient="horizontal").pack(fill=tk.X, pady=10)

        self.small_var = tk.StringVar(value="建议：先打开 QQ经典农场窗口，再点击开始。")
        ttk.Label(container, textvariable=self.small_var, font=("Microsoft YaHei UI", 9)).pack(anchor="w")

    def setup_silent_logging(self) -> None:
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.CRITICAL)
        root_logger.addHandler(logging.NullHandler())

    def is_worker_alive(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def on_start(self) -> None:
        """
        第一次点击开始：启动后台线程。
        后续点击开始：从暂停状态恢复。
        """
        self.setup_silent_logging()

        if self.is_worker_alive():
            self.paused = False
            self.tip_var.set("已恢复运行。")
            return

        self.paused = False
        self.stopped = False
        self.last_error = ""
        self.started_once = True

        try:
            config = load_runtime_config("config_runtime.json")
            self.bot = FarmStateMachine(config=config)
        except Exception as e:
            self.last_error = repr(e)
            messagebox.showerror("启动失败", f"初始化失败：\n{e}")
            self.paused = True
            return

        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()
        self.tip_var.set("正在运行。需要暂停时点击“暂停”。")

    def on_pause(self) -> None:
        if not self.started_once:
            self.tip_var.set("当前还没有启动。")
            return

        self.paused = True
        self.tip_var.set("已暂停。点击“开始”可继续。")

    def on_exit(self) -> None:
        self.paused = False
        self.stopped = True
        self.status_var.set("状态：正在结束...")
        self.tip_var.set("正在退出。")

        try:
            if self.bot is not None:
                self.bot.close()
        except Exception:
            pass

        self.root.after(300, self.root.destroy)

    def _run_worker(self) -> None:
        try:
            assert self.bot is not None
            self.bot.run_loop(
                should_pause=lambda: self.paused,
                should_stop=lambda: self.stopped,
            )
        except Exception as e:
            self.last_error = repr(e)
        finally:
            self.paused = True

    def refresh_ui(self) -> None:
        if self.stopped:
            return

        alive = self.is_worker_alive()

        if not self.started_once:
            self.status_var.set("状态：未启动")
        elif self.last_error:
            self.status_var.set("状态：异常")
            self.tip_var.set(f"错误：{self.last_error}")
        elif alive and self.paused:
            self.status_var.set("状态：已暂停")
        elif alive and not self.paused:
            self.status_var.set("状态：运行中")
        elif self.started_once and not alive:
            self.status_var.set("状态：已结束")
            self.tip_var.set("后台任务已结束。点击“开始”可重新启动。")

        if self.bot is not None:
            s = self.bot.stats
            self.stats_var.set(
                f"访问：{s.visits}    摘取：{s.picks}    "
                f"跳过务农：{s.skips_farm}    无可摘：{s.no_pick}"
            )
            self.small_var.set(
                f"循环：{s.cycles}    未知页面：{s.unknown_pages}    异常：{s.errors}"
            )

        self.root.after(500, self.refresh_ui)


def main() -> None:
    root = tk.Tk()

    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    QQFarmHelperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
state_machine.py
QQ经典农场辅助工具的状态机。

它只负责“识别页面 → 点击对应按钮 → 等待跳转”。
真正的窗口寻找、截图、视觉识别、点击分别由：
    window_manager.py
    screen_capture.py
    vision.py
    clicker.py
提供。

页面逻辑：
    SELF_HOME:
        只检测并点击右下角“好友”。

    FRIEND_LIST:
        只检测右侧所有“拜访”按钮，选择最上方且最近未访问过的一行。

    FRIEND_HOME:
        只在好友农场页检测白名单动作按钮。
        默认点击“一键摘取 / 摘取手形 / 一键务农”，动作处理完成后回家。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, List
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from window_manager import GameWindow, find_game_window, bring_to_front, DEFAULT_WINDOW_KEYWORDS
from screen_capture import ScreenCapture
from vision import Vision, PageType, DEFAULT_ROIS, DEFAULT_THRESHOLDS, DEFAULT_CLICK_POINTS, resource_path
from clicker import Clicker


LOGGER = logging.getLogger("qq_farm_helper")


DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "template_dir": "templates",
    "window_keywords": DEFAULT_WINDOW_KEYWORDS,
    "thresholds": DEFAULT_THRESHOLDS,
    "rois": DEFAULT_ROIS,
    "click_points": DEFAULT_CLICK_POINTS,

    "timing": {
        "loop_interval": 0.25,
        "after_click": 0.65,
        "after_visit_click": 1.80,
        "after_pick_click": 1.20,
        "after_action_click": 1.20,
        "after_home_click": 1.20,
        "page_wait_timeout": 6.0,
        "unknown_sleep": 1.00,
        "no_button_sleep": 1.50,
        "no_unvisited_sleep": 30.0
    },

    "behavior": {
        "skip_recently_visited": True,
        "visited_cache_seconds": 180,
        "all_visible_visited_action": "revisit_top",
        "allow_fallback_click": True,
        "max_visible_visit_buttons": 8,
        "enabled_actions": ["pick_button", "pick_hand", "farm_button"],
        "max_actions_per_friend": 5,
        "max_steps": 0,
        "stop_after_cycles": 0
    },

    "clicker": {
        "dry_run": False,
        "random_offset": 2,
        "click_down_up_delay": 0.05,
        "bring_front_before_click": True
    },

    "debug": {
        "save_unknown_screenshot": True,
        "save_error_screenshot": True,
        "debug_dir": "logs/runtime_debug"
    }
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_runtime_config(path: str | Path = "config_runtime.json") -> Dict[str, Any]:
    """
    优先读取 config_runtime.json。
    如果不存在，再尝试读取 config_v3.json。
    都不存在则使用内置默认配置。
    支持 PyInstaller 打包后的路径（sys._MEIPASS）。
    """
    path = Path(path)
    if not path.is_absolute():
        path = resource_path(path)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return deep_merge(DEFAULT_RUNTIME_CONFIG, json.load(f))

    v3 = resource_path("config_v3.json")
    if v3.exists():
        with v3.open("r", encoding="utf-8") as f:
            return deep_merge(DEFAULT_RUNTIME_CONFIG, json.load(f))

    return dict(DEFAULT_RUNTIME_CONFIG)


@dataclass
class BotStats:
    cycles: int = 0
    visits: int = 0
    picks: int = 0
    farms: int = 0
    no_pick: int = 0
    unknown_pages: int = 0
    errors: int = 0


@dataclass
class FriendVisitRecord:
    last_seen: float
    hits: int = 1


class FarmStateMachine:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dry_run: Optional[bool] = None,
    ) -> None:
        self.config = deep_merge(DEFAULT_RUNTIME_CONFIG, config or {})

        if dry_run is not None:
            self.config["clicker"]["dry_run"] = bool(dry_run)

        self.vision = Vision(
            template_dir=self.config.get("template_dir", "templates"),
            thresholds=self.config.get("thresholds", {}),
            rois=self.config.get("rois", {}),
        )

        self.capture = ScreenCapture()

        click_cfg = self.config.get("clicker", {})
        self.clicker = Clicker(
            dry_run=bool(click_cfg.get("dry_run", False)),
            random_offset=int(click_cfg.get("random_offset", 2)),
            click_down_up_delay=float(click_cfg.get("click_down_up_delay", 0.05)),
            bring_front_before_click=bool(click_cfg.get("bring_front_before_click", True)),
        )

        self.window: Optional[GameWindow] = None
        self.stats = BotStats()
        self.visited_rows: Dict[int, FriendVisitRecord] = {}

        debug_dir = Path(self.config["debug"].get("debug_dir", "logs/runtime_debug"))
        debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = debug_dir

    # ---------- 基础工具 ----------

    def close(self) -> None:
        self.capture.close()

    def ensure_window(self) -> bool:
        if self.window is not None and self.window.is_valid:
            return True

        keywords = self.config.get("window_keywords", DEFAULT_WINDOW_KEYWORDS)
        self.window = find_game_window(keywords=keywords)

        if self.window:
            LOGGER.info("找到窗口：%s", self.window.title)
            bring_to_front(self.window.hwnd)
            return True

        LOGGER.warning("没有找到 QQ经典农场窗口")
        return False

    def grab(self) -> Optional[np.ndarray]:
        if not self.ensure_window() or self.window is None:
            return None

        try:
            return self.capture.capture_window(self.window, client_only=True, refresh=True)
        except Exception:
            LOGGER.exception("截图失败")
            self.stats.errors += 1
            return None

    def save_debug_frame(self, frame: np.ndarray, prefix: str) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.debug_dir / f"{prefix}_{ts}.png"
        cv2.imwrite(str(path), frame)
        return path

    def sleep(self, key: str) -> None:
        seconds = float(self.config.get("timing", {}).get(key, 0.5))
        time.sleep(max(0.0, seconds))

    # ---------- 好友去重 ----------

    def purge_visited_cache(self) -> None:
        ttl = float(self.config.get("behavior", {}).get("visited_cache_seconds", 900))
        now = time.time()

        expired = [h for h, rec in self.visited_rows.items() if now - rec.last_seen > ttl]
        for h in expired:
            del self.visited_rows[h]

    @staticmethod
    def dhash(image_bgr: np.ndarray) -> int:
        """
        计算好友行截图的感知哈希，避免重复拜访同一行。
        不识别昵称，不依赖 OCR。
        """
        if image_bgr.size == 0:
            return 0

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        diff = small[:, 1:] > small[:, :-1]

        value = 0
        for bit in diff.flatten():
            value = (value << 1) | int(bit)
        return int(value)

    @staticmethod
    def crop_friend_row(frame_bgr: np.ndarray, button_match) -> np.ndarray:
        """
        根据“拜访”按钮位置，裁取按钮左侧的整行好友信息。
        """
        if not getattr(button_match, "box", None):
            return frame_bgr[0:1, 0:1]

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = button_match.box

        row_top = max(0, y1 - int(0.025 * h))
        row_bottom = min(h, y2 + int(0.025 * h))
        row_left = int(0.03 * w)
        row_right = int(0.70 * w)

        return frame_bgr[row_top:row_bottom, row_left:row_right].copy()

    def choose_visit_button(self, frame_bgr: np.ndarray, buttons: List[Any]) -> Optional[Any]:
        """
        选择“拜访”按钮。

        v4 逻辑：
            1. 优先选择最上方且近期没访问过的好友。
            2. 如果当前可见 5~6 个好友都访问过，不再卡住不动。
               默认策略是 revisit_top：直接重新访问最上方好友。
            3. 如果你想恢复旧逻辑，可以在 config_runtime.json 里设置：
               "all_visible_visited_action": "wait"
        """
        if not buttons:
            return None

        behavior = self.config.get("behavior", {})
        if not behavior.get("skip_recently_visited", True):
            return buttons[0]

        self.purge_visited_cache()

        # 先找未访问过的可见好友。
        for button in buttons:
            row = self.crop_friend_row(frame_bgr, button)
            row_hash = self.dhash(row)

            if row_hash not in self.visited_rows:
                self.visited_rows[row_hash] = FriendVisitRecord(last_seen=time.time())
                return button

        # 当前可见好友都访问过时，不要停住。
        action = behavior.get("all_visible_visited_action", "revisit_top")

        if action == "wait":
            return None

        if action == "clear_cache":
            self.visited_rows.clear()
            button = buttons[0]
            row = self.crop_friend_row(frame_bgr, button)
            row_hash = self.dhash(row)
            self.visited_rows[row_hash] = FriendVisitRecord(last_seen=time.time())
            LOGGER.info("当前可见好友都访问过，已清空访问缓存并重新访问最上方好友")
            return button

        # 默认 revisit_top：保留缓存，但仍访问最上方好友。
        button = buttons[0]
        row = self.crop_friend_row(frame_bgr, button)
        row_hash = self.dhash(row)
        rec = self.visited_rows.get(row_hash)
        if rec:
            rec.last_seen = time.time()
            rec.hits += 1
        else:
            self.visited_rows[row_hash] = FriendVisitRecord(last_seen=time.time())

        LOGGER.info("当前可见好友都访问过，重新访问最上方好友，避免程序停住")
        return button

    # ---------- 页面处理 ----------

    def step(self) -> None:
        """
        执行一次状态机步骤。
        main.py 会循环调用它。
        """
        frame = self.grab()
        if frame is None:
            time.sleep(1.0)
            return

        detection = self.vision.detect_page_type(frame)
        page = detection.page_type

        LOGGER.info("page=%s score=%.3f", page, detection.score)

        if page == PageType.SELF_HOME:
            self.handle_self_home(frame)
        elif page == PageType.FRIEND_LIST:
            self.handle_friend_list(frame)
        elif page == PageType.FRIEND_HOME:
            self.handle_friend_home(frame)
        else:
            self.handle_unknown(frame, detection)

    def handle_self_home(self, frame: np.ndarray) -> None:
        """
        我的主页：点击右下角“好友”。
        """
        if self.window is None:
            return

        match = self.vision.detect_self_home_friend_menu(frame)

        if match.found:
            result = self.clicker.click_match(self.window, match)
            LOGGER.info("点击好友按钮：%s", result)
        else:
            # 兜底：只在页面已经被判定为 SELF_HOME 时使用相对坐标。
            if self.config.get("behavior", {}).get("allow_fallback_click", True):
                rx, ry = self.config["click_points"]["friend_menu"]
                result = self.clicker.click_relative(self.window, rx, ry)
                LOGGER.warning("好友按钮模板未命中，使用兜底坐标：%s", result)
            else:
                LOGGER.warning("好友按钮未识别，不点击")
                self.sleep("no_button_sleep")
                return

        self.sleep("after_click")

    def handle_friend_list(self, frame: np.ndarray) -> None:
        """
        好友列表：找所有“拜访”，点击最上面且最近没访问过的。
        """
        if self.window is None:
            return

        max_buttons = int(self.config.get("behavior", {}).get("max_visible_visit_buttons", 8))
        buttons = self.vision.find_visit_buttons(frame, max_results=max_buttons)

        if not buttons:
            LOGGER.warning("好友列表里没有识别到拜访按钮")
            self.sleep("no_button_sleep")
            return

        button = self.choose_visit_button(frame, buttons)
        if button is None:
            LOGGER.info("当前可见好友都在近期访问缓存里，暂停一会儿")
            time.sleep(float(self.config["timing"].get("no_unvisited_sleep", 30.0)))
            return

        result = self.clicker.click_match(self.window, button)
        LOGGER.info("点击拜访：%s score=%.3f center=%s", result, button.score, button.center)

        self.stats.visits += 1
        self.sleep("after_visit_click")

    def handle_friend_home(self, frame: np.ndarray) -> None:
        """
        好友家主页：
            1. 检测白名单动作按钮。
            2. 找到动作就点击，点击后重新截图继续找。
            3. 没有动作后回家。
        """
        if self.window is None:
            return

        actions_clicked = self.handle_friend_actions(frame)
        if actions_clicked <= 0:
            self.stats.no_pick += 1

        # 重新截图后再找回家，避免摘取后画面变化
        frame2 = self.grab()
        if frame2 is None:
            return

        home = self.vision.detect_friend_home_home_button(frame2)
        if home.found:
            result = self.clicker.click_match(self.window, home)
            LOGGER.info("点击回家：%s", result)
        else:
            if self.config.get("behavior", {}).get("allow_fallback_click", True):
                rx, ry = self.config["click_points"]["home"]
                result = self.clicker.click_relative(self.window, rx, ry)
                LOGGER.warning("回家按钮模板未命中，使用兜底坐标：%s", result)
            else:
                LOGGER.warning("回家按钮未识别，不点击")
                self.sleep("no_button_sleep")
                return

        self.stats.cycles += 1
        self.sleep("after_home_click")

    def handle_friend_actions(self, frame: np.ndarray) -> int:
        """
        在好友家页面处理所有允许的动作按钮。
        每点一次都重新截图，避免同一位置重复点击旧匹配结果。
        """
        if self.window is None:
            return 0

        behavior = self.config.get("behavior", {})
        enabled_actions = behavior.get("enabled_actions", ["pick_button", "pick_hand", "farm_button"])
        max_actions = int(behavior.get("max_actions_per_friend", 5))
        actions_clicked = 0
        current = frame
        clicked_keys: set[tuple[str, tuple[int, int] | None]] = set()

        for _ in range(max(1, max_actions)):
            actions = [
                action
                for action in self.vision.detect_friend_actions(current, enabled_actions=enabled_actions)
                if (action.action_name, action.match.center) not in clicked_keys
            ]
            if not actions:
                LOGGER.info("好友家没有识别到可点击动作")
                break

            action = actions[0]
            match = action.match
            clicked_keys.add((action.action_name, match.center))
            LOGGER.info(
                "点击好友动作：%s score=%.3f center=%s",
                action.action_name, match.score, match.center
            )
            result = self.clicker.click_match(self.window, match)
            LOGGER.info("动作点击结果：%s", result)

            if action.action_name in ("pick_button", "pick_hand"):
                self.stats.picks += 1
            elif action.action_name == "farm_button":
                self.stats.farms += 1

            actions_clicked += 1
            self.sleep("after_action_click")

            refreshed = self.grab()
            if refreshed is None:
                break

            detection = self.vision.detect_page_type(refreshed)
            if detection.page_type != PageType.FRIEND_HOME:
                LOGGER.info("动作点击后页面已离开好友家：%s", detection.page_type)
                break
            current = refreshed

        return actions_clicked

    def handle_unknown(self, frame: np.ndarray, detection) -> None:
        self.stats.unknown_pages += 1
        LOGGER.warning("未知页面 score=%.3f evidence=%s", detection.score, detection.evidence)

        if self.config.get("debug", {}).get("save_unknown_screenshot", True):
            path = self.save_debug_frame(frame, "unknown")
            LOGGER.warning("未知页面截图已保存：%s", path)

        self.sleep("unknown_sleep")

    # ---------- 循环 ----------

    def run_loop(self, should_pause, should_stop) -> None:
        """
        should_pause 和 should_stop 是函数，返回 bool。
        由 main.py 的热键控制。
        """
        LOGGER.info("状态机启动")
        steps = 0
        try:
            while not should_stop():
                if should_pause():
                    time.sleep(0.2)
                    continue

                try:
                    self.step()
                except Exception:
                    self.stats.errors += 1
                    LOGGER.exception("状态机 step 异常")
                    time.sleep(1.0)

                steps += 1

                max_steps = int(self.config.get("behavior", {}).get("max_steps", 0))
                if max_steps > 0 and steps >= max_steps:
                    LOGGER.info("达到 max_steps=%s，停止", max_steps)
                    break

                stop_after = int(self.config.get("behavior", {}).get("stop_after_cycles", 0))
                if stop_after > 0 and self.stats.cycles >= stop_after:
                    LOGGER.info("达到 stop_after_cycles=%s，停止", stop_after)
                    break

                time.sleep(float(self.config.get("timing", {}).get("loop_interval", 0.25)))
        finally:
            self.close()
            LOGGER.info("状态机结束：%s", self.stats)

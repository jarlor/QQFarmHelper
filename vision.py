# -*- coding: utf-8 -*-
"""
vision.py v3
模板匹配、ROI 裁剪、多帧确认、页面识别。

根据三张 820×1529 的 client_raw 图修正：
1. 我的主页：只识别右下角“好友”按钮。
2. 好友列表：只识别好友弹窗 tab 和右侧所有“拜访”按钮。
3. 好友家主页：只识别右下角“回家”和中下方“一键摘取/一键务农”区域。

核心原则：
    不要在所有页面检测所有按钮。
    先 detect_page_type()，再按页面类型检测对应按钮。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import sys
import time

import cv2
import numpy as np


RatioROI = Sequence[float]


class PageType:
    UNKNOWN = "UNKNOWN"
    SELF_HOME = "SELF_HOME"
    FRIEND_LIST = "FRIEND_LIST"
    FRIEND_HOME = "FRIEND_HOME"


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "friend_menu": 0.72,
    "friend_tab": 0.78,
    "visit_button": 0.80,
    "home_button": 0.76,
    "pick_button": 0.80,
    "pick_hand": 0.76,
    "farm_button": 0.72,
}


DEFAULT_ROIS: Dict[str, RatioROI] = {
    # 下面所有 ROI 都基于 capture_window(client_only=True) 得到的完整客户区截图。
    # 你当前 raw 图尺寸是 820×1529，顶部“QQ经典农场”白色标题栏也在截图里。

    # 1. 我的主页：右下角“好友”底部菜单按钮
    # 约 x=625~815, y=1340~1515
    "self_friend_menu": [0.74, 0.86, 1.00, 1.00],

    # 兼容旧调用名
    "friend_menu": [0.74, 0.86, 1.00, 1.00],

    # 2. 好友列表页：弹窗顶部区域，用于确认这是好友列表弹窗
    # 约 y=140~335，包含标题“好友”和 tabs
    "friend_popup_header": [0.00, 0.09, 1.00, 0.25],

    # 选中的“好友”tab 本体，约 x=20~220, y=240~335
    "friend_tab": [0.00, 0.14, 0.32, 0.24],

    # 好友列表右侧所有绿色“拜访”按钮
    # 约 x=630~770, y=485~1325
    "visit_buttons": [0.72, 0.30, 0.98, 0.89],

    # 3. 好友家主页：右下角“回家”按钮
    # 约 x=680~810, y=1090~1200
    "friend_home_home_button": [0.78, 0.67, 1.00, 0.81],

    # 兼容旧调用名
    "home_button": [0.78, 0.67, 1.00, 0.81],

    # 好友家主页：一键务农 / 一键摘取按钮区域
    # 你给的图里“一键务农”中心约 (410,1140)，但当两个按钮同时出现会轻微偏移。
    # 所以这里给宽一些，只在 FRIEND_HOME 页面使用，避免误扫自己的主页。
    "friend_action_buttons": [0.22, 0.64, 0.75, 0.82],

    # 更紧的单按钮区域，调试用
    "friend_action_center": [0.34, 0.69, 0.63, 0.79],

    # 兼容旧调用名
    "bottom_action": [0.22, 0.64, 0.75, 0.82],
}


# 推荐点击点，基于完整客户区相对坐标。
# 自动点击时优先点击模板匹配中心；只有兜底时才用这些点。
DEFAULT_CLICK_POINTS: Dict[str, Tuple[float, float]] = {
    "friend_menu": (0.905, 0.940),
    "first_visit": (0.855, 0.342),
    "home": (0.910, 0.748),
    "action_center": (0.500, 0.745),
}


@dataclass
class MatchResult:
    template_name: str
    found: bool
    score: float
    box: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[int, int]] = None
    scale: float = 1.0
    roi: Optional[Tuple[int, int, int, int]] = None

    @property
    def x(self) -> Optional[int]:
        return None if self.center is None else self.center[0]

    @property
    def y(self) -> Optional[int]:
        return None if self.center is None else self.center[1]


@dataclass
class PageDetection:
    page_type: str
    score: float
    evidence: Optional[MatchResult] = None


def resource_path(relative_path: str | Path) -> Path:
    relative_path = Path(relative_path)
    if hasattr(sys, "_MEIPASS"):
        candidate = Path(sys._MEIPASS) / relative_path
        if candidate.exists():
            return candidate
    return Path.cwd() / relative_path


def ratio_roi_to_box(image_shape: Tuple[int, int], roi: RatioROI) -> Tuple[int, int, int, int]:
    if len(roi) != 4:
        raise ValueError("roi 必须是 [x1, y1, x2, y2]")

    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(float, roi)

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"非法 roi: {roi}")

    px1 = max(0, min(w - 1, int(round(w * x1))))
    py1 = max(0, min(h - 1, int(round(h * y1))))
    px2 = max(px1 + 1, min(w, int(round(w * x2))))
    py2 = max(py1 + 1, min(h, int(round(h * y2))))

    return px1, py1, px2, py2


def crop_roi(image_bgr: np.ndarray, roi: Optional[RatioROI]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = image_bgr.shape[:2]
    if roi is None:
        return image_bgr, (0, 0, w, h)

    x1, y1, x2, y2 = ratio_roi_to_box((h, w), roi)
    return image_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class TemplateStore:
    def __init__(self, template_dir: str | Path = "templates") -> None:
        self.template_dir = Path(template_dir)
        if not self.template_dir.is_absolute():
            self.template_dir = resource_path(self.template_dir)
        self.templates: Dict[str, np.ndarray] = {}

    def load(self, name: str, filename: Optional[str] = None) -> np.ndarray:
        filename = filename or f"{name}.png"
        path = self.template_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"找不到模板文件: {path}")

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"模板读取失败: {path}")

        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        self.templates[name] = img
        return img

    def load_all(self, names: Iterable[str]) -> None:
        for name in names:
            self.load(name)

    def get(self, name: str) -> np.ndarray:
        if name not in self.templates:
            return self.load(name)
        return self.templates[name]


class Vision:
    def __init__(
        self,
        template_dir: str | Path = "templates",
        thresholds: Optional[Dict[str, float]] = None,
        rois: Optional[Dict[str, RatioROI]] = None,
        use_gray: bool = True,
    ) -> None:
        self.store = TemplateStore(template_dir)

        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

        self.rois = dict(DEFAULT_ROIS)
        if rois:
            self.rois.update(rois)

        self.use_gray = use_gray

    def load_default_templates(self) -> None:
        self.store.load_all(DEFAULT_THRESHOLDS.keys())

    def _prepare(self, image_bgr: np.ndarray, template_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_gray:
            return to_gray(image_bgr), to_gray(template_bgr)
        return image_bgr, template_bgr

    def match_template(
        self,
        frame_bgr: np.ndarray,
        template_name: str,
        roi: Optional[RatioROI] = None,
        threshold: Optional[float] = None,
        scales: Sequence[float] = (1.0,),
    ) -> MatchResult:
        threshold = float(threshold if threshold is not None else self.thresholds.get(template_name, 0.80))
        roi_img, roi_box = crop_roi(frame_bgr, roi)
        roi_x1, roi_y1, _, _ = roi_box

        template = self.store.get(template_name)

        best = MatchResult(
            template_name=template_name,
            found=False,
            score=-1.0,
            box=None,
            center=None,
            roi=roi_box,
        )

        source = to_gray(roi_img) if self.use_gray else roi_img
        source_h, source_w = source.shape[:2]

        for scale in scales:
            if scale <= 0:
                continue

            tpl = template
            if abs(scale - 1.0) > 1e-6:
                new_w = max(1, int(round(template.shape[1] * scale)))
                new_h = max(1, int(round(template.shape[0] * scale)))
                tpl = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)

            search_img, tpl_img = self._prepare(roi_img, tpl)
            th, tw = tpl_img.shape[:2]

            if th > source_h or tw > source_w:
                continue

            result = cv2.matchTemplate(search_img, tpl_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if float(max_val) > best.score:
                x, y = max_loc
                abs_x1 = roi_x1 + int(x)
                abs_y1 = roi_y1 + int(y)
                abs_x2 = abs_x1 + int(tw)
                abs_y2 = abs_y1 + int(th)
                best = MatchResult(
                    template_name=template_name,
                    found=float(max_val) >= threshold,
                    score=float(max_val),
                    box=(abs_x1, abs_y1, abs_x2, abs_y2),
                    center=((abs_x1 + abs_x2) // 2, (abs_y1 + abs_y2) // 2),
                    scale=float(scale),
                    roi=roi_box,
                )

        if best.score < 0:
            best.score = 0.0

        best.found = best.score >= threshold
        return best

    def find_all_templates(
        self,
        frame_bgr: np.ndarray,
        template_name: str,
        roi: Optional[RatioROI] = None,
        threshold: Optional[float] = None,
        max_results: int = 10,
        nms_iou: float = 0.30,
        scales: Sequence[float] = (1.0,),
    ) -> List[MatchResult]:
        threshold = float(threshold if threshold is not None else self.thresholds.get(template_name, 0.80))
        roi_img, roi_box = crop_roi(frame_bgr, roi)
        roi_x1, roi_y1, _, _ = roi_box

        template = self.store.get(template_name)
        candidates: List[MatchResult] = []

        for scale in scales:
            tpl = template
            if abs(scale - 1.0) > 1e-6:
                new_w = max(1, int(round(template.shape[1] * scale)))
                new_h = max(1, int(round(template.shape[0] * scale)))
                tpl = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)

            search_img, tpl_img = self._prepare(roi_img, tpl)
            th, tw = tpl_img.shape[:2]
            sh, sw = search_img.shape[:2]

            if th > sh or tw > sw:
                continue

            result = cv2.matchTemplate(search_img, tpl_img, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= threshold)

            for x, y in zip(xs, ys):
                score = float(result[y, x])
                abs_x1 = roi_x1 + int(x)
                abs_y1 = roi_y1 + int(y)
                abs_x2 = abs_x1 + int(tw)
                abs_y2 = abs_y1 + int(th)
                candidates.append(
                    MatchResult(
                        template_name=template_name,
                        found=True,
                        score=score,
                        box=(abs_x1, abs_y1, abs_x2, abs_y2),
                        center=((abs_x1 + abs_x2) // 2, (abs_y1 + abs_y2) // 2),
                        scale=float(scale),
                        roi=roi_box,
                    )
                )

        candidates.sort(key=lambda m: m.score, reverse=True)
        return non_max_suppression(candidates, iou_threshold=nms_iou)[:max_results]

    # ---------- 页面识别 ----------

    def detect_page_type(self, frame_bgr: np.ndarray) -> PageDetection:
        """
        先判断当前页面，再决定后续检测什么。
        顺序：
            1. 好友列表：绿色拜访按钮/好友 tab
            2. 好友家主页：右下角回家按钮
            3. 我的主页：右下角好友按钮
        """
        best_evidence: Optional[MatchResult] = None

        # 好友列表：多个绿色“拜访”最有辨识度
        try:
            visits = self.find_visit_buttons(frame_bgr, max_results=3)
            if visits:
                return PageDetection(PageType.FRIEND_LIST, visits[0].score, visits[0])
        except FileNotFoundError:
            pass

        # 好友列表备用：好友 tab
        try:
            tab = self.match_template(
                frame_bgr,
                "friend_tab",
                roi=self.rois["friend_tab"],
                scales=(0.90, 0.95, 1.0, 1.05, 1.10),
            )
            best_evidence = tab
            if tab.found:
                return PageDetection(PageType.FRIEND_LIST, tab.score, tab)
        except FileNotFoundError:
            pass

        # 好友家主页：右下角是“回家”
        try:
            home = self.match_template(
                frame_bgr,
                "home_button",
                roi=self.rois["friend_home_home_button"],
                scales=(0.90, 0.95, 1.0, 1.05, 1.10),
            )
            if best_evidence is None or home.score > best_evidence.score:
                best_evidence = home
            if home.found:
                return PageDetection(PageType.FRIEND_HOME, home.score, home)
        except FileNotFoundError:
            pass

        # 我的主页：右下角是“好友”
        try:
            friend = self.match_template(
                frame_bgr,
                "friend_menu",
                roi=self.rois["self_friend_menu"],
                scales=(0.90, 0.95, 1.0, 1.05, 1.10),
            )
            if best_evidence is None or friend.score > best_evidence.score:
                best_evidence = friend
            if friend.found:
                return PageDetection(PageType.SELF_HOME, friend.score, friend)
        except FileNotFoundError:
            pass

        return PageDetection(PageType.UNKNOWN, best_evidence.score if best_evidence else 0.0, best_evidence)

    # ---------- 页面内检测 ----------

    def detect_self_home_friend_menu(self, frame_bgr: np.ndarray) -> MatchResult:
        return self.match_template(
            frame_bgr,
            "friend_menu",
            roi=self.rois["self_friend_menu"],
            scales=(0.90, 0.95, 1.0, 1.05, 1.10),
        )

    def find_visit_buttons(
        self,
        frame_bgr: np.ndarray,
        threshold: Optional[float] = None,
        max_results: int = 8,
    ) -> List[MatchResult]:
        buttons = self.find_all_templates(
            frame_bgr,
            "visit_button",
            roi=self.rois["visit_buttons"],
            threshold=threshold,
            max_results=max_results,
            nms_iou=0.30,
            scales=(0.90, 0.95, 1.0, 1.05, 1.10),
        )
        buttons.sort(key=lambda m: (m.center[1] if m.center else 999999, -m.score))
        return buttons

    def detect_friend_home_home_button(self, frame_bgr: np.ndarray) -> MatchResult:
        return self.match_template(
            frame_bgr,
            "home_button",
            roi=self.rois["friend_home_home_button"],
            scales=(0.90, 0.95, 1.0, 1.05, 1.10),
        )

    def detect_pick_or_farm(self, frame_bgr: np.ndarray) -> Tuple[str, MatchResult]:
        """
        只应该在 FRIEND_HOME 页面调用。

        v4 逻辑：
            - 同时检测一键摘取、一键务农、摘取手势。
            - 如果“摘取”和“务农”同时出现，优先返回 PICK，只点击摘取。
            - 如果只有“一键务农”，返回 SKIP，不点击。
            - 这样不会因为 farm_button 存在而错过一键摘取。

        返回：
            ("PICK", result)     识别到一键摘取或摘取手势
            ("SKIP", result)     只识别到一键务农，绝不点击
            ("NO_PICK", result)  没识别到
        """
        roi = self.rois["friend_action_buttons"]

        farm = self.match_template(
            frame_bgr,
            "farm_button",
            roi=roi,
            threshold=self.thresholds.get("farm_button", 0.72),
            scales=(0.90, 0.95, 1.0, 1.05, 1.10),
        )

        pick = self.match_template(
            frame_bgr,
            "pick_button",
            roi=roi,
            threshold=self.thresholds.get("pick_button", 0.80),
            scales=(0.90, 0.95, 1.0, 1.05, 1.10),
        )

        hand = self.match_template(
            frame_bgr,
            "pick_hand",
            roi=roi,
            threshold=self.thresholds.get("pick_hand", 0.76),
            scales=(0.90, 0.95, 1.0, 1.05, 1.10),
        )

        # 摘取优先：如果“一键摘取”和“一键务农”同时出现，只点摘取。
        pick_candidates = [r for r in (pick, hand) if r.found]
        if pick_candidates:
            best_pick = max(pick_candidates, key=lambda r: r.score)
            return "PICK", best_pick

        # 只有务农时才跳过。
        if farm.found:
            return "SKIP", farm

        best = max([farm, pick, hand], key=lambda r: r.score)
        return "NO_PICK", best

    # ---------- 多帧确认 ----------

    def confirm_template(
        self,
        capture_fn: Callable[[], np.ndarray],
        template_name: str,
        roi: Optional[RatioROI] = None,
        threshold: Optional[float] = None,
        required_hits: int = 2,
        max_frames: int = 3,
        interval: float = 0.15,
        scales: Sequence[float] = (1.0,),
    ) -> MatchResult:
        hits: List[MatchResult] = []
        best: Optional[MatchResult] = None

        for _ in range(max_frames):
            frame = capture_fn()
            result = self.match_template(frame, template_name, roi=roi, threshold=threshold, scales=scales)
            if best is None or result.score > best.score:
                best = result
            if result.found:
                hits.append(result)
            if len(hits) >= required_hits:
                hit = hits[-1]
                hit.found = True
                return hit
            time.sleep(interval)

        if best is None:
            return MatchResult(template_name=template_name, found=False, score=0.0)
        best.found = False
        return best

    def wait_until_page(
        self,
        capture_fn: Callable[[], np.ndarray],
        target_page: str,
        timeout: float = 6.0,
        interval: float = 0.25,
    ) -> PageDetection:
        start = time.time()
        best = PageDetection(PageType.UNKNOWN, 0.0, None)

        while time.time() - start <= timeout:
            frame = capture_fn()
            det = self.detect_page_type(frame)
            if det.score > best.score:
                best = det
            if det.page_type == target_page:
                return det
            time.sleep(interval)

        return best

    def save_roi_debug(self, frame_bgr: np.ndarray, out_dir: str | Path = "logs/roi_debug") -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, roi in self.rois.items():
            crop, _ = crop_roi(frame_bgr, roi)
            cv2.imwrite(str(out_dir / f"{name}.png"), crop)


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return 0.0 if union <= 0 else inter / union


def non_max_suppression(matches: List[MatchResult], iou_threshold: float = 0.30) -> List[MatchResult]:
    kept: List[MatchResult] = []

    for m in sorted(matches, key=lambda x: x.score, reverse=True):
        if m.box is None:
            continue

        duplicated = False
        for k in kept:
            if k.box is not None and box_iou(m.box, k.box) >= iou_threshold:
                duplicated = True
                break

        if not duplicated:
            kept.append(m)

    return kept


def draw_match(
    frame_bgr: np.ndarray,
    match: MatchResult,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = frame_bgr.copy()
    if match.box:
        x1, y1, x2, y2 = match.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out,
            f"{match.template_name} {match.score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def draw_roi(
    frame_bgr: np.ndarray,
    roi: RatioROI,
    label: str,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = frame_bgr.copy()
    x1, y1, x2, y2 = ratio_roi_to_box(frame_bgr.shape[:2], roi)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        out,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python vision.py <screenshot.png>")
        sys.exit(0)

    frame_path = Path(sys.argv[1])
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise SystemExit(f"截图读取失败: {frame_path}")

    vision = Vision(template_dir="templates")
    det = vision.detect_page_type(frame)
    print(f"page_type={det.page_type}, score={det.score:.3f}, evidence={det.evidence}")

    if det.page_type == PageType.SELF_HOME:
        r = vision.detect_self_home_friend_menu(frame)
        print("friend_menu:", r)

    elif det.page_type == PageType.FRIEND_LIST:
        buttons = vision.find_visit_buttons(frame)
        print("visit buttons:", len(buttons))
        for i, b in enumerate(buttons, 1):
            print(i, b)

    elif det.page_type == PageType.FRIEND_HOME:
        home = vision.detect_friend_home_home_button(frame)
        state, action = vision.detect_pick_or_farm(frame)
        print("home:", home)
        print("action:", state, action)

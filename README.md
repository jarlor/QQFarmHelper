# QQFarmHelper

基于 OpenCV 模板匹配的桌面自动化实验项目，用于识别 QQ 经典农场窗口、好友列表、拜访按钮、一键摘取/一键务农按钮，并执行本地自动化操作。

当前支持 Windows 和 macOS。Windows 使用 pywin32 操作窗口和鼠标；macOS 使用 PyObjC/Quartz 操作窗口和鼠标。

## 功能

- 自动寻找 QQ 经典农场窗口
- 截取窗口客户区画面
- 使用 OpenCV 模板匹配识别按钮
- 自动打开好友列表
- 自动拜访好友
- 识别好友家的一键摘取、摘取手形、一键务农等动作按钮
- 只点击白名单里的动作按钮，默认会点击一键摘取和一键务农
- 当前可见好友都访问过时，默认重新访问最上方好友
   config_runtime.json:
   behavior.all_visible_visited_action = "revisit_top"
   可选值：
   - "revisit_top" 默认，全部访问过时继续访问最上方
   - "clear_cache" 全部访问过时清空缓存后访问最上方
   - "wait" 恢复旧逻辑，全部访问过就等待

   behavior.enabled_actions 控制好友家允许点击的动作：
   - "pick_button" 一键摘取
   - "pick_hand" 摘取手形
   - "farm_button" 一键务农
## 使用方法

1. 电脑端打开 QQ 经典农场，尽量让游戏内容区域铺满窗口，避免因为分辨率不一致导致匹配失败。
2. 如果匹配失败，可以自行裁取对应按钮图片，替换 `templates/` 目录下的模板：

| 模板文件 | 说明 |
|---------|------|
| `friend_menu.png` | 右下角”好友”按钮 |
| `friend_tab.png` | 好友弹窗里的”好友”标签 |
| `visit_button.png` | 第一位好友右侧”拜访”按钮 |
| `home_button.png` | 右下角”回家”按钮 |
| `pick_button.png` | 一键摘取按钮 |
| `pick_hand.png` | 摘取手形图标 |
| `farm_button.png` | 一键务农按钮 |

## 环境

- Python 3.10+
- OpenCV
- mss
- numpy
- Windows: pywin32
- macOS: PyObjC Quartz/Cocoa

## 安装

```bash
pip install -r requirements.txt
```

## 运行

终端模式：

```bash
python main_cli.py
```

只处理一个好友循环后停止：

```bash
python main_cli.py --cycles 1
```

只打印日志和点击坐标，不真实点击：

```bash
python main_cli.py --dry-run --max-steps 3
```

GUI 模式仍然保留：

```bash
python main_gui.py
```

macOS 首次运行前，需要在系统设置里给运行 Python 的程序授权：

1. 系统设置 -> 隐私与安全性 -> 屏幕录制：勾选 Terminal、iTerm、PyCharm 或你实际启动脚本的 App。
2. 系统设置 -> 隐私与安全性 -> 辅助功能：勾选同一个 App，用于发送鼠标点击。
3. 授权后重启对应 App，再运行程序。

如果程序找不到窗口，可以先打印当前可见窗口，确认窗口标题是否包含配置里的关键词：

```bash
python window_manager.py
```

如果 macOS 上标题不匹配，修改 `config_runtime.json` 里的 `window_keywords`，加入实际显示的 App 或窗口标题。

## 免责声明

本项目仅用于学习 Windows 桌面自动化、OpenCV 模板匹配和 Python 项目打包。使用者应自行遵守相关软件、游戏或平台的服务条款。

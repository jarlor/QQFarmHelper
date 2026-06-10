# QQFarmHelper

基于 OpenCV 模板匹配的 Windows 桌面自动化实验项目，用于识别 QQ 经典农场窗口、好友列表、拜访按钮、一键摘取/一键务农按钮，并执行本地自动化操作。

## 功能

- 自动寻找 QQ 经典农场窗口
- 截取窗口客户区画面
- 使用 OpenCV 模板匹配识别按钮
- 自动打开好友列表
- 自动拜访好友
- 识别一键摘取和一键务农
- 优先摘取，不主动务农
- 当前可见好友都访问过时，默认重新访问最上方好友
   config_runtime.json:
   behavior.all_visible_visited_action = "revisit_top"
   可选值：
   - "revisit_top" 默认，全部访问过时继续访问最上方
   - "clear_cache" 全部访问过时清空缓存后访问最上方
   - "wait" 恢复旧逻辑，全部访问过就等待
## 使用方法

1. 电脑端打开 QQ 经典农场，点击右上角**最大化**，避免因为分辨率不一致导致匹配失败。
2. 如果匹配失败，可以自行裁取对应按钮图片，替换 `templates/` 目录下的模板：

| 模板文件 | 说明 |
|---------|------|
| `friend_menu.png` | 右下角”好友”按钮 |
| `friend_tab.png` | 好友弹窗里的”好友”标签 |
| `visit_button.png` | 第一位好友右侧”拜访”按钮 |
| `home_button.png` | 右下角”回家”按钮 |
| `pick_button.png` | 一键摘取按钮 |
| `pick_hand.png` | 摘取手形图标 |
| `farm_button.png` | 一键务农按钮（负样本，禁止点击） |

## 环境

- Windows 11
- Python 3.10+
- OpenCV
- pywin32
- mss
- keyboard
- numpy

## 安装

```bash
pip install -r requirements.txt
```

## 免责声明

本项目仅用于学习 Windows 桌面自动化、OpenCV 模板匹配和 Python 项目打包。使用者应自行遵守相关软件、游戏或平台的服务条款。
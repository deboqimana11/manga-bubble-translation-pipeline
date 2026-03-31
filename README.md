# Manga Bubble Translation Pipeline

本项目用于将日文漫画气泡自动翻译为简体中文，支持命令行、桌面 UI，以及可选的本地常驻服务模式。

## 功能概览

- 使用 `huyvux3005/manga109-segmentation-bubble` 做气泡分割
- 使用 `manga-ocr` 做日文 OCR
- 默认使用 `facebook/nllb-200-distilled-600M` 做日译中
- 支持切换到 OpenAI 兼容接口翻译
- 默认使用 `LaMa` 去字，也支持 `opencv` 作为替代后端
- 集成 `vendor/comic-text-detector` 改进文字遮罩和纵排/横排判断
- 提供缓存机制，重复处理未变化页面时会更快

## 仓库结构

- `translate_manga.py`: 主处理脚本
- `ui_translate_manga.py`: 桌面图形界面
- `manga_service.py`: 常驻服务
- `translation_rules.json`: 术语和翻译修正规则
- `vendor/comic-text-detector`: 本地内置文字检测依赖
- `requirements.txt`: 运行依赖
- `.env.example`: OpenAI 兼容接口环境变量示例
- `ai_profile.example.json`: UI 使用的 AI 配置示例

## 在新电脑上部署

建议在 Windows PowerShell 下执行。

### 1. 安装 Python

建议使用 Python `3.13`。

### 2. 克隆仓库

```powershell
git clone <your-github-repo-url>
cd <repo-folder>
```

### 3. 创建虚拟环境并安装依赖

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你要使用 NVIDIA GPU，可以按你目标机器的 CUDA 环境额外安装匹配版本的 `torch` / `torchvision`。

## 启动方式

### 图形界面

```powershell
.\启动翻译界面.bat
```

或：

```powershell
.\.venv\Scripts\python.exe .\ui_translate_manga.py
```

### 命令行

```powershell
.\.venv\Scripts\python.exe .\translate_manga.py --input . --output .\outputs
```

首次运行会下载 Hugging Face / OCR / LaMa 所需模型，因此启动较慢属于正常现象。

### 服务模式

启动服务：

```powershell
.\run_service.ps1
```

调用服务处理：

```powershell
.\process_via_service.ps1
```

自动启动服务并处理：

```powershell
.\start_and_process.ps1
```

## OpenAI 兼容接口配置

你可以使用以下两种方式之一：

### 方式 1：环境变量

先参考 `.env.example` 设置：

```powershell
$env:OPENAI_COMPAT_ENDPOINT="https://your-openai-compatible-endpoint/"
$env:OPENAI_COMPAT_API_KEY="your_api_key_here"
$env:OPENAI_COMPAT_MODEL="your-model-name"
```

然后启动 UI 或命令行。UI 在没有 `ai_profile.json` 时会自动读取这些环境变量。

### 方式 2：UI 内保存配置

在 UI 中切换到 “AI 翻译”，填写 `Endpoint`、`API Key`、`Model` 后点击“保存 AI 参数”。

注意：

- `ai_profile.json` 和 `ui_settings.json` 已被 `.gitignore` 忽略，不应提交到 GitHub
- 不要把真实 API Key 提交到仓库

## 输出目录

程序会在你指定的输出目录下生成：

- `translated`: 成品图
- `debug`: 调试图
- `json`: OCR 和翻译记录

## 翻译规则

自定义翻译修正位于 `translation_rules.json`：

- `term_glossary`: 术语替换
- `exact_overrides`: 整句覆盖
- `pattern_rules`: 常见误译修正

修改后建议加 `--force` 重新生成输出。

## GitHub 发布建议

建议提交这些文件：

- 源码脚本
- `vendor/`
- `translation_rules.json`
- `requirements.txt`
- `README.md`
- 启动脚本

不要提交这些文件：

- `.venv/`
- `__pycache__/`
- `outputs*/`
- `ui_settings.json`
- `ai_profile.json`
- `service.log`
- `service.err.log`

## 注意事项

- 某些依赖体积较大，首次安装和首次模型下载会比较慢
- 如果系统没有中文字体，程序会回退到 Pillow 默认字体，显示效果可能较差，建议手动指定中文字体
- 本项目依赖的第三方模型和库可能有各自的许可证要求，发布前建议自行确认用途是否符合许可

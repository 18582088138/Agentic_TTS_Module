# 05 · 部署 / Deployment

三种场景：**本机 CPU 开发**、**4060 8 GB CUDA 本地部署**、**服务化**。

---

## 1 · 本机 CPU（开发机，已实测）

环境 `conda ov_env_py312` 里依赖已齐，**不需要 pip install 任何东西**：

```
torch 2.8.0+cpu  torchaudio 2.8.0+cpu  transformers 4.57.3  accelerate 1.12.0
numpy 1.26.4（被 funasr 锁在 1.x）  soundfile 0.14.0  librosa 0.11.0
fastapi 0.139.2  uvicorn 0.51.0  nicegui 3.14.0  typer 0.25.1  pytest 9.1.1
```

```bash
conda activate ov_env_py312
cd Agent_TTS_Module
python -m agentic_tts.cli doctor
python -m agentic_tts.cli speak "今天的人工智能资讯。" --voice Serena
```

**实测（本机 CPU，1.7B CustomVoice，float32）**：

| 项 | 值 |
|---|---|
| 权重加载 | 4.3 s |
| 10 字短句 | 音频 1.84 s，耗时约 24 s，**RTF 12.93** |

RTF ≈ 13 意味着**CPU 上 1 分钟音频要跑 13 分钟** —— 能用来验证功能，
不适合批量出片。批量请上 GPU。

⚠️ 两条环境注意：

1. `sox` 命令不在 PATH 上（上游会打一行 `'sox' is not recognized`），
   **不影响合成** —— Python 侧的 `sox` 包只在个别路径用到。
2. `flash-attn` 没装，上游会提示走 manual PyTorch 实现。CPU 上本来也用不上。

---

## 2 · 4060 8 GB CUDA（本地部署目标）

### 2.1 装 CUDA 版 torch（**必须先手工装，再装本项目**）

```bash
conda activate ov_env_py312     # ⚠️ 这个环境是多项目共用的，动它之前先看下面的警告
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

⚠️ **共用环境的风险**：`ov_env_py312` 同时被 DailyNewsAssistant / verify_qwen3_tts /
若干 CV 项目使用，把 `torch` 从 `+cpu` 换成 CUDA 版会影响它们（虽然通常是好事）。
若不想动，新建一个专用环境：

```bash
conda create -n tts_cu124 python=3.12 -y && conda activate tts_cu124
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[server,gui]"
```

### 2.2 配置

```yaml
engine:
  device: auto          # 有卡就 cuda:0
  dtype: auto           # cuda → bfloat16（float32 需要约 9.5 GB，8 GB 卡直接 OOM）
  max_resident: 1       # ⭐ 必须保持 1
  attn_impl: eager      # 想更省显存可装 flash-attn 后改 flash_attention_2
```

### 2.3 为什么 `max_resident` 必须是 1（实测数据）

| 项 | bf16 | float32 |
|---|---|---|
| 一个 1.7B 的权重（含声码器） | 3.9 GB | 7.8 GB |
| 跑起来（+KV cache/激活/分配器碎片） | 约 5.0 GB | 约 9.5 GB |

8 GB **装得下一个，装不下两个**。所以：

- 换模式时**先卸再载**（顺序反了会让两份同时在显存里，正是要避免的那件事）；
- 卸载后必须显式 `torch.cuda.empty_cache()` —— 仅 `del` 不会把显存还给驱动，
  下一次加载照样 OOM，而报错只说「显存不足」，完全看不出是上一个模型赖着没走；
- 加载前预检显存，不够就直接报错并给出该改哪一项（不必白等几十秒才 OOM）；
- 多段混用不同模式时，门面**按 checkpoint 分组执行**，轮动次数 = 模式数（≤3），
  而不是段数。

想彻底避免轮动：把工作流拆成两遍跑（先把所有 design 段跑完，再跑 clone 段），
而不是提高 `max_resident`。

### 2.4 复测命令（在 4060 机器上）

```bash
python -m agentic_tts.cli doctor                       # 应显示 cuda:0 / bfloat16 / 显存够
python -m pytest tests -q                              # 离线 108 条
python -m pytest tests/test_qwen3_real.py -q -m real -s   # 三种模式各一条，含轮动
python -m agentic_tts.cli batch samples/segments.yaml -n gpu_check
```

⚠️ **同一份权重在 CPU 与 GPU 上输出不同**（实测：贪心解码下第 38 步就分歧，
波形余弦 0.018；而同设备两次完全一致）。换设备等于换音色细节，**要用耳朵复核**，
不能假设 CPU 上听着好的在 GPU 上也一样。

---

## 3 · 服务化

```bash
python -m agentic_tts.server --host 0.0.0.0 --port 8300      # HTTP
python -m agentic_tts.gui --host 0.0.0.0 --port 8301         # GUI（同进程直调，不经 HTTP）
```

部署时用环境变量覆盖配置，**不必改文件**：

```bash
TTS_DEVICE=cuda TTS_DTYPE=bfloat16 TTS_MODELS_DIR=/data/models \
TTS_OUTPUT_DIR=/data/tts_out TTS_SERVER_PORT=8300 python -m agentic_tts.server
```

| 变量 | 覆盖 |
|---|---|
| `TTS_BACKEND` | 引擎（`qwen3` / `sine`） |
| `TTS_DEVICE` `TTS_DTYPE` `TTS_SEED` `TTS_MAX_RESIDENT` | 引擎运行参数 |
| `TTS_MODELS_DIR` `TTS_REPO_DIR` `TTS_VOICES_DIR` `TTS_OUTPUT_DIR` | 路径 |
| `TTS_SERVER_HOST` `TTS_SERVER_PORT` | 监听地址 |
| `TTS_LOG_LEVEL` | 日志级别 |

**并发**：单进程内合成是串行的（一把锁）。要吞吐就起多个进程/节点，每个独占一张卡；
在一张 8 GB 卡上放两个 worker 必然 OOM。

**产物清理**：`outputs/<时间戳>/` 会持续堆积（已 gitignore）。长期服务请挂定时清理，
模块不自动删 —— 删别人的音频产物应该是显式动作。

---

## 4 · 权重

```
Models/
  Qwen3-TTS-12Hz-1.7B-CustomVoice     # 内置音色
  Qwen3-TTS-12Hz-1.7B-VoiceDesign     # 音色设计
  Qwen3-TTS-12Hz-1.7B-Base            # 音色克隆
```

缺失时 `doctor` 会直接给出下载命令：

```bash
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir <path>
```

`third_party/qwen3_tts/` 是**上游源码的 vendored 副本**（commit `022e286`）——
本模块只从这里 import `qwen_tts`，不依赖环境里的 pip 安装、也不引用别的项目目录。
环境里那份 `qwen-tts 0.0.4` editable 安装是坏的（finder 指向已删除目录），
`doctor` 会打印**实际导入的** `qwen_tts.__file__` 供核对。

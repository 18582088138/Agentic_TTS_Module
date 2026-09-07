# 03 · 单元测试说明 / Unit tests

```bash
conda activate ov_env_py312
cd Agent_TTS_Module

python -m pytest tests -q                 # 默认：只跑离线用例（秒级，不加载任何权重）
python -m pytest tests -q -m real         # 真实模型：手动触发，每条几十秒起
python -m pytest tests/test_module.py -q   # 只跑一层
```

默认套件被 `pyproject.toml` 的 `addopts = "-m 'not real'"` 挡住了真实模型用例 ——
**日常 pytest 不该花几分钟**，那样就没人跑了。

## 分布

| 文件 | 条数 | 覆盖 | 权重 |
|---|---|---|---|
| `test_core.py` | 22 | 配置校验/路径解析/环境变量、**模式推断**、请求自校验、注册表懒加载 | ❌ |
| `test_text.py` | 22 | 事件标记、**标记与预处理的顺序**、四种拆分规则、URL/Markdown 预处理 | ❌ |
| `test_audio_guard.py` | 17 | 失控判据边界、重试种子、拼接（间隔/采样率统一/长度校验）、读写 | ❌ |
| `test_voices.py` | 11 | 音色档案增删改查、请求覆盖档案、克隆参数校验、角色映射 | ❌ |
| `test_module.py` | 23 | 门面端到端：停顿变静音、内部分块、失控重试、失败段不入合并、分组执行、种子稳定 | ❌ |
| `test_server_cli.py` | 21 | HTTP 端点与状态码映射、目录穿越防护、CLI 各命令 | ❌ |
| `test_subtitle.py` | 12 | SRT 格式与时间轴（含段间静音）、事件标记不进字幕、开关、耗时与音频时长是两个字段 | ❌ |
| `test_qwen3_real.py` | 5 | 真实权重：三种模式各一条短句、instruct 生效标记、URL 预处理 | ✅ |

离线 126 条，`sine` 引擎（正弦波占位）+ 纯函数，**3 秒跑完**。

## 几条刻意写死的断言（改代码时会先撞到它们）

| 断言 | 为什么必须钉住 |
|---|---|
| `parse_events` 先于 `prepare` | 实测 bug：数字规范化把 `[pause:500ms]` 写成 `[pause:五百毫秒]`，停顿悄悄消失、方括号还被念出来。见 `issues/001` |
| `[pause:1000ms]` ⇒ 合并时长 +1.00s | 停顿是唯一确定性的节奏控制手段，"确定"就体现在时长可断言 |
| 撞时长上限 ⇒ 判失控 | 41 秒噪声的波形指标一条不报、ASR 只回读「嗯 k」，**只有这一条判据测得出来** |
| 失败段不进合并音频 | 宁可短一段，也不能把噪声当成品交出去 |
| 段种子 = base + 段序号 | GUI「只重做第 3 句」的前提；重试种子用质数偏移，不与下一段撞 |
| 混模式多段 ⇒ 按 checkpoint 分组 | 8 GB 卡上一次轮动是几十秒；不分组则轮动次数 = 段数 |
| 音色名拼错 ⇒ 报错并列出可选 | 静默回落到默认音色是最难查的一类问题 |
| `qwen3` 不声明 `vocal_events` | 能力表不能撒谎；开源权重确实没有原生事件 token |
| 产物下载解析后再校验路径 | 只比字符串会被 `..%2f` 绕过 |
| 连续两次自动命名的生成落在不同目录 | 秒级时间戳 + 可重复的段文件名曾导致静默覆盖（issues/004） |
| `merge_audio` 期间引擎调用次数为 0 | 合并只许拼接，重跑会得到和试听过不一样的音频 |
| 字幕时间轴含段间静音 | 漏算则越往后字幕越提前，尾部差几秒 |
| `elapsed` 与 `seconds` 是两个字段 | 混在一起展示会被读成「合成只花了 1.8 秒」 |

## 真实模型用例的前置

1. `configs/config.yaml` 的 `engine.models_dir` 指向权重目录；
2. 三个 checkpoint 齐备（`python -m agentic_tts.cli doctor` 一眼看出缺哪个）；
3. 只跑最便宜的一条：`-m real -k custom_voice`。

`real_module` 夹具是 `scope="module"` 的 —— 权重加载几十秒，每条用例重载一次不可接受；
同一个实例还顺带验证了轮动加载（换模式时 `resident` 只剩一个）。

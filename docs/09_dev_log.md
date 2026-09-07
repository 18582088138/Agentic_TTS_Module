# 09 · 开发与测试记录 / Development log

一次做完 S1–S7（2026-09-07）。这里记**实际发生的事**：跑出来的数字、与方案的偏差、
以及三个已记档的问题。

---

## 1 · 交付内容

| 层 | 文件 | 状态 |
|---|---|---|
| 核心 | `core/{config,types,registry,errors,logging}.py` | ✅ |
| 引擎 | `engines/{base,sine,qwen3,placeholders}.py` | ✅ qwen3 已用真实权重实测 |
| 文本 | `text/{__init__,ascii_noise,lexicon,numbers,segment,events,split}.py` | ✅ 前五个从 verify_qwen3_tts **拷贝**后改导入 |
| 音频 | `audio/{io,guard}.py` | ✅ |
| 音色 | `voices/store.py` + `voices/voices.yaml` | ✅ |
| 门面 | `engine.py` | ✅ |
| 接口 | `cli.py`、`server/app.py`、`gui/app.py` | ✅ 三者都实跑过 |
| 配置 | `configs/config.yaml` | ✅ |
| 上游 | `third_party/qwen3_tts/`（commit `022e286`） | ✅ vendored |

## 2 · 测试结果

```
python -m pytest tests -q                → 108 passed, 5 deselected, 2.3s
python -m pytest tests -q -m real -s     → 5 passed, 101s（含三次权重加载）
```

**真实权重实测（本机 CPU，1.7B，float32，eager）**：

| 场景 | 数字 |
|---|---|
| CustomVoice 权重加载 | 4.3 s |
| 10 字短句（custom_voice） | 音频 1.84 s，**RTF 12.93** |
| voice_clone（x-vector） | 音频 2.16 s，RTF 10.10 |
| **轮动加载** | 日志实测 `卸载 base 腾出显存 → 加载 CustomVoice（4.3s）`，`resident` 始终为 1 |
| 4 段混模式样例（`samples/segments.yaml`） | 合计 17.32 s 音频，RTF 8.22，产物 4 段 + `merged.wav` + `manifest.json` |

样例那一跑同时验到了四件事：`[pause:400ms]` 让第 1 段内部分成 2 块、
`voice_design` 段触发了**真实的失控拦截并靠换种子救回**（`retries=1`）、
`[laugh]` 转 instruct 的 warning 正常冒出来、`effective_gen` 里如实记下了
`subtalker_temperature` 等来自 `generation_config.json` 的真值。

GUI 也起过一遍（`--engine sine`，HTTP 200，两个页签与各控件均渲染，日志无异常）。

## 3 · 与方案的偏差（都是当时定的四条决策的直接结果）

| 偏差 | 原因 |
|---|---|
| **不含任何 LLM 相关代码** | 决策④：本模块是纯 TTS，instruct/文案由调用方给 |
| 上游源码 **vendored 进仓库** | 补充要求①：不直接引用 `verify_qwen3_tts` 的文件 |
| `text/` 前五个文件是**拷贝**而非重写 | 同上；它们是纯函数且有 11 条 A/B 实测背书，重写只会丢掉依据 |
| 新增 `text/split.py`（四种拆条规则 + 角色映射） | 补充要求③ |
| GUI 从「最简」做到「逐段完整控件 + 逐句重生成 + 两种落盘」 | 补充要求②③ |
| `SplitRequest.min_chars` 默认 **0** | 说了"一句一段"就该是一句一段；碎片化在门面的模型级分段那一层解决 |
| 门面新增「按 checkpoint 分组执行」 | 补充要求④：8 GB 卡上轮动一次几十秒，不分组则轮动次数 = 段数 |

## 4 · 已记档的问题

| # | 问题 | 状态 |
|---|---|---|
| `issues/001` | 文本预处理会把 `[pause:500ms]` 改写成中文，导致停顿静默消失、方括号被念出来 | ✅ 已修（改为先解析标记）+ 回归用例 |
| `issues/002` | `sine` 的失控钩子按调用次数计数，被无关调用消耗掉 | ✅ 已修（改为按文本计数） |
| `issues/003` | 失控的**辅助**判据与「慢语速 instruct」会互相踩，可能白跑一次生成 | 已知行为，可配置，待 4060 上扫描后再定 |

## 5 · 返修轮（2026-09-07 晚，按人工反馈）

| # | 反馈 | 改动 |
|---|---|---|
| 1 | NiceGUI 上传报 `'UploadEventArguments' object has no attribute 'name'` | 3.x 把文件挪进了 `event.file`（`read()` 是 async）；两个大版本都兼容 |
| 2 | 第二次点生成没有新文件 | `issues/004`：产物目录唯一化 + GUI 拒绝重入 |
| 3 | 生成时要有运行状态提示，多段要指出跑到哪一段 | 门面加 `progress` 回调；顶栏状态灯带秒表，正在合成的卡片高亮，逐段状态标记（○▶●◐▲） |
| 4 | 多段产物要在同一个文件夹 | `new_run_name()` 定下 `gui-<时间戳>/`，一次会话（含逐段重做与合并）都写在里面 |
| 5 | 上传框太大、界面要科技风 | 新增 `gui/theme.py`（深色控制台，参考 DNA 工作台）；上传框压成虚线细带且只在克隆模式出现 |
| 6 | 多段要能统一设置声音（含克隆/设计） | 多段面板顶部统一配置，默认覆盖所有段；每段可展开「单独配置」覆盖 |
| 7 | 合并时每段都有音频就直接组合，别重生成 | `TTSModule.merge_audio()`；只补跑未生成/文本已改的段。单测用**引擎调用次数必须为 0** 钉死 |

离线单测 108 → **114 条**（新增：产物不覆盖、显式 name 仍复用、合并不重跑、
进度逐段上报、会话目录名唯一）。GUI 重新起过并渲染无异常。

### 返修轮补充（同日，三处小改）

| # | 反馈 | 改动 |
|---|---|---|
| 8 | 生成完要能打开落盘文件夹 | 播放器旁一个 📂 图标；⚠️ 打开的是**跑 GUI 那台机器**的文件管理器（浏览器无本地文件权限），所以路径也一直显示着 |
| 9 | 时间开销统计不对，明显快了 | 之前界面只显示 `result.seconds`（**音频时长**）却被读成耗时，两者差一个数量级（CPU 上 RTF≈13）。现在 `SynthResult` / `SegmentResult` 各加 `elapsed` 字段，GUI 与 CLI 都写成 `耗时 X.Xs　音频 Y.YYs　RTF Z` |
| 10 | 要能导出字幕，剪映能打开 | 新增 `audio/subtitle.py` + `output.subtitles` 开关（GUI 底部也有勾选框）。合并时写 `merged.srt`：**一段音频一条字幕**，时间轴含段间静音，UTF-8 带 BOM（否则剪映读中文乱码）。SRT 剪映/Premiere/DaVinci 都能直接导入，所以没退化成 txt |

字幕的两条实现细节：文字用**原文去掉事件标记**（`[laugh]` 是控制信息不是台词；
预处理后的文本删了 URL、把数字展开成读法，那是给模型看的）；时间轴必须算进段间静音，
漏算会让字幕越往后越提前。

实测样例（`sine`，3 段带 `[pause:400ms]` 与角色切换）：

```
1  00:00:00,000 --> 00:00:03,400   今天的人工智能资讯，三条重点。
2  00:00:04,000 --> 00:00:08,000   第一条，通义千问发布了新的语音合成模型。
3  00:00:08,600 --> 00:00:11,000   以上就是今天的全部内容。
```

段间那 0.6s 是换角色的停顿，与 `merged.wav` 的 11.0s 总长对得上。

离线单测 114 → **126 条**。

## 6 · 下一步（需要 4060 或需要人工听审）

1. **在 4060 上复测**：`doctor` 应显示 `cuda:0 / bfloat16 / 显存够`，
   跑 `-m real` 与样例 batch，记 RTF 与显存峰值。
   ⚠️ 同一份权重在 CPU 与 GPU 上**输出不同**（实测第 38 步就分歧），音色细节要用耳朵复核。
2. **人工听审**：本模块只保证"不吐噪声、不丢停顿、参数如实记录"，
   音色选型与参数扫描属于 `verify_qwen3_tts` 的活。
3. **Breeze-TTS 2 / IndexTTS-2**：能力表已占位（未实测），接入按 `docs/06_add_engine.md`。
4. **给下游（DailyNewsAssistant / K12）接线**：`scripts/client.py` 是零依赖样例；
   建议走 HTTP，避免把 torch 依赖拖进调用方进程。

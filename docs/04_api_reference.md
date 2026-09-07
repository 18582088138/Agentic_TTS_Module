# 04 · 接口参考 / API reference

三种调用面（库 / HTTP / CLI）**一一对应**：CLI 与 HTTP 都不含业务逻辑，只做参数解析。

---

## 1 · 库 / Library

```python
from agentic_tts import TTSModule, SynthRequest, BatchSynthRequest, SplitRequest
tts = TTSModule(config=None)        # None = configs/config.yaml；也可传路径或 Config
```

| 方法 | 说明 |
|---|---|
| `synthesize(req \| str, *, save=True, name=None, **kw)` | 合成一段（内部可能再分块，对外仍是一条音频） |
| `synthesize_many(BatchSynthRequest \| list[SynthRequest], *, progress=None)` | 多段；每段单独落盘，`merge=True` 时另出 `merged.wav`。`progress(event, index, total, result)` 在**工作线程**里被调用（`event` 为 `"start"`/`"done"`），界面据此显示"跑到第几段" |
| `split(SplitRequest \| str, rule=…, max_chars=…)` | 一整段文本 → `list[TextPiece]`（text / role / voice） |
| `merge_audio(paths, *, gaps_ms=None, texts=None, name=None, filename="merged.wav")` | 把**已有的**音频拼成一条，**不重跑模型**；给了 `texts` 且开了字幕就顺带写 SRT。返回 `{path, seconds, sample_rate, count, subtitle_path}` |
| `new_run_name(prefix="run")` | 取一个未被占用的产物目录名（界面用它让一次会话的产物落在同一目录） |
| `voice_list()` | 内置音色 + 本地档案 |
| `languages()` / `capabilities()` / `engines()` | 语言表 / 当前引擎能力 / 所有引擎的静态能力表 |
| `info()` | 引擎、设备、精度、checkpoint 路径、驻留情况、显存、依赖版本 |
| `release()` | 卸权重、清显存（也支持 `with TTSModule() as tts:`） |

### `SynthRequest`

| 字段 | 默认 | 说明 |
|---|---|---|
| `text` | 必填 | 可带 `[pause:400ms]` / `[laugh]` 标记 |
| `mode` | `None` | `custom_voice` / `voice_design` / `voice_clone`；留空按参数推断 |
| `voice` | `None` | 内置音色名（大小写不敏感）或 `voices.yaml` 里的档案名 |
| `language` | `Chinese` | ⚠️ 方言不在这里，靠音色拿（Dylan 京片子 / Eric 川普） |
| `instruct` | `None` | 语气/语速指令；`voice_design` 下是音色描述 |
| `ref_audio` / `ref_text` / `x_vector_only` | — | 克隆参数；`ref_audio` 只接受本地文件或 base64 |
| `gen` | `{}` | 生成参数覆盖（`temperature`、`max_new_tokens`、`non_streaming_mode`…） |
| `seed` | `None` | 留空 = `engine.seed + 段序号` |
| `role` / `pause_ms` | `None` | 多段时的角色名 / 本段之后的静音 |
| `prepare_text` / `segment` / `events` | `None` | 文本链路开关，`None` = 跟随配置 |

**模式推断**（写死在 `infer_mode`，有单测）：`ref_audio` 有值 → 克隆；无 `voice` 但有
`instruct` → 音色设计；否则内置音色。参数与 `mode` 冲突时**直接报错**，不静默忽略。

### `SynthResult`

`ok` / `seconds`（**音频时长**）/ `elapsed`（**墙钟耗时**）/ `rtf` / `sample_rate` /
`path`（合并音频）/ `subtitle_path`（`merged.srt`）/ `output_dir` / `warnings` /
`engine` / `segments[]`，`wav` 是 numpy 波形（不进 JSON）。

⚠️ `seconds` 是音频有多长，`elapsed` 是这次跑了多久 —— 两者差一个数量级，
展示时**必须分开写**，否则会被读成"合成只花了 1.8 秒"。

每个 `SegmentResult`：`index` `text` `spoken_text`（实际送进模型的文本）`mode` `voice`
`role` `seconds`（音频时长）`elapsed`（耗时）`rtf` `seed` `chunks` `retries` `failed`
`effective_gen` `instruct` `instruct_applied` `warnings` `path`。

> `effective_gen` 记的是**实际生效**的参数（直接调模型的 `_merge_generate_kwargs`），
> 不是"我传了什么" —— 事后最想知道的就是「这条音频到底用 temperature 0.9 还是 0.7 跑的」。

产物目录还会写一份 `manifest.json`（就是 `result.as_dict()`）。

---

## 2 · HTTP

```bash
python -m agentic_tts.cli serve         # 默认 127.0.0.1:8300，交互文档在 /docs
                                        # 图形界面在 /gui（同进程，共用一份权重）
python -m agentic_tts.cli serve --no-gui   # 只要 API
```

**界面默认挂在服务里。** `create_app(cfg, gui=True)` 把 NiceGUI 挂到 `/gui`，
和 HTTP 端点共用同一个 `TTSModule`、同一把锁（`module.lock`）。
另起一个 `cli gui` 进程会**再加载一份权重** —— 8 GB 卡上两份 1.7B 直接顶满，
所以那条路只留给「只要界面、不要 API」的场合。
The GUI shares the module and its lock; a separate process would load a second copy.

| 端点 | Body / 参数 | 返回 |
|---|---|---|
| `GET /health` | — | `{"status":"ok","backend":…}` |
| `GET /info` | — | 同 `info()` |
| `GET /engines` | — | 所有引擎的静态能力表（GUI 置灰用） |
| `GET /capabilities` | — | 当前引擎能力 + 语言表 |
| `GET /voices` | — | 音色列表 + 角色映射 |
| `POST /voices` | `VoiceInfo` | 新增/更新档案，落盘 `voices.yaml` |
| `DELETE /voices/{name}` | — | 删档案 |
| `POST /voices/ref_audio` | multipart `file` | 上传参考音频到 `voices/refs/` |
| `POST /tts/synthesize` | `SynthRequest` + `encoding`/`save`/`name` | JSON、`audio_base64`，或直接 `audio/wav` |
| `POST /tts/design` `POST /tts/clone` | 同上 | 语法糖，内部固定 mode |
| `POST /tts/batch` | `BatchSynthRequest` + `encoding` | 每段路径 + 合并路径 |
| `POST /tts/split` | `SplitRequest` | `{"segments":[{text,role,voice}]}` |
| `POST /gui/handoff` | `segments[]`/`text` + `title`/`source`/`voice`/`instruct`/`meta` | `{"token","gui_url"}` |
| `GET /gui/handoff/{token}` | — | 那张交接单（`status` / `files` / `run` / `done` / `total` / `meta`） |
| `GET /gui` | — | 图形界面本体（挂载点，见上） |
| `GET /outputs/{run}/{file}` | — | 取产物（路径解析后校验，挡目录穿越） |

**状态码是有含义的**，调用方可以自动处置：

| 码 | 含义 | 该怎么办 |
|---|---|---|
| 400 | 配置/音色错（`ConfigError`） | 改参数（报错里带可选项） |
| 409 | 当前引擎不支持该能力（`CapabilityError`） | 换引擎或降级 |
| 422 | 请求不合法（pydantic）或合成失败/失控 | 改请求或重试 |
| 503 | 服务端环境问题（权重缺失、显存不足） | 看服务端 `doctor` |

⚠️ **服务内只驻留一个 `TTSModule`，合成串行**（一把锁）：8 GB 卡上并发两条会同时要
两份权重，必然 OOM，而 OOM 报错完全看不出是并发导致的。要并发就横向起多个进程/节点。

### 交接单：把稿子交给 GUI 精修 / Handing a script to the GUI

调用方（如 DailyNewsAssistant）想让人在**多段界面**里换音色、克隆、逐句重掷时：

```
POST /gui/handoff  {"segments": ["第一段。", "第二段。"], "source": "DNA",
                    "return_url": "http://127.0.0.1:8080/"}
   → {"token": "20260908-101530-a1b2c3", "gui_url": "http://127.0.0.1:8301/?import=…"}
打开 gui_url            → 文本已经填进多段界面的各个文本框
在界面里生成/合并       → GUI 把 status=done 与产物清单写回同一条记录
                          （逐段生成时只写 status=running + done/total 供调用方显示进度，
                           点「全部生成」或「合并」才算完成，否则调用方会收走半成品）
                          生成完还会按 return_url **关掉本标签页／跳回调用方**
GET /gui/handoff/{token} → files: ["gui-20260908-101601/merged.wav", …]
GET /outputs/{run}/{file} → 取回产物
```

三条设计取舍 / Three decisions:

- **落成文件**（`outputs/handoff/*.json`）而不是放内存：`cli serve` 与 `cli gui`
  是**两个进程**，内存里的字典彼此看不见。
- **由服务端落盘**：调用方可能在另一台机器上，写不了这台机器的磁盘，它只能发 HTTP。
- **调用方轮询，服务端不回调**：TTS 这边不该知道调用方的地址，
  调用方也不必为此开一个入站端口。

token 出现在 URL 里并参与拼路径，因此**字符集是固定的**（`[A-Za-z0-9_-]{6,64}`）：
不校验的话 `../../` 就能读写输出目录之外的文件。交接单 7 天过期，新建时顺手清理。

客户端样例：`scripts/client.py`（零依赖、可直接拷进调用方项目）。

---

## 3 · CLI

```bash
python -m agentic_tts.cli <命令>        # 装了包之后可以直接写 tts <命令>
```

| 命令 | 用途 |
|---|---|
| `doctor` | 体检：配置、三份权重、设备/显存、**实际导入的 `qwen_tts` 是哪一份** |
| `engines` | 各引擎能力表（不加载权重） |
| `voices` | 可选音色 + 角色映射 |
| `speak TEXT [-v 音色] [-i 指令] [--no-prepare] [--no-segment]` | 合成一段 |
| `design INSTRUCT -t TEXT` | 音色设计 |
| `clone -t TEXT -a 参考音频 [-r 原话 \| --x-vector-only]` | 音色克隆 |
| `split 文件或文本 [--rule chars\|punct\|role\|blank] [--json]` | 拆条（`--json` 可喂给 `batch`） |
| `batch 段落文件 [--no-merge]` | 多段合成；`.yaml/.json` = 段落列表，`.txt` = 自动拆分 |
| `serve` / `gui` | 起服务 / 起界面 |

全局选项：`--config/-c` 指定配置文件，`--engine/-e` 临时换引擎（`sine` 可零权重试跑）。

段落文件格式：

```yaml
- text: "旁白第一句"
  voice: Serena
  role: 旁白
- text: "记者追问"
  voice: Ryan
  role: 记者
  pause_ms: 800
```

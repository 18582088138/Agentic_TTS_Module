# 10 · API 后端引擎 / API backend engine (DRAFT)

> 状态：**草案 v0.1，待对齐**。本文先讲清"长什么样 / 怎么切 / 怎么配 / 怎么测"，
> 写代码时按本文逐项落地。

---

## 0 · 目标 / Goals

加一个 `api` 引擎，复用门面（`TTSModule`）的全部能力：
文本链路 / 事件标记 / 分段 / 拼接插静音 / 失控兜底 / 角色 / 字幕 / GUI / HTTP。
**门面一行不动**，只新增一个引擎模块 + 一组配置 + 一组测试。

之所以"先做 API 过渡"——本地 Qwen3-TTS 1.7B 在 8 GB 显卡上 1 个 checkpoint
就要 5 GB，三个 mode 各绑一份权重轮动加载，RTF 不理想。API 后端绕开这个限制。

---

## 1 · 架构 / Architecture

### 1.1 引擎层（`agentic_tts/engines/api_backend.py`）

完全按 `docs/06_add_engine.md` 的"四步走"实现：

- 实现 `TTSEngine.synthesize(chunk) -> RawAudio`
- 声明 `declared_capabilities()`
- **重依赖在函数体内 import**（httpx / 任何 SDK）
- 静态能力表：参考 Qwen3 开源权重能做什么就声明什么，**不夸大**
- 不支持的就 `require()` 报错，**绝不静默降级**

### 1.2 HTTP 客户端层（`agentic_tts/engines/api_http.py`）

**与具体供应商解耦**的薄客户端层。定义一个内部协议 `APIClient`：

```python
class APIClient(Protocol):
    def synthesize(self, *, mode: Mode, text: str, language: str,
                   speaker: str | None, instruct: str | None,
                   ref_audio: bytes | None, ref_text: str | None,
                   x_vector_only: bool, voice_id: str | None,
                   gen: dict, seed: int) -> APISynthResult: ...
    def upload_voice(self, *, name: str, audio: bytes, ref_text: str | None) -> str: ...
    def resolve_voice_id(self, *, mode: Mode, ref_audio_hash: str,
                         ref_audio: bytes, ref_text: str | None,
                         speaker: str | None) -> str: ...
```

然后为每个供应商写一个 adapter：
- `DashScopeAdapter`（第一个落地，对应 Qwen3-TTS 官方 API）
- 后续可加 `OpenAIAdapter` / `ElevenLabsAdapter` / …

### 1.3 引擎与客户端的关系

```
TTSModule.synthesize
  └─ APITTSEngine.synthesize(chunk)
       └─ self.client.<具体方法>(...)        ← 通过 config.api.provider 选 adapter
            └─ httpx.post(...)               ← 实际 HTTP
```

`APITTSEngine` 自己**不直接发 HTTP**——它持有 `APIClient`，把 chunk 翻成 client 的入参。
这样换供应商只换 client，引擎不变。

---

## 2 · 切换粒度 / Selection granularity

**Q3 的决定：单个请求级别切换。**

### 2.1 三种切法

| 粒度 | 怎么切 | 优 | 劣 |
|---|---|---|---|
| 顶层（config） | `engine.backend: api` | 简单 | 不灵活，全局只能一种 |
| **单请求** | `SynthRequest.engine="api"` | 调用方按需切 | 多一条字段 |
| 按 voice | `voices.yaml` 里每个 voice 标 engine | 混跑灵活 | 复杂、需协调采样率 |

选**单请求**，理由：
- 顶层切换不够灵活（"过渡"必然有来回切的一天）
- 按 voice 切换需要引擎间音频协调（采样率/格式/音量），得不偿失
- 单请求切换只多一个字段，调用方完全控制

### 2.2 实现细节

`SynthRequest` 新增字段：

```python
engine: Optional[str] = None  # None = 用 config.engine.backend
```

`TTSModule._synthesize_one()` 里：

```python
# 现在：
self.engine.require_mode(voice.mode)
# 改成：
engine = self._pick_engine(req)  # req.engine or config.engine.backend
engine.require_mode(voice.mode)
chunks, warnings = self._plan(req, voice)
...
raw = engine.synthesize(SynthChunk(...))
```

**关键约束**：本地引擎与 API 引擎**仍走同一个锁**（`TTSModule.lock`），
不会同时在显存里要两份权重，也不会同时并发打爆 API 配额。

### 2.3 与现有 voices 系统的关系

`voices.yaml` 里**不**加 `engine` 字段，保持现状。理由：
- 当前所有 voice 都是 qwen3（local）
- API 引擎的 voice 通过 speaker（内置音色）或 voice_id（克隆）解决，不在档案层加新概念
- 如果未来真的要混跑，再单独加 `VoiceInfo.engine` 字段、并且加混跑校验（避免同一段文字里不同段要不同采样率时出问题）

---

## 3 · 配置 / Configuration

### 3.1 `configs/config.yaml` 新增节

```yaml
# ---------------- API 后端 / API backend (optional) ----------------
api:
  # 是否启用 API 引擎。"false" 也不报错，只是 engine="api" 时初始化失败
  enabled: false

  # 供应商：dashscope | openai | elevenlabs | ...（第一个落地 dashscope）
  provider: dashscope

  # 端点与鉴权
  endpoint: ""               # 空 = 用 provider 默认；想自建代理可以填
  api_key: ""                # ⚠️ 留空 = 必须从环境变量 TTS_API_KEY 读
  timeout_s: 60              # 单次合成超时
  max_retries: 2             # 5xx / 网络错误时重试次数
  retry_backoff_s: 1.0       # 指数退避基数

  # 通用生成参数（透传到各 provider）
  options: {}                # 例如：{ "sample_rate": 24000, "format": "wav" }

  # voice_id 缓存：上传过的参考音频 → 返回的 voice_id
  # 避免同一段音频每次合成都重新上传
  voice_cache_path: "./voices/api_voice_cache.json"

  # 限流（可选；0 = 不限）
  rate_limit_per_min: 0

  # provider 私有参数（不通用，每个 provider 自己解析）
  provider_options: {}
```

### 3.2 环境变量覆盖

复用 `_apply_env()` 的白名单机制，新增：

```python
"TTS_API_ENABLED":      (self.api, "enabled", bool),
"TTS_API_PROVIDER":     (self.api, "provider", str),
"TTS_API_KEY":          (self.api, "api_key", str),
"TTS_API_ENDPOINT":     (self.api, "endpoint", str),
"TTS_API_TIMEOUT_S":    (self.api, "timeout_s", int),
"TTS_API_MAX_RETRIES":  (self.api, "max_retries", int),
```

**鉴权优先级**（避免 key 进 git）：
1. 进程环境变量 `TTS_API_KEY`
2. 配置文件 `api.api_key`（仅本机用，**禁止提交**）
3. 都没有 → 初始化时报清晰错误（"找不到 API key……"）

⚠️ **`api.api_key` 字段加进 `.gitignore` 提示**（跟 `models_dir` 那条绝对路径一样的处理）。

### 3.3 校验

`Config.validate_runtime()` 新增：

```python
if self.api.enabled:
    if self.api.provider not in _API_PROVIDERS:
        raise ConfigError(...)
    if not self.api.api_key and not os.environ.get("TTS_API_KEY"):
        raise ConfigError(
            "启用了 api 后端但没配 API key　"
            "设 TTS_API_KEY 环境变量，或在 configs/config.yaml 的 api.api_key 填"
        )
    if self.api.timeout_s <= 0:
        raise ConfigError("api.timeout_s 必须为正")
```

---

## 4 · Voice clone：本地缓存策略（Q2 决定）

### 4.1 现状 vs 需求

- 现状：local 引擎直接传 `ref_audio` 路径
- API 引擎：上游通常要求"先创建 voice → 拿 voice_id → 用 voice_id 合成"
- 用户期望："同一个 ref_audio 不要再传一次"

### 4.2 方案：本地 voice_id 缓存

- 位置：`voices/api_voice_cache.json`（gitignore）
- key：参考音频的 SHA-256（**内容 hash**，不依赖路径/文件名）
- value：`{ voice_id, provider, ref_text, x_vector_only, uploaded_at }`
- 命中条件：hash + provider + ref_text + x_vector_only 都一致

伪代码（写在 `api_http.py` 的 `BaseAPIClient` 里）：

```python
def resolve_voice_id(self, *, mode, ref_audio, ref_text, x_vector_only, ...):
    if mode is not Mode.VOICE_CLONE:
        return None
    audio_bytes = self._read_audio(ref_audio)
    digest = sha256(audio_bytes).hexdigest()
    cached = self._cache_lookup(digest, ref_text=ref_text, x_vector_only=x_vector_only)
    if cached:
        return cached
    voice_id = self.upload_voice(name=digest[:12], audio=audio_bytes, ref_text=ref_text)
    self._cache_save(digest, voice_id, ref_text=ref_text, x_vector_only=x_vector_only)
    return voice_id
```

### 4.3 边界情况

- 缓存文件不存在 → 视为空缓存，正常上传
- 缓存命中但 provider 端 voice_id 失效（404）→ 删缓存条目、重传
- 缓存文件损坏 → 备份成 `.broken`、重建空缓存
- ref_audio 是 base64 / URL —— **API 引擎目前只接本地路径**（沿用 local 的约束）

---

## 5 · 能力声明 / Capabilities

两个引擎的能力表**完全对齐**（除 BATCH）—— 这保证 GUI 永不灰任何开关，
门面 `parse_events()` 决策时也不必区分走哪个引擎。

| 能力 | `qwen3`（local） | `api` | 备注 |
|---|---|---|---|
| `CUSTOM_VOICE` | ✅ | ✅ | 内置 speaker 合成 |
| `VOICE_DESIGN` | ✅ | ✅ | 自然语言描述音色，必填 `instruct` |
| `VOICE_CLONE` | ✅ | ✅ | 参考音频克隆，必填 `ref_audio` |
| `CLONE_ICL` | ✅ | ✅ | 传 `ref_text`（不走 `x_vector_only`） |
| `INSTRUCT` | ✅ | ✅ | 走 `instruct` 参数 |
| `VOCAL_EVENTS` | ✅ **改为声明支持** | ✅ | 见 §13——**实际行为**两引擎都走"降级到 instruct 转译"（无原生事件 token） |
| `BATCH` | ✅ | ❌ | 第一版 API 不做（门面会按段循环，对外行为一致） |

### 5.1 引擎私有配置

- `qwen3`：`codes_per_second = 12.5`、`default_max_new_tokens = 8192`
  → guard 的"时长撞上限"主判据有效
- `api`：`codes_per_second = None`、`default_max_new_tokens = None`
  → guard 的"时长撞上限"主判据**失效**（API 端不暴露 max_new_tokens），
  fallback 到 `chars_per_sec` 启发式。在 `SynthResult.warnings` 里加一条说明

### 5.2 引擎必填参数校验（与 Python 端 `SynthRequest._check_consistency` 对齐）

| Mode | 必填 | 互斥 | demo URL: https://9izbunktg1rxa.space.mcode.cn 可手动触发 |
|---|---|---|---|
| `custom_voice` | `voice` | — | 内置 speaker 即可 |
| `voice_design` | `instruct` | 不接受 `ref_audio` | 自然语言描述 |
| `voice_clone` | `ref_audio` | `x_vector_only=false` 时必填 `ref_text` | 参考音频克隆 |

校验失败直接报 `ValueError`，**绝不静默降级**（参见 0.6B 静默丢 `instruct` 的教训）。

---

## 6 · 错误处理 / Error handling

错误分类（每类都给出可操作的建议，绝不只抛原始 HTTP 错）：

| 错误 | 处理 |
|---|---|
| 401 / 403 | "鉴权失败：检查 TTS_API_KEY 是否对、是否过期" |
| 404 | "端点不对：检查 api.endpoint 或 provider 名字" |
| 429 | 触发 max_retries + 指数退避；仍 429 → 报"被限流，建议降速" |
| 5xx | 触发 max_retries + 退避；仍 5xx → 报"上游故障，重试 N 次仍失败" |
| 超时 | "超过 {N}s 没回——是不是文本太长或上游卡了？长文本调小 max_chars" |
| 网络断 | "连不上 {endpoint} —— 是不是没设代理？公司网络要走 HTTP_PROXY" |
| 400 + 业务码 | 直接把上游业务错误原文透传（往往是最有信息量的） |
| voice_id 失效 | 缓存条目删掉、重传 |

每条都进 `SynthResult.warnings`（不致命）或抛 `SynthError`（致命）。

---

## 7 · 测试 / Testing

### 7.1 单元测试（默认跑）

- `test_api_engine.py`：
  - 配置解析（缺 key、provider 错、enabled 切换）
  - engine=api 走通门面的最小流程（**用 mock httpx，不发真请求**）
  - `resolve_voice_id` 缓存命中 / 未命中 / 缓存命中但上游 404
  - 错误分类（401/429/5xx/超时）→ 落到 `warnings` 还是抛 `SynthError`
- `test_api_http.py`：
  - cache 读写、损坏恢复
  - SHA-256 hash 一致性

### 7.2 真请求测试（`@pytest.mark.real` + `@pytest.mark.api`）

- 配置 `TTS_API_KEY` 才跑
- 跑 3–5 句短文，断言：
  - `result.ok == True`
  - `result.seconds > 0`
  - wav 长度与 `sample_rate * seconds` 一致
  - 同一段再跑一次，结果一致（缓存命中）

### 7.3 门面/服务/GUI 的测试**不用改**

它们走 `sine` 引擎，与新引擎无关——这是 `06_add_engine.md` 的硬约束。

---

## 8 · CLI / HTTP 暴露

### 8.1 CLI

新增子命令：

```bash
tts api-doctor               # 检查 api 配置：enabled / provider / key 在不在 / 端点通不通
tts api-test "你好"          # 用 API 引擎合成一句（不传 mode，自动 custom_voice）
tts api-voice upload <wav>   # 上传参考音频，打印 voice_id
```

### 8.2 HTTP

- 已有 `POST /tts/synthesize` **不动**：请求体里加 `engine: "api" | "qwen3"` 即可
- 新增 `GET /api/info`：返回当前 api 配置（**mask 掉 api_key**）+ provider 列表
- 新增 `GET /api/voice-cache`：列出已缓存的 voice_id（方便调试）

---

## 9 · 不做的事（明确划界）

- ❌ **不在引擎里重写文本链路**（门面已实现）
- ❌ **不实现 batch API**（门面会按段循环，行为对调用方一致）
- ❌ **不支持 ref_audio 是 URL**（沿用 local 约束，理由：代理配置不通用）
- ❌ **不做 provider 自动 fallback**（dashscope 挂了不会自动切 openai——会让用户搞不清是哪家在响）
- ❌ **api_key 不进 voice_id 缓存文件**（缓存文件可能被读出来看）

---

## 10 · 落地步骤 / Implementation order

每步独立可测、独立 commit：

1. **schema + 校验**（不改行为）—— `core/config.py` 加 `APIConfig` + 校验；`config.yaml` 加 `api:` 节
2. **APIClient 抽象 + DashScope adapter**（含 httpx 客户端、错误分类、voice 缓存）
3. **APITTSEngine** —— 持有 client，按 `chunk` 调对应方法
4. **SynthRequest.engine + TTSModule 单请求切换** —— 让 `engine="api"` 在门面上跑通
5. **CLI 子命令** —— `api-doctor` / `api-test` / `api-voice upload`
6. **HTTP 端点** —— `/api/info` / `/api/voice-cache`；`/tts/synthesize` 支持 `engine` 字段
7. **测试 + 文档** —— 单元测试 + 真请求测试 + 改 `04_api_reference.md`

每步都给 diff，**你 review 后再 commit**。

---

## 11 · 已决定 / Decisions locked-in

| # | 议题 | 决定 |
|---|---|---|
| Q1 | Provider 选型 | **MiniMax**（用户已有 API plan，2.8 原生支持 sound tags） |
| Q2 | API key 管理 | `MINIMAX_API_KEY` 环境变量优先 + `api.api_key` 配置 fallback |
| Q3 | `.gitignore` | 忽略 `voices/api_voice_cache.json`、`voices/refs/*.mp3` |
| Q4 | GUI 切换粒度 | 顶栏三档开关 `local` / `api` / `auto(fallback)` |
| Q5 | fallback 策略 | local 失败 → API，**仅一次**，尊重显式选择 |
| Q6 | VOCAL_EVENTS 声明 | 两个引擎都声明支持 |
| Q7 | Vocal events 实际行为 | local 降级 / **API 走 MiniMax 2.8 原生标记** |
| Q8 | 模型 | `speech-2.8-turbo`（默认，便宜）；`speech-2.8-hd`（高质量） |
| Q9 | Endpoint | `https://api.minimax.io` |
| Q10 | fallback HTTP 返回码 | 200 + warnings |
| Q11 | fallback 计数 | 暴露到 `/info` 端点 |
| Q12 | 提交粒度 | 每落地步骤独立 commit |
| Q13 | 测试策略 | 只测新代码相关用例，不开全量 |
| Q14 | Branch | `feature/api-integration-test` |
| Q15 | 默认参考音频 | `voices/refs/ref_audio_zh.mp3` + `voices.yaml` 的 `default_ref_voice` |

---

## 12 · GUI 开关与 fallback / GUI toggle & automatic fallback

### 12.1 需求拆解

需求拆成两条独立但相关的功能：

1. **GUI 手动切换**：用户能在界面里切 `local` / `api`
2. **自动 fallback**：local 引擎失败时**自动用 API 重试一次**，用户无感知

### 12.2 实现策略

两条需求走不同的代码位置：

| 需求 | 实现位置 | 粒度 |
|---|---|---|
| GUI 手动切换 | GUI 状态（`gui/app.py`）+ `TTSModule` 引擎切换 | 全局（GUI 内一次切到底） |
| 自动 fallback | `TTSModule._synthesize_chunk()` 内部 | 单 chunk |

### 12.3 GUI 手动切换

- 顶栏加一个三档开关：`local` / `api` / `auto(fallback)`
- 切到 `local` 或 `api`：等价于"这次请求强制走这一档"，与单请求切换语义一致
- 切到 `auto`：正常用 local，失败自动 fallback（**默认档**）
- HTTP 服务的 `POST /tts/synthesize` 加 `engine` 字段；HTTP 不感知"auto"概念（HTTP 一次一调，调用方自己决定）
- 切换时打 warning 日志（"engine switched: local → api"）

**约束**：切到 `api` 时如果 api 配置不完整（缺 key、provider 错），**报错而不降回 local**——
避免"我切到 API 了为啥还在跑 local"这种困惑。

### 12.4 自动 fallback

**触发条件**（任一即触发）：

- local 引擎 `synthesize()` 抛 `EngineError` 或 `SynthError`（加载失败、推理失败、显存 OOM）
- 失控重试用尽（guard retry 全部失败）

**不触发**的情况（保持原错误抛出）：

- `ConfigError`、`CapabilityError`——参数错，不该 fallback
- 用户在请求里显式 `engine="qwen3"`——**尊重显式选择**，不自动切
- 失控 heuristic 命中但引擎本身没抛错（重试机制在管）

**实现位置**：`TTSModule._synthesize_chunk()` 改成：

```python
def _synthesize_chunk(self, chunk, voice, seed, gen):
    if self._mode == "auto" and self._local_engine_available():
        try:
            return self._do_synthesize(self._local_engine, chunk, voice, seed, gen)
        except (EngineError, SynthError) as exc:
            if not self._can_fallback():
                raise
            warnings.append(f"local 合成失败，自动 fallback 到 API：{exc}")
            raw = self._do_synthesize(self._api_engine, chunk, voice, seed, gen)
            raw.warnings.append("本段由 API 引擎 fallback 生成")
            return raw
    engine = self._pick_engine_for_mode()
    return self._do_synthesize(engine, chunk, voice, seed, gen)
```

**关键约束**：

- fallback **只触发一次**——API 也失败直接抛，不做"再 fallback 回 local"的死循环
- 必须在 `SynthResult.warnings` 里**显式记录**："本段由 API 引擎 fallback 生成"
- 落到 manifest.json 的 `fallback_count` 字段
- 离线自测（`--engine sine`）时 fallback 默认关闭

### 12.5 Config 新增项

```yaml
api:
  ...
  # 自动 fallback：local 失败时自动重试 API
  fallback:
    enabled: true          # 整体开关；false = 永不自动 fallback
    on_errors: ["engine_error", "synth_error"]   # 哪些异常类型触发
    skip_voices: []        # 不参与 fallback 的 voice（高级用法）
```

### 12.6 GUI 行为细节

- 三档开关的位置：放在顶栏 voice 选择器旁边（视觉上跟 voice 配置在一起）
- fallback 触发时状态灯多一档："合成中" → "fallback 中" → "完成"
- 顶栏"auto"档的小指示灯 fallback 触发时**闪一下**（3 秒恢复），给用户视觉反馈
- fallback 计数显示在 info 区

---

## 13 · 能力表修订与 VOCAL_EVENTS / Capabilities update

### 13.1 现状

- `qwen3`（local）**不声明** `VOCAL_EVENTS`（开源权重没原生事件 token）
- 门面策略：引擎不支持时把 `[laugh]` 等标记**转成中文 instruct 提示**
- 这条降级路径走的是 `parse_events()` 里 `native=False` 的分支

### 13.2 新需求解读

> API 和 local 模型都需要支持 vocal event 功能，
> 只是目前 Local 模型暂不支持,后续更换模型后仍需这个功能

拆成两层：

1. **能力声明**：两个引擎**都**声明 `VOCAL_EVENTS` —— GUI 永不灰这个开关
2. **实际行为**：local 现在是"降级到 instruct 转译"；第一版 API 也是降级；以后真有原生支持的引擎换上即自动生效

### 13.3 修订后

| 引擎 | `VOCAL_EVENTS` 声明 | 实际行为 |
|---|---|---|
| `qwen3`（local） | ✅ **改为声明支持** | 降级：标记 → 中文 instruct（行为不变，只是声明改了） |
| `api` | ✅ 声明支持 | 第一版降级（DashScope Qwen3-TTS 没原生 token）；后续换模型自动生效 |

### 13.4 为什么"声明支持"是对的

- 你的设计文档明确："`Capability` 由引擎静态声明，GUI 据此置灰开关"
- **声明能力 ≠ 一定能交付**——声明"我能做"，实际能不能做是另一回事
- 真实情况写进 `warnings`，**不让能力表撒谎**：
  - `qwen3` 声明 `VOCAL_EVENTS`，`warnings` 写"无原生事件支持，已转 instruct，效果不保证"
  - `api` 第一版同 qwen3；以后真有原生支持时 warnings 自然消失

**结果**：GUI 的 vocal events 开关**永远不灰**——正符合"后续更换模型后仍需这个功能"。

### 13.5 端到端流程

GUI 永远能开 vocal events（两引擎都声明支持）
↓
门面 `parse_events()` 调 `engine.supports(Capability.VOCAL_EVENTS)` → True
↓
但引擎拿到的 `chunk.text` **已不含标记**（门面板转 instruct 去了）
↓
**实际效果**：local 引擎跑出来 = 语气提示生效（instruct 转译）
↓
API 引擎跑出来 = 同样表现（第一版也是降级）
↓
**将来**真接原生事件引擎时，门面按 `supports()` 自动切原样传递——不动门面、不动引擎

### 13.6 第一版简化方案

保持现状：`Capability.VOCAL_EVENTS` 即"支持某种形式"。
引擎只声明 bool，**不引入 `event_support_level` 枚举**（复杂度上升，第一版不需要）。
扩展点已经在了，将来真有 `NATIVE` 引擎再升级。

### 13.7 GUI 改动

- vocal events 启用开关不再因引擎灰显
- 但说明文字改一下："vocal events 已开启（当前由引擎以 instruct 方式实现，效果取决于引擎）"

---

## 14 · 联调 Demo / Live demo for review

为了在**没有 Python 运行环境的情况下**也能 review API 行为，部署了一个静态 demo：

- **URL**：https://9izbunktg1rxa.space.mcode.cn
- **手机 / 电脑浏览器**直接打开即用
- 完整覆盖：
  - ✅ 三档引擎切换（local / api / auto）
  - ✅ 三种合成模式（custom_voice / voice_design / voice_clone）
  - ✅ vocal events 解析与降级（`[pause:400ms]` / `[laugh]` 等）
  - ✅ 自动 fallback 触发与 warnings 标注
  - ✅ voice_id 缓存（API 引擎特性：SHA-256 + ref_text + x_vector_only）
  - ✅ JSON 响应形状与 Python `SynthResult.as_dict()` 对齐
  - ✅ Mode 必填校验（与 Python `SynthRequest._check_consistency` 行为一致）
- **限制**：
  - 音频用浏览器 `SpeechSynthesis`，效果**不是**真 TTS
  - API 引擎走 mock，不真发 HTTP 请求
  - 仅验证**接口形状与流程逻辑**，不验证**音频质量**

---

## 15 · 已决定汇总 / Decisions summary

| # | 议题 | 决定 |
|---|---|---|
| 1 | Provider 选型 | DashScope（第一个落地） |
| 2 | API key 管理 | `TTS_API_KEY` 环境变量优先 + 配置 fallback |
| 3 | `.gitignore` | 忽略 `voices/api_voice_cache.json`；`api.api_key` 正常提交 |
| 4 | GUI 切换粒度 | 三档开关 `local` / `api` / `auto(fallback)` |
| 5 | fallback 策略 | local 失败 → API，**仅一次**，尊重显式选择 |
| 6 | VOCAL_EVENTS 声明 | 两个引擎都声明支持 |
| 7 | Vocal events 实际行为 | 第一版两引擎都走 instruct 降级；后续自动切原生 |

---

## 16 · 待你拍板（最后 3 个开放问题）

1. **fallback 时 HTTP 返回码**：保持 200 + warnings，还是 207（Multi-Status）？
   我的建议：保持 200 + warnings 在 manifest 里。
2. **fallback 计数**要不要暴露到 `/info` 端点？我的建议：暴露。
3. **代码提交粒度**：每落地步骤独立 commit（按 §10），还是合并到一个 commit？
   我的建议：独立 commit，便于 review 与回滚。

回这三点 + 整体方案 OK，我开干。

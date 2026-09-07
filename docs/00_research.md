# 00 · 调研 / Research

> 时间：2026-09-07　环境：`conda ov_env_py312`
> 所有「实测」条目都在本机跑过或直接读过源码/权重文件，未验证的一律标 ⚠️。

---

## 1 · 上游 Qwen3-TTS 的真实调用面

源码：`verify_qwen3_tts/Qwen3-TTS/qwen_tts/inference/qwen3_tts_model.py`（commit `022e286`）

三个入口 **各绑一个 checkpoint，门禁在模型侧硬写**（换 API 必须换权重）：

| API | checkpoint | 关键参数 |
|---|---|---|
| `generate_custom_voice` | `*-CustomVoice` | `text, speaker, language, instruct, non_streaming_mode=True` |
| `generate_voice_design` | `*-VoiceDesign` | `text, instruct, language, non_streaming_mode=True` |
| `generate_voice_clone` | `*-Base` | `text, language, ref_audio, ref_text, x_vector_only_mode, non_streaming_mode=False` |

- 返回 `(wavs: list[np.ndarray], sample_rate)`，支持 batch（list 入参）。
- 校验走模型方法：`model.get_supported_languages()` / `model.get_supported_speakers()`。
- 内置音色 9 个：Vivian / Serena / Uncle_Fu / Dylan(京) / Eric(川) / Ryan / Aiden / Ono_Anna / Sohee。
- 生成参数三级合并：**显式传参 > `generation_config.json` > 代码硬编码兜底**，
  模型自带 `_merge_generate_kwargs()` 可直接拿到「实际生效」的那份。
- `x_vector_only_mode=False` 是 ICL 克隆，**必须给 `ref_text`**；`True` 只用 speaker embedding。

### 已下载的权重（`C:/Users/test/Downloads/xkd/Models`）

`Qwen3-TTS-12Hz-1.7B-Base` / `-CustomVoice` / `-VoiceDesign` / `Qwen3-TTS-Tokenizer-12Hz`
（另有 0.6B-CustomVoice 与一份 0.6B 的 OV IR，本模块不用）。

---

## 2 · 从 verify_qwen3_tts 继承的实测结论（可直接复用，不必重验）

| # | 结论 | 出处 |
|---|---|---|
| 1 | **模型会失控**：停不下来一直生成到 `max_new_tokens`，产出几十秒噪声/静音且**不报错**。判据只能是「时长撞上限」，静音检测和波形指标都测不出来。12Hz → `秒 = codes / 12.5` | `docs/issues/004`、实测 |
| 2 | **别为了可复现关采样**：同 12 条文本，贪心解码 7 条失控，官方采样默认值 0 条失控 | 实测 |
| 3 | **文本预处理收益极大且不均匀**：URL 不处理 CER 4.43（ASR 回读字数是原文 4 倍），Markdown 残留 0.83→0.17；小数/emoji 处理了白做 | `docs/07_textkit_spec.md` 11 条 A/B |
| 4 | **0.6B CustomVoice 静默丢弃 `instruct`**（`qwen3_tts_model.py:799`）；1.7B 上 instruct 有效，「语速平缓/偏快」时长差 30–42%、CER 不变差 | 实测 |
| 5 | **方言不能用 `language=` 选**，`*_dialect` 在 `supported_languages` 里被过滤；方言只能靠音色（Dylan 京片子 / Eric 川普） | 读源码 |
| 6 | **SSML / LaTeX 标记在开源权重里不存在**，`Qwen3TTSProcessor` 无任何标记解析；逐词读法控制只能靠文本预处理 | 全仓库 grep 零命中 |
| 7 | 1.7B bf16 权重 ≈3.9 GB，跑起来 ≈5 GB；8 GB 卡**装得下一个、装不下两个** → 模型池上限默认 1，换 checkpoint 先卸再载 | 实测 |
| 8 | 仅 `del` 不释放显存，必须 `torch.cuda.empty_cache()`，否则下次加载 OOM 而报错完全看不出原因 | 实测 |
| 9 | 只固定 seed **不足以**复现：主 talker 与 sub-talker 都在采样，要逐层一致得两个 `do_sample` 都关 | 实测 |
| 10 | 参考音频的 URL 加载走 `urllib.request.urlopen`，**不读代理配置**，公司网络下会超时 → 只支持本地文件/base64 更稳 | 实测 |

---

## 3 · Vocal Events：先把事实说清楚

**Qwen3-TTS 开源权重没有原生 vocal-event token。** 核实方式：

- `tokenizer_config.json` 的 special tokens 全表里只有 `<|im_start|>` / `<|audio_*|>` /
  `<|vision_*|>` 这类，**没有** laugh / breath / sigh / cough 之类的事件 token；
- 上游仓库对 `laugh|breath|vocal|event` 全仓库 grep 零命中（`README` 里那条 "breath support"
  是 voice-design 的自然语言描述，不是标记）。

所以本模块的 `vocal_events` 只能是**能力声明 + 适配层**：

| 事件类型 | Qwen3-TTS 实现方式 | 可靠性 |
|---|---|---|
| 停顿 `[pause:400ms]` | 分段合成 + 在拼接处插静音 | ✅ 确定性，完全可控 |
| 笑/叹气/哽咽等 | 转成 `instruct` 提示（1.7B 有效，见结论 4）+ 文本侧保留语义线索 | ⚠️ 尽力而为，不保证出声 |
| 原生事件标记 | 留给有原生支持的引擎（Breeze-TTS 2 / IndexTTS-2）在各自 engine 里映射 | 本阶段不验证 |

**结论：能力必须按引擎声明（capability 表），调用方按声明降级，而不是假装所有引擎都一样。**

---

## 4 · 环境实测（`ov_env_py312`，本机）

```
torch 2.8.0+cpu   torchaudio 2.8.0+cpu   cuda_available=False
transformers 4.57.3  accelerate 1.12.0  einops 0.8.2  onnxruntime 1.27.0
soundfile 0.14.0  librosa 0.11.0  numpy 1.26.4（被 funasr 锁在 1.x）
fastapi 0.139.2  uvicorn 0.51.0  pydantic 2.12.5  pydantic-settings 2.14.2
typer 0.25.1  pytest 9.1.1  PyYAML 6.0.3
```

**依赖全齐，本阶段不需要 pip install 任何东西。** 两个坑：

1. `qwen-tts 0.0.4` 是**坏的 editable 安装** —— finder 指向已删除的
   `openvino_notebooks/notebooks/qwen3-tts/Qwen3-TTS/qwen_tts`。阴险之处是它不总是失败：
   cwd 恰好在某份源码仓库根时 `sys.path` 的 `''` 会让它悄悄导到那一份上。
   → **本模块不修全局安装**（会影响共用同一环境的 DNA），改为显式把配置里的
   `repo_dir` 插到 `sys.path[0]`，并在 doctor 里打印实际导入的 `qwen_tts.__file__`。
2. 磁盘上有两份 Qwen3-TTS 源码且 commit 不同：`Models/Qwen3-TTS`（`1ab0dd7`，DNA 在用）
   与 `verify_qwen3_tts/Qwen3-TTS`（`022e286`，已验证的那份）。
   → 配置默认指向**已验证的那份**，并在 manifest 里记下路径。

---

## 5 · 可参考的骨架：Agentic_RAG_Module

`Multi-agent-app/Agentic_RAG_Module` 的分层（本模块照此对齐，便于三个 module 一起部署）：

```
agentic_rag/
  core/{config,types,registry,errors}.py   # 配置、pydantic 类型、后端懒加载注册表
  embeddings/ stores/ retrieval/ ingest/   # 各后端实现（正交轴）
  engine.py                                # 门面：一个类收全部能力
  server/app.py                            # FastAPI，端点与门面一一对应
  cli.py  gui/                             # typer CLI
configs/config.yaml  docs/  tests/  scripts/client.py
```

值得抄的三条：

- **注册表懒加载**：`import` 包永远不拉起 torch，只有选中的后端才 import 重依赖。
- **配置单一入口**：一切行为由 `configs/config.yaml` 决定，注释写在配置文件里。
- **相互依赖的路径统一相对一个根解析**（`models_dir`），避免「搬了一半」（DNA issue 006-A 的教训）。

RAG 的「两个正交后端轴」在 TTS 这里对应的是 **引擎轴（model backend）× 设备轴（cpu/cuda）**，
以及**功能轴（custom_voice / voice_design / voice_clone / vocal_events）**——
功能轴不是自由组合：**能力由引擎声明，不支持就明确拒绝**。

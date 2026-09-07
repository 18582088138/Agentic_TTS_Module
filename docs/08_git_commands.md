# 08 · Git 提交指令汇总（人工执行）

仓库 `Agent_TTS_Module` 当前在 `master` 分支上、**还没有任何提交**。
下面 18 步按「功能/职责」分组，每步一条 commit，直接复制执行即可。

> **一条说明（不粉饰历史）**：开发与返修过程中发现的四个问题（`docs/issues/001`–`004`）
> 改的都是已有文件，而工作区里已经是**修好之后**的最终内容 ——
> 所以这里不单列 "fix" commit（那样的 commit 会没有 diff）。每个问题的现象、判定、
> 修法与回归用例都记在 `docs/issues/` 里：001/002 随第 11 步提交，
> 003/004 在第 17 步单独提交（它们是首轮交付之后按人工反馈修的）。
> 同理，`gui/` 那一步提交的是**改版之后**的界面，提交信息按最终形态写。

```bash
cd c:/Users/test/Downloads/xkd/Agent_TTS_Module
```

---

### 1 · 调研与方案

```bash
git add docs/00_research.md docs/01_design.md docs/02_dev_plan.md
git commit -m "docs: Qwen3-TTS 调研结论、模块方案与开发计划

- 00 调研：上游三 API 各绑一个 checkpoint；10 条可复用的实测结论
  （失控不报错、文本预处理收益、instruct 规格差异、显存占用）；
  核实开源权重无原生 vocal event token（tokenizer 全表 + 全仓库 grep）
- 01 方案：引擎只负责「一段文本→一段波形」，文本链路/兜底/拼接/轮动排序
  全部放门面，新增引擎只需一个方法 + 一张能力表
- 02 计划：七阶段，S1-S6 只有两步需要真实权重"
```

### 2 · 工程骨架与依赖

```bash
git add pyproject.toml .gitignore
git commit -m "chore: 工程骨架、依赖声明与产物忽略

- torch 只给下界不钉版本：开发机 +cpu、4060 要 cu124，钉死会让 pip 换掉 CUDA 版
- pytest 默认 addopts=-m 'not real'：真实模型用例每条几十秒，不进日常套件
- 音频产物一律 gitignore"
```

### 3 · 核心层

```bash
git add agentic_tts/__init__.py agentic_tts/core/
git commit -m "feat(core): 配置、类型、后端注册表、异常与日志

- Config：单一 YAML + 少量环境变量覆盖；checkpoint 统一相对 models_dir 解析
- ensure_qwen_tts_importable：只往 sys.path 插一条并校验实际导入的是哪一份
  （环境里那份 editable 安装指向已删除目录，靠 cwd 碰运气会静默导错副本）
- types：Mode/Capability/SplitRule + 请求自校验 + infer_mode 单点推断
- registry：懒加载，import 包与查能力表都不拉起 torch（GUI 要秒开）
- errors 分四类，对应「改参数/换引擎/看环境/可重试」四种处置"
```

### 4 · 引擎抽象与离线引擎

```bash
git add agentic_tts/engines/__init__.py agentic_tts/engines/base.py \
        agentic_tts/engines/sine.py agentic_tts/engines/placeholders.py
git commit -m "feat(engines): 引擎抽象、正弦波离线引擎与未接入引擎占位

- TTSEngine 只有一个必实现方法 synthesize(chunk) + 静态能力声明
- require()/require_mode()：能力不足直接报错，绝不静默降级
- sine：零重依赖的确定性占位引擎，让门面/文本/服务/GUI 四层测试秒级离线
- breeze2/index2：只声明能力表（未实测）并在实例化时报「本阶段未接入」，
  这样 GUI 能提前置灰、配置写错能得到一句能照着做的话"
```

### 5 · vendored 上游源码

```bash
git add third_party/
git commit -m "vendor: 引入 Qwen3-TTS 上游源码副本（commit 022e286）

不引用别的项目目录、也不依赖环境里的 pip 安装：
- 环境里 qwen-tts 0.0.4 的 editable 安装 finder 指向已删除目录
- 磁盘上另有 commit 不同的副本，靠 cwd 会静默导入错的那一份
配置默认 repo_dir 为空即指向这份副本，doctor 会打印实际导入路径"
```

### 6 · Qwen3-TTS 引擎（含 8 GB 轮动加载）

```bash
git add agentic_tts/engines/qwen3.py
git commit -m "feat(engines): Qwen3-TTS torch 引擎与 8GB 显存轮动加载

- 三模式路由到三份 checkpoint（上游门禁硬写，换模式即换权重）
- LRU 模型池 max_resident=1：先卸再载 + 显式 empty_cache()
  （实测 1.7B bf16 跑起来约 5GB，8GB 卡装得下一个装不下两个；仅 del 不还显存）
- 加载前预检显存，报错直接给出该改哪一项，不必白等几十秒才 OOM
- effective_gen 调模型自己的 _merge_generate_kwargs 取「实际生效」参数
- instruct 按 checkpoint 规格判定是否真的生效并如实记录（0.6B 会被静默丢弃）
- ref_audio 拒绝 URL：上游加载器不读代理配置，公司网络下会挂住
- device/dtype 的 auto 解析：cpu→float32、cuda→bfloat16"
```

### 7 · 文本预处理与分段（移植）

```bash
git add agentic_tts/text/__init__.py agentic_tts/text/ascii_noise.py \
        agentic_tts/text/lexicon.py agentic_tts/text/numbers.py agentic_tts/text/segment.py
git commit -m "feat(text): 移植文本预处理与长文本分段（源自 verify_qwen3_tts 实测）

- 纯函数、零外部依赖；prepare() 固定正确顺序（Markdown→URL→数字→缩写）
- 实测收益极不均匀：URL 不处理 CER 4.43（模型逐字母念网址），Markdown 0.83→0.17，
  小数/emoji 做了白做；多音字与专名词典会换字，默认关闭
- segment()：短句合并、超长句在逗号处再切、开头短标题向后并
- options_from_config 单点映射配置到开关"
```

### 8 · 事件标记与拆条规则

```bash
git add agentic_tts/text/events.py agentic_tts/text/split.py
git commit -m "feat(text): vocal event 标记解析与四种拆条规则

- events：[pause:400ms] 切段（确定性）；其余事件在无原生支持时转 instruct 提示
  并往 warnings 写明「效果不保证」；有原生支持时原样交给引擎
- 支持中文别名（[停顿]/[笑]）；不认识的方括号原样保留，像标记的会提示
- split：chars/punct/role/blank 四种规则，与模型级分段共用同一套实现
- 角色规则顺带把角色映射成音色，并能报出「哪些角色没配音色」
  （沉默地全用默认音色 = 所有人听起来是同一个人且没有报错）"
```

### 9 · 音频层与失控兜底

```bash
git add agentic_tts/audio/
git commit -m "feat(audio): 拼接/重采样/落盘、失控兜底与字幕导出

- join：段间插确定长度静音、结尾不拖静音、跨引擎统一采样率
- resample 在缺 librosa 时退回线性插值（少一个可选依赖不该让拼接失败）
- guard：主判据「时长撞上限」（实测 41 秒噪声的波形指标一条不报、
  ASR 只回读出「嗯 k」，只有这条测得出来）；辅判据按字数估时长；
  处置是换种子重试而非关采样（贪心解码 12 条里 7 条失控，采样默认值 0 条）
- next_seed 用质数偏移，避免重试种子撞上下一段的段种子
- subtitle：一段音频一条字幕，SRT（剪映/Premiere/DaVinci 可直接导入，不必退化成 txt）；
  时间轴含段间静音（漏算则越往后字幕越提前）；UTF-8 带 BOM（否则剪映读中文乱码）"
```

### 10 · 音色档案

```bash
git add agentic_tts/voices/ voices/voices.yaml voices/refs/.gitkeep
git commit -m "feat(voices): 音色档案与角色映射

- 一个名字统一「内置音色 / 音色设计 / 参考音频克隆」，调用方不必知道背后是哪套 API
- 请求显式给的字段优先，其余用档案填（「用这个音色但换个 instruct」自然可写）
- ref_audio 相对 voices/ 解析（相互依赖的路径统一相对一个根）
- 音色名拼错报错并列出可选，绝不静默回落到默认音色"
```

### 11 · 门面与开发中发现的问题

```bash
git add agentic_tts/engine.py configs/config.yaml \
        docs/issues/001-event-markup-eaten-by-text-prepare.md \
        docs/issues/002-sine-hook-consumed-by-earlier-calls.md
git commit -m "feat(engine): 门面 TTSModule 串起全链路，并记录两个已修问题

- 流水线：事件标记 → 文本预处理 → 分段 → 逐块合成（带失控重试）→ 拼接落盘
- 段种子固定为 base+段序号：单段重生成不影响其他段（GUI 逐句重做的前提）
- 多段混用模式时按 checkpoint 分组执行：轮动次数从段数压到模式数
- 失败段不混进合并音频，warnings 里点名；每段单独落盘 + merged.wav + manifest.json
- 结果里 elapsed（墙钟耗时）与 seconds（音频时长）分开记，展示层不必自己乘 RTF
- 合并时按拼接时间轴写 merged.srt（可用 output.subtitles 关掉）
- 自动命名的产物目录保证唯一（mkdir exist_ok=False 逐个试 -2/-3），
  显式 name 仍复用目录：秒级时间戳 + 可重复的段文件名曾导致静默覆盖
- merge_audio()：把**已有**音频拼成一条，不重跑模型（重跑会得到不一样的音频）
- new_run_name()：给界面一个未占用的会话目录名，一次会话的产物落在同一处
- synthesize_many(progress=…)：把「跑到第几段」暴露给界面（回调在工作线程里）
- issues/001：文本预处理会把 [pause:500ms] 改写成中文导致停顿静默消失，
  已改为「先解析标记再清洗内容」，并加回归用例
- issues/002：sine 的失控钩子按调用次数计数会被无关调用消耗，改为按文本计数"
```

### 12 · CLI

```bash
git add agentic_tts/cli.py
git commit -m "feat(cli): doctor/engines/voices/speak/design/clone/split/batch/serve/gui

- 结果表把「音频时长」与「耗时」分成两列：只给一个数字会被读成「这次跑了多久」，
  而 CPU 上 RTF≈13，两者差一个数量级
- doctor 会打印**实际导入的 qwen_tts 路径**与显存预估，缺 checkpoint 直接给下载命令
- 所有命令支持 --engine sine，可在没有权重的机器上跑通
- CLI 不含业务逻辑，一条命令对应门面一个方法；错误统一收敛成一行人话 + 退出码 1"
```

### 13 · HTTP 服务

```bash
git add agentic_tts/server/
git commit -m "feat(server): FastAPI 服务与有含义的状态码

- 端点与门面方法一一对应；synthesize 可回 JSON/base64/audio-wav
- 异常映射：400 改参数、409 换引擎、422 请求或合成失败、503 服务端环境
- 单进程只驻留一个 TTSModule 且合成串行（一把锁）：
  8GB 卡上并发两条会同时要两份权重，必然 OOM 且报错看不出是并发导致
- 产物下载路径解析后再校验，挡住 ..%2f 目录穿越"
```

### 14 · GUI

```bash
git add agentic_tts/gui/
git commit -m "feat(gui): NiceGUI 深色控制台界面，单段试听 + 多段流水线

- theme.py：深底/细线网格/等宽数字/单一强调色，与 DNA 工作台同一路数；
  状态从不只靠颜色，每种状态带形状符号（○ ▶ ● ◐ ▲）；字体不走 CDN（代理会拦）
- 运行状态：顶栏状态灯带秒表，多段时报「第 N 段（共 M 段）」，
  正在合成的那张卡高亮；进度经门面 progress 回调 + ui.timer 轮询（不跨线程碰 UI）
- 多段面板：顶部统一声音配置（含克隆/设计）默认覆盖所有段，
  某段可展开「单独配置」覆盖；与单段面板共用同一份控件实现
- 一整段文本按字数/标点/角色/空行一键拆条；可单段生成、换种子重掷、删除
- 一次会话的产物全部写在同一个 gui-<时间戳>/ 目录，便于查找；旁边一个图标打开该目录
  （打开的是跑 GUI 那台机器上的文件管理器，远程访问时无效，所以路径也一直显示着）
- 底部一个「导出字幕 SRT」开关（默认取自 output.subtitles），合并时顺带写 merged.srt
- 状态与段落信息都把「耗时」与「音频时长」分开写
- 合并成整段优先**复用已生成音频**，只补跑未生成或文本已改的段
- 控件按模式显隐：参考音频只在克隆模式出现，上传框压成一条虚线细带
- 引擎不支持的模式在下拉里标注（能力表静态可查，不必加载权重）
- 合成扔 run.io_bound + 全局锁串行，并拒绝重入（连点会撞产物目录且双份权重会 OOM）
- 上传事件兼容 NiceGUI 3.x（event.file/async read）与 2.x（event.name/content）
- UI 建在 @ui.page('/') 内（顶层建会让入口脚本被重跑并 500）"
```

### 15 · 测试

```bash
git add tests/
git commit -m "test: 离线单测 126 条 + 真实模型用例 5 条（默认不跑）

- 离线全部走 sine 与纯函数，3 秒跑完，不加载任何权重
- 钉住几条容易被改回去的行为：标记先于预处理、[pause:1s] 时长 +1.00s、
  撞上限判失控、失败段不入合并、段种子稳定、混模式按 checkpoint 分组、
  音色拼错报错、qwen3 不声明 vocal_events、产物下载防目录穿越
- 连续两次自动命名的生成必须落在不同目录（静默覆盖的回归用例）
- merge_audio 期间引擎调用次数必须为 0（合并不许重跑模型）
- progress 回调必须逐段上报 start/done（界面的进度提示依赖它）
- 字幕：时间轴含段间静音、事件标记不进字幕、可关、SRT 带 BOM
- elapsed（耗时）与 seconds（音频时长）必须是两个字段且都有值
- test_qwen3_real.py 标 real，module 级夹具复用权重（加载几十秒）"
```

### 16 · 使用文档与样例

```bash
git add README.md docs/03_unit_tests.md docs/04_api_reference.md \
        docs/05_deployment.md docs/06_add_engine.md docs/07_gui_guide.md \
        docs/09_dev_log.md samples/ scripts/client.py
git commit -m "docs: 单测/接口/部署/接引擎/GUI 说明、样例与零依赖客户端

- 05 部署含本机 CPU 实测数据（加载 4.3s、10 字短句 RTF 12.9）与 4060 8GB 配置要点
- 06 接引擎：一个方法 + 一张能力表，附五条硬约束
- scripts/client.py 只用标准库，可直接拷进调用方项目"
```

### 17 · 返修阶段发现的两个问题

```bash
git add docs/issues/003-guard-heuristic-vs-slow-speech-instruct.md \
        docs/issues/004-second-run-silently-overwrote-the-first.md
git commit -m "docs(issues): 记录失控启发式与产物静默覆盖两个问题

- 003：辅助判据（字数×3 倍）与「语速偏慢」类 instruct 会互相踩，
  样例里那次拦对了但判据偏紧；默认值不动（宁可多跑一次也别放过真失控），
  可用 guard.duration_ratio / chars_per_sec 调，待 4060 上扫一批数据再定
- 004：秒级时间戳 + 可重复的段文件名 ⇒ 同一秒内第二次生成静默覆盖第一次，
  表现为「点了生成却没有新文件」。修法与回归用例见文档
  （产物目录唯一化在第 11 步、GUI 拒绝重入在第 14 步）"
```

### 18 · 提交指令自身

```bash
git add docs/08_git_commands.md
git commit -m "docs: git 分步提交指令汇总"
```

---

## 核对

```bash
git log --oneline            # 应为 20 条（含步骤 19、20）
git status --short           # 应为空（outputs/ 已忽略）
```

⚠️ 不要 `git add -A` 一把梭：`outputs/` 已忽略，但 `voices/refs/` 下可能有你自己放的
参考音频（人声素材），那是**不该进仓库**的东西。
---

## 步骤 19/19 · 交接单：把稿子交给 GUI 精修（2026-09-08）

下游（DailyNewsAssistant）要能把稿子推进 TTS 的多段界面，人在那边精修完，
产物清单再回传给它。**只加了一个模块、两个端点、GUI 的导入与回写**，
合成链路一行未动。

```bash
cd /c/Users/test/Downloads/xkd/Agent_TTS_Module

git add agentic_tts/core/handoff.py
git add agentic_tts/server/app.py
git add agentic_tts/gui/app.py
git add tests/test_handoff.py
git add docs/03_unit_tests.md docs/04_api_reference.md docs/07_gui_guide.md
git add docs/08_git_commands.md

git commit -m "feat(handoff): 交接单——外部应用把稿子推进多段界面，产物清单回传

- core/handoff.py：outputs/handoff/<token>.json，落成文件而不是放内存，
  因为 cli serve 与 cli gui 是两个进程，内存里的字典彼此看不见
- 由服务端落盘：调用方可能在另一台机器上，它只能发 HTTP
- token 出现在 URL 里并参与拼路径，字符集固定为 [A-Za-z0-9_-]{6,64}，
  非法 token 一律当作不存在——不校验的话 ../../ 就能读写输出目录之外的文件
- POST /gui/handoff 回 token 与现成的 gui_url；GET /gui/handoff/{token} 供轮询
- GUI 认 ?import=<token>：切到多段页签、逐段填文本、填统一声音配置；
  每次生成或合并完把 status=done 与产物清单写回同一条记录
- 找不到交接单只提示、不报错——不该因为一个 URL 参数让页面起不来
- 调用方轮询，这边不回调：TTS 不该知道调用方的地址
- 离线单测 126 → 134 条"

git status --short          # 预期：空（outputs/handoff/ 在 outputs 下，已忽略）
```

核对：`git log --oneline` 应为 19 条。

---

## 步骤 20/20 · 界面挂进服务，交接单回传进度与返回（2026-09-08）

下游实测：服务与界面各起一个进程，**各加载一份权重**，8 GB 卡顶满，
而且每次点「高级配置」都要等第二个进程冷启动。

```bash
cd /c/Users/test/Downloads/xkd/Agent_TTS_Module

git add agentic_tts/engine.py
git add agentic_tts/gui/app.py
git add agentic_tts/server/app.py
git add agentic_tts/cli.py
git add docs/04_api_reference.md docs/07_gui_guide.md docs/08_git_commands.md

git commit -m "feat(server): 图形界面挂进 HTTP 服务，一个进程一份权重

- gui/app.py 拆成 build()/run()/mount()：mount() 用 ui.run_with 挂到 /gui，
  与服务共用同一个 TTSModule。cli serve 默认挂（--no-gui 可关）
- 锁移到 TTSModule.lock：界面与 HTTP 请求必须**共用**一把锁，
  各拿一把会在「界面点生成 + 一条 HTTP 请求」时同时要两份权重
- 挂载后静态产物路由带前缀，media_url 跟着拼 /gui/outputs/...，否则播放器 404
- cli gui 保留为独立进程，但显式提示它会另加载一份权重
- 交接单增加 return_url / done / total：
  · 逐段生成只写 running + done/total（调用方据此显示进度条）
  · 「全部生成」「合并」才写 done + 产物清单，避免调用方收走半成品
  · 完成后按 return_url 关掉本标签页／跳回调用方，另留「回传并返回」按钮
- handoff 的 gui_url 改成相对路径：写死 host/port 在「服务绑 0.0.0.0、
  从局域网访问」时会给出一个打不开的地址"

git status --short          # 预期：空
```

核对：`git log --oneline` 应为 20 条。


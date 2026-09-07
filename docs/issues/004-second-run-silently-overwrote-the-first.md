# 004 · 第二次生成没有新文件（同一秒的产物目录互相覆盖）

**状态**：已修复（`engine.py::_output_dir` + `gui/app.py` 防重复点击）
**发现**：2026-09-07，GUI 连续点两次「生成」
**影响范围**：所有**自动命名**（未显式给 `name`）的生成；库 / CLI / HTTP / GUI 都中

## 现象

GUI 里点第二次「生成」，`outputs/` 下没有出现新文件。

## 复现（三行）

```python
m = TTSModule(); m.config.engine.backend = "sine"
for i in (1, 2, 3):
    print(m.synthesize(f"第{i}次", voice="test_female").path)
```

```
outputs\20260907-221129\seg_001_test_female.wav
outputs\20260907-221129\seg_001_test_female.wav      ← 同一个路径
outputs\20260907-221129\seg_001_test_female.wav      ← 同一个路径
```

磁盘上只剩最后一次的音频，前两次**被静默覆盖**。

## 根因（两个条件叠在一起）

1. 产物目录是 `time.strftime("%Y%m%d-%H%M%S")` —— **只有秒级精度**；
2. 段文件名只由「段序号 + 音色」决定（`seg_001_test_female.wav`），跨运行必然重名。

于是同一秒内的两次生成落进同一个目录、写同一个文件名。`sine` 引擎毫秒级返回时必然撞；
真实引擎一条要几十秒，本来撞不上 —— **但 GUI 的生成按钮在合成期间没有禁用**，
连点两次会排队跑两遍，两遍都在最初那一秒建目录，于是照样互相覆盖。

（`pytest_real` 目录里 `test_custom_voice` 与 `test_voice_clone_x_vector` 互相覆盖
`seg_001*.wav` 也是同一个原因，只是那里是显式 `name`，属于预期行为。）

## 修法

**① 自动命名保证唯一**（`engine.py::_output_dir`）：撞了就加 `-2`、`-3`…
用 `mkdir(parents=True)` 且 **`exist_ok=False`**，让文件系统本身做互斥 ——
多进程同时跑也不会撞（`exist_ok=True` 会让两个进程都以为自己拿到了目录）。

显式给了 `name` 则**照用不改**：`-n pytest_real` 这类是明确要求写到固定目录，
覆盖是调用方自己的选择，不该被自动改名。

**② GUI 拒绝重入**（`gui/app.py::make_busy_guard`）：合成期间按钮禁用，
第二次点击直接被拒并提示，而**不是排队**。排队是错的处置：同一张 8 GB 卡上
两条请求会同时要两份权重（必然 OOM），且用户等的是这一条的结果，不是两条。

## 回归用例

- `tests/test_module.py::test_repeated_auto_named_runs_do_not_overwrite_each_other`
  —— 连续三次自动命名，三个路径互不相同且文件都在；
- `tests/test_module.py::test_explicit_name_still_reuses_the_directory`
  —— 显式 `name` 仍复用目录（保证修法①没有顺手改掉这个语义）。

## 教训

「文件名可重复 + 时间戳精度不够」是一对**互相掩护**的缺陷：单看任何一个都像没问题，
合起来就是静默数据丢失。**产物路径的唯一性要由创建时的原子操作保证，
不能靠"时间戳大概不会撞"。**

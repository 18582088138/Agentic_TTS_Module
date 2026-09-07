# 02 · 开发计划 / Development plan

七个阶段，**每阶段＝可跑的代码 + 只测本阶段的单元测试 + 一组 git commit**。
测试文件头写完整复测命令。默认 `pytest` 只跑离线测试；真实模型合成标 `@pytest.mark.real`，
**手动触发**（一次 CPU 合成几十秒起，不进默认套件）。

环境：`conda ov_env_py312`（依赖已齐，本阶段无需 pip install）
Python：`C:/Users/test/miniforge3/envs/ov_env_py312/python.exe`

| 阶段 | 内容 | 产出与测试 | 依赖模型 |
|---|---|---|---|
| **S1 骨架** | `core/{errors,logging,types,config,registry}` + `engines/base` + `engines/sine` + `cli doctor` | `tests/test_config.py`、`test_types_mode.py`（mode 推断）、`test_sine_engine.py` —— 全离线秒级 | ❌ |
| **S2 Qwen3 引擎** | `engines/qwen3.py`（三 checkpoint 路由、ModelPool、显存预检、effective_gen）+ `audio/io.py` + `audio/guard.py` | `test_guard.py`（离线，构造波形）、`test_qwen3_engine.py::test_custom_voice_short`（`real` 标记，一条短句） | ✅ CustomVoice |
| **S3 三大功能** | voice_design / voice_clone（ICL + x-vector）+ `voices/store.py` 音色档案 | `test_voices_store.py`（离线）、`test_qwen3_features.py`（`real`，design/clone 各一条） | ✅ 三个 |
| **S4 文本链路** | `text/{normalize,segment,events}` —— 从 vq3 textkit 精简移植 + 事件标记解析 | `test_normalize.py`、`test_segment.py`、`test_events.py` —— 纯函数断言，全离线 | ❌ |
| **S5 门面** | `engine.py::TTSModule` 串起 prepare→segment→合成→guard→拼接落盘 + `cli.py` 全命令 | `test_module_e2e.py`（sine 引擎跑通全链路，含停顿与失控重试）、`test_cli.py` | ❌ |
| **S6 服务与界面** | `server/app.py`（FastAPI）+ `gui/app.py`（NiceGUI）+ `scripts/client.py` | `test_server.py`（TestClient + sine，离线） | ❌ |
| **S7 收尾** | `docs/03_unit_tests.md`、`04_api_reference.md`、`05_deployment.md`、`README.md`、项目专属 skill + memory、`docs/08_git_commands.md` | 无新代码 | ❌ |

**S1→S6 之间只在 S2/S3 需要真实模型**，其余全部离线可测 —— 这是把成本压住的关键：
`sine` 引擎让门面、文本链路、服务、界面四层完全不碰 torch。

## 测试策略（成本纪律）

```bash
# 默认：只跑离线，秒级
python -m pytest tests -q
# 真实模型（手动，CPU 上每条几十秒）
python -m pytest tests -q -m real
```

- 每个功能只测自己那一层，**不做全量回归** —— 全功能测试与整体 review 由人工阶段性触发。
- 真实合成的用例一律用**短句**（≤20 字）且只跑 1 条。
- 音频产物落 `outputs/<时间戳>/`，`.gitignore` 掉。

## 风险与预案

| 风险 | 预案 |
|---|---|
| 本机 torch 是 `+cpu`，`cuda` 路径无法实测 | 设备解析与显存预检写成纯函数单测；`cuda` 分支在 4060 机器上验（`docs/05_deployment.md` 留复测命令） |
| 1.7B 在 CPU 上合成慢 | 只在 `real` 测试里跑短句；RTF 记进结果，慢是预期不是 bug |
| `qwen_tts` 导入到错误的源码副本 | `repo_dir` 插 `sys.path[0]` + `doctor` 打印 `qwen_tts.__file__` |
| Breeze/IndexTTS-2 的真实接口未知 | 只在注册表留名，实例化即报「本阶段未实现」；引擎抽象保持只有一个必实现方法，接入成本最低 |

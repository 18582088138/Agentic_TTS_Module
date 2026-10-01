# 006 · MiniMax 账号里的克隆音色选不到（VoiceStore 不认这个名字）

**状态**：已修复（`engines/minimax.py::builtin_voices`）
**发现**：2026-10-01，想在 DNA 里用已有克隆音色 `va0d7850d_790636` 当内置音色
**影响范围**：`api.provider: minimax` 时，账号里克隆 / 设计出的所有音色

## 现象

克隆出的音色在 MiniMax 侧就是普通 voice_id，本该走 `custom_voice` 直接用。但：
- `GET /voices` 只有 4 个写死的系统音色，DNA 设置页下拉框里没有它；
- 手填 `voice="va0d7850d_790636"` → `VoiceStore.resolve` 抛
  `ConfigError: 没有名为 'va0d7850d_790636' 的音色`。

## 判定

`MiniMaxEngine.builtin_voices()` 是静态列表；VoiceStore 只认档案 + 内置，没有「开放音色名」的透传。
不加透传是对的（「不静默降级」：拼错的名字必须报错），缺的是**把账号音色登记进内置**。

## 修法

`builtin_voices()` = 静态系统音色 + `POST /v1/get_voice {"voice_type":"all"}` 里的
`voice_cloning` / `voice_generation`。查询失败只记警告并退回静态列表，服务照常起。

已知限制：VoiceStore 在服务进程里只建一次 —— **账号里新增的音色要重启 TTS 服务才出现**。

## 回归用例

`tests/test_minimax_voices.py`（假 httpx，零网络）：
- `test_account_cloned_voice_is_selectable`
- `test_listing_failure_falls_back_to_static_voices`

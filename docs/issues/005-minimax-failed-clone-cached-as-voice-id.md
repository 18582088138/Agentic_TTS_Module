# 005 · MiniMax 克隆失败被当成功缓存，之后每段都 2054 voice id not exist

**状态**：代码已修复（`engines/minimax.py`）；账号侧问题待人工处理（见「根因」）
**发现**：2026-10-01，DailyNewsAssistant 短视频音频「4 段全部合成失败」
**影响范围**：`api.provider: minimax` + `voice_clone` 模式的全部调用

## 现象

```
TTS 服务返回 422：MiniMax 业务错（https://api.minimaxi.com）：2054 voice id not exist
```

`voices/api_voice_cache.json` 里有一条当天写入的 `v958152d6_840691`，
但 `POST /v1/get_voice {"voice_type":"voice_cloning"}` 列出的账号音色里没有它。

## 根因（两层）

1. **代码**：`_do_clone` 只看 HTTP 码（`raise_for_status`）。MiniMax 的业务错是
   **HTTP 200 + `base_resp.status_code` 非 0**，于是失败的克隆被当成功，
   一个从未创建的 voice_id 写进了缓存；缓存无失效机制 → 之后每次都 2054，永不自愈。
2. **账号**：真实的克隆错误是
   `2061 your current token plan not support model, voice_clone`。
   实测 speech-2.8-turbo / 2.8-hd / 2.6-turbo / 2.6-hd / 02-turbo / 02-hd **全部** 2061 ——
   当前 Token Plan（`sk-cp-` key）不含克隆权限，换模型解决不了。

## 修法

- 上传与克隆的响应都过 `_checked()`：HTTP 码与 `base_resp` 都过关才算成功，
  否则抛带 MiniMax 原始错误码的 `SynthError`，**不写缓存**。
- t2a 回 2054 且 voice_id 来自克隆缓存 → 作废该条缓存、重克隆一次再合成
  （也覆盖 MiniMax 回收久未使用的临时音色这一情形）。
- voice_id 后缀从「秒级时间戳」改为随机数：重克隆发生在同一秒内，时间戳会撞出同一个 id。

修后同样的请求会直接报出 2061 的真实原因，而不是误导性的 2054。

## 回归用例

`tests/test_minimax_clone.py`（假 httpx，零网络）：
- `test_failed_clone_raises_and_is_not_cached`
- `test_stale_cached_voice_is_recloned_once`

# 真 API Demo（本地跑）

## 静态 UI 部署版（看 UI）
访问：https://rwo3qn6hp9pcz.space.mcode.cn
- UI 完整可用（mobile + desktop）
- 合成按钮按下后会 404 —— **没后端**

## 本地跑（完整功能）

### 1. 装依赖
```bash
pip install fastapi uvicorn httpx pydantic soundfile numpy
```

### 2. 设环境变量
```bash
export MINIMAX_API_KEY=你的key
# 可选：
# export MINIMAX_MODEL=speech-2.8-hd   # 高质量
# export MINIMAX_ENDPOINT=https://api.minimax.chat  # 默认
```

### 3. 启动
```bash
cd demo/api
python server.py
# 默认监听 0.0.0.0:8310
```

### 4. 浏览器访问
```
http://127.0.0.1:8310
```

## 端点
- `GET /`           UI 主页
- `GET /api/info`   后端配置（key masked）
- `POST /api/synthesize` 调真 MiniMax 合成

## 支持的合成模式
- **custom_voice**: 系统 voice_id（male-qn-qingse 等）
- **voice_design**: instruct 描述音色
- **voice_clone**: ref_audio base64 + ref_text，自动 voice_id 缓存

## 事件标记（自动翻译成 MiniMax 2.8 原生）
- `[laugh]` → `(laughs)`
- `[sigh]` → `(sighs)`
- `[breath]` → `(breath)`
- `[pause:400ms]` → `<#0.40#>`
- `[pause]` → `<#0.3#>`

## 已知限制
- 沙箱不能跑持续 Python 服务（只有静态部署）
- 因此真合成必须在你本地跑
- UI 已部署到公网可以直接看界面布局
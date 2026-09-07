"""
界面主题 / GUI theme —— 深色控制台风，与 DailyNewsAssistant 工作台同一路数。

**这是一块要盯着看的操作台**（几十段文案、逐段试听重做），不是落地页。所以走仪表盘/
终端那一路：深底、细线网格、等宽数字、一个高对比强调色。
A dark console: dark ground, hairline grid, tabular figures, one high-contrast accent.

颜色是有语义的 / The palette carries meaning:
    青绿 = 已生成        琥珀 = 正在合成        红 = 失败        灰 = 还没做
    **绝不能只靠颜色区分**：每种状态同时带一个形状符号（● / ▶ / ▲ / ○），
    色觉障碍者与黑白截图下都要能分辨。
    Never colour alone: every state carries a glyph too.

字体不走 CDN：公司代理会把 Google Fonts 拦下来，页面要么裸奔要么卡住。
只用 Windows 上一定存在的字体栈。
Fonts are never fetched; the corporate proxy blocks Google Fonts.
"""

from __future__ import annotations

from nicegui import ui

# 状态 → (符号, CSS 类, 文案)。符号在前，颜色只是加强。
STATES = {
    "idle": ("○", "tt-idle", "待生成"),
    "running": ("▶", "tt-run", "合成中"),
    "done": ("●", "tt-ok", "已生成"),
    "stale": ("◐", "tt-warn", "文本已改"),
    "failed": ("▲", "tt-fail", "失败"),
}

_CSS = """
:root {
  --tt-bg:      #0a0e14;
  --tt-panel:   #111823;
  --tt-panel-2: #161f2c;
  --tt-line:        rgba(125, 165, 205, 0.14);
  --tt-line-strong: rgba(125, 165, 205, 0.32);
  --tt-text:  #c8d6e5;
  --tt-dim:   #7b8ea4;
  --tt-faint: #55677a;
  --tt-accent: #3ddc97;   /* 已生成 */
  --tt-warn:   #f2a33c;   /* 正在合成 / 需要注意 */
  --tt-danger: #ff6b6b;   /* 失败 */
  --tt-info:   #22d3ee;   /* 强调信息 */
  --tt-mono: ui-monospace, "Cascadia Mono", "JetBrains Mono", Consolas, monospace;
  --tt-sans: "Microsoft YaHei", "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
}

body, .nicegui-content {
  background: var(--tt-bg); color: var(--tt-text); font-family: var(--tt-sans);
}

/* 极淡网格底纹：给深色背景一点纵深，又不抢内容 */
body::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(125,165,205,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(125,165,205,.035) 1px, transparent 1px);
  background-size: 44px 44px;
}
.nicegui-content { position: relative; z-index: 1; }

/* ---- 顶栏 ---- */
.tt-header {
  background: linear-gradient(180deg, #0f1622 0%, #0b111a 100%);
  border-bottom: 1px solid var(--tt-line-strong);
}
.tt-title { font-family: var(--tt-mono); letter-spacing: .08em; font-weight: 700; }
.tt-title .accent { color: var(--tt-accent); }
.tt-meta { font-family: var(--tt-mono); font-size: 11px; color: var(--tt-dim); }

/* ---- 状态胶囊：顶栏那颗「程序在不在跑」的指示灯 ----
   这是用户最常看的一个元素：合成一条要几十秒，没有它就只能盯着不动的界面猜。
   The single most-watched element: synthesis takes tens of seconds. */
.tt-status {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--tt-mono); font-size: 12px;
  padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--tt-line-strong); background: var(--tt-panel);
  white-space: nowrap;
}
.tt-status.is-run { border-color: var(--tt-warn); color: var(--tt-warn);
                    background: rgba(242,163,60,.10); }
.tt-status.is-ok  { border-color: var(--tt-accent); color: var(--tt-accent);
                    background: rgba(61,220,151,.10); }
.tt-status.is-err { border-color: var(--tt-danger); color: var(--tt-danger);
                    background: rgba(255,107,107,.10); }

/* ---- 卡片 ---- */
.tt-card {
  background: var(--tt-panel); border: 1px solid var(--tt-line);
  border-radius: 10px; box-shadow: none;
}
.tt-card-title {
  font-family: var(--tt-mono); font-size: 12px; font-weight: 700;
  letter-spacing: .10em; text-transform: uppercase; color: var(--tt-text);
}
.tt-hint { font-size: 11px; color: var(--tt-faint); }
.tt-mono { font-family: var(--tt-mono); font-variant-numeric: tabular-nums; }

/* ---- 段落卡：正在合成的那一段要**一眼看到** ---- */
.tt-seg { border-left: 2px solid transparent; transition: border-color .12s ease; }
.tt-seg.is-running {
  border-left-color: var(--tt-warn);
  background: rgba(242,163,60,.06);
  box-shadow: 0 0 0 1px rgba(242,163,60,.35);
}
.tt-seg.is-done   { border-left-color: var(--tt-accent); }
.tt-seg.is-failed { border-left-color: var(--tt-danger); }

/* ---- 状态标记 ---- */
.tt-chip {
  display: inline-flex; align-items: center; gap: 4px;
  font-family: var(--tt-mono); font-size: 11px; line-height: 1;
  padding: 3px 7px; border-radius: 4px; border: 1px solid var(--tt-line-strong);
}
.tt-idle { color: var(--tt-faint); }
.tt-run  { color: var(--tt-warn);   border-color: var(--tt-warn); }
.tt-ok   { color: var(--tt-accent); border-color: var(--tt-accent); }
.tt-warn { color: var(--tt-warn);   border-color: var(--tt-warn); }
.tt-fail { color: var(--tt-danger); border-color: var(--tt-danger); }

/* ---- 表单控件：Quasar 默认是亮底，深色下必须压一遍 ---- */
.q-field__control { background: var(--tt-panel-2) !important; }
.q-field__native, .q-field__input, .q-field__label,
.q-field__prefix, .q-field__suffix { color: var(--tt-text) !important; }
.q-field--outlined .q-field__control:before { border-color: var(--tt-line) !important; }
.q-textarea .q-field__native { font-family: var(--tt-sans); line-height: 1.7; }
.q-checkbox__label, .q-toggle__label { color: var(--tt-text); font-size: 12px; }
.q-expansion-item__container { border-top: 1px solid var(--tt-line); }
.q-tab { font-family: var(--tt-mono); letter-spacing: .06em; }

/* ---- 紧凑上传框 ----
   Quasar 的 q-uploader 默认是一个带标题栏和文件列表的大盒子，占掉半屏还不好看。
   这里压成一条虚线细带：文件名我们自己在旁边显示，列表隐藏掉。
   Quasar's uploader is a tall panel with its own header and file list; it is squashed
   into a single dashed strip and the list is hidden (we render the name ourselves). */
.tt-upload .q-uploader {
  min-height: 0; max-height: 34px; width: 100%;
  background: transparent; box-shadow: none;
  border: 1px dashed var(--tt-line-strong); border-radius: 6px;
}
.tt-upload .q-uploader__header {
  min-height: 32px; padding: 0 8px; background: transparent;
  color: var(--tt-dim); border: none;
}
.tt-upload .q-uploader__header-content { padding: 0; min-height: 32px; align-items: center; }
.tt-upload .q-uploader__title { font-size: 11px; font-weight: 400; line-height: 1; }
.tt-upload .q-uploader__subtitle,
.tt-upload .q-uploader__list { display: none; }
.tt-upload .q-btn { font-size: 11px; }

/* ---- 播放器：默认那条太亮，压暗一点 ---- */
audio { width: 100%; height: 32px; filter: invert(.88) hue-rotate(180deg) saturate(.7); }

/* ---- 按钮 ---- */
.q-btn { font-family: var(--tt-mono); letter-spacing: .04em; }
.tt-btn-main { background: rgba(61,220,151,.14) !important; color: var(--tt-accent) !important;
               border: 1px solid var(--tt-accent) !important; }
.tt-btn-alt  { background: rgba(34,211,238,.12) !important; color: var(--tt-info) !important;
               border: 1px solid var(--tt-info) !important; }

@media (prefers-reduced-motion: reduce) {
  .tt-seg { transition: none; }
}
"""


def apply() -> None:
    """注入主题 / Inject the theme（在每个 page 里调一次）。"""
    ui.dark_mode().enable()
    ui.add_css(_CSS)


def chip(state: str) -> tuple[str, str]:
    """
    状态 → (显示文本, CSS 类) / State → label and class.

    文本里带符号，**颜色只是加强** —— 黑白截图和色觉障碍下也要读得出来。
    """
    glyph, css, label = STATES.get(state, STATES["idle"])
    return f"{glyph} {label}", f"tt-chip {css}"


__all__ = ["STATES", "apply", "chip"]

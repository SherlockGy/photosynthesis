# 时间线：全片网格 = 原创配乐的拍子（96 BPM，每小节 2.5 秒）
# 段落第一句对齐小节线：段落转折、音乐的重音、画面的层级切换落在同一格
import json
import math
from pathlib import Path

from script import LINES

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "assets" / "voice" / "durations.json").read_text(encoding="utf-8"))

FPS = 24
W, H = 1920, 1080
BPM = 96
BEAT = 60 / BPM
BAR = BEAT * 4
EIGHTH = BEAT / 2
VOICE_LAG = 0.10    # 小节线之后旁白再进来一点点，重音不和字头打架
LEAD = 0.6          # 片头：第 0 帧就有画面，旁白稍后进

# 需要从小节线开始的段落
SNAP = {"leaf", "chloro", "light", "dark", "sum", "earth", "end"}
# 段落之前额外空出的小节数：镜头需要时间穿过层级
EXTRA_BARS = {}
# 关键词对齐：把这句往后推一点，让这个词正好落在下一条小节线上（推迟不超过 MAX_ALIGN 秒）
ALIGN = {}
MAX_ALIGN = 1.2

PUNCT = "，。：；？！、"
TEXT = {lid: text for lid, text, _sec, _gap in LINES}


def _pauses(lid):
    """从配音里找停顿：10ms 能量低于阈值、持续 70ms 以上的区间（秒，相对句首）"""
    import numpy as np
    from scipy.io import wavfile
    sr, x = wavfile.read(ROOT / "assets" / "voice" / f"{lid}.wav")
    x = x.astype(np.float32) / 32768.0
    hop = int(0.01 * sr)
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    thr = max(0.008, np.percentile(rms, 60) * 0.08)
    quiet = rms < thr
    out = []
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) >= 7 and i > 5 and j < n - 5:
                out.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    return out


_ANCHORS = {}


def anchors(lid):
    """句中标点 ↔ 配音停顿的对应关系，返回 [(字位置, 相对句首的秒)]"""
    if lid in _ANCHORS:
        return _ANCHORS[lid]
    text = TEXT[lid]
    d = META[lid]["dur"]
    nchar = sum(1 for ch in text if ch not in PUNCT)
    pts = [(0.0, 0.0)]
    pauses = _pauses(lid)
    used = -1
    k = 0
    for i, ch in enumerate(text):
        if ch not in PUNCT:
            k += 1
            continue
        if i == len(text) - 1:
            break
        est = d * k / max(1, nchar)
        best = None
        for j, (p0, p1) in enumerate(pauses):
            if j <= used:
                continue
            mid = (p0 + p1) / 2
            if abs(mid - est) < 0.7 and (best is None or abs(mid - est) < abs((pauses[best][0] + pauses[best][1]) / 2 - est)):
                best = j
        if best is not None:
            used = best
            p0, p1 = pauses[best]
            pts.append((i, p0))
            pts.append((i + 1, p1))
    pts.append((float(len(text)), d))
    _ANCHORS[lid] = pts
    return pts


def word_off(lid, sub, frac=0.0):
    """词在句中开始的时间（相对句首）：在相邻的真实停顿之间按字数插值"""
    import numpy as np
    text = TEXT[lid]
    i = text.index(sub) + frac * len(sub)
    pts = anchors(lid)
    return float(np.interp(i, [p[0] for p in pts], [p[1] for p in pts]))


T = {}
SECTION = {}
_t = LEAD
_prev = None
for lid, text, sec, gap in LINES:
    d = META[lid]["dur"]
    start = _t
    if sec != _prev and sec in SNAP:
        bar_i = math.ceil((start - VOICE_LAG - 0.05) / BAR - 1e-9) + EXTRA_BARS.get(sec, 0)
        bar_t = bar_i * BAR
        SECTION[sec] = bar_t
        start = bar_t + VOICE_LAG
    if lid in ALIGN:
        off = word_off(lid, ALIGN[lid])
        target = math.ceil((start + off) / BAR - 1e-9) * BAR
        if target - (start + off) <= MAX_ALIGN:
            start = target - off
    if sec not in SECTION:
        SECTION[sec] = 0.0 if sec == "hook" else start
    T[lid] = (start, start + d)
    _t = start + d + gap
    _prev = sec

TOTAL = round(_t + 0.2, 3)
NFRAMES = int(TOTAL * FPS)


def s(lid):
    return T[lid][0]


def e(lid):
    return T[lid][1]


def wt(lid, sub, frac=0.0):
    """词开始的绝对时间"""
    return T[lid][0] + word_off(lid, sub, frac)


def snap_beat(t, tol=0.16):
    """吸附到音乐网格：离拍点 tol 秒以内就落在拍点上，否则落在最近的八分音符上"""
    b = round(t / BEAT) * BEAT
    if abs(b - t) <= tol:
        return b
    return round(t / EIGHTH) * EIGHTH


if __name__ == "__main__":
    for lid, (a, b) in T.items():
        print(f"{lid} {a:7.2f}-{b:7.2f} 小节 {a / BAR:6.2f}-{b / BAR:6.2f}  {TEXT[lid]}")
    print("段落", {k: round(v / BAR, 2) for k, v in SECTION.items()})
    for lid, w in ALIGN.items():
        print(f"对齐 {lid}「{w}」→ {wt(lid, w):.3f}s = 第 {wt(lid, w) / BAR:.3f} 小节")
    print("TOTAL", TOTAL, f"{int(TOTAL // 60)}:{TOTAL % 60:04.1f}", "小节", round(TOTAL / BAR, 1))

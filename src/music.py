# 配乐与混音：为《光合作用》原创作曲。每一个音都按 arrange.py 的编排表写，转折是“乐队一起停、一起进”
# 原则：
#   1. 明亮、好奇，不闷：全程不做低通；暗场段落是乐器变少、音区变低，高频的竖琴和钢琴还在
#   2. 节奏要有：鼓用底鼓、边击、拍手、沙锤，律动段每个八分音符都有东西往前推
#   3. 不用金属音色：没有钟、铃、钢片琴、钟琴、三角铁、镲、踩镲；没有任何音效
#   4. 人声第一：每句话说话时配乐压到比旁白低 TARGET_LD；没人说话的空档，配乐放回原来的音量
import math
import os
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import tinysoundfont as tsf
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, fftconvolve, sosfilt

import arrange as A
from timeline import BAR, BEAT, T, TOTAL

ROOT = Path(__file__).resolve().parent.parent
# 音源：GeneralUser GS（https://schristiancollins.com/generaluser.php，不随仓库分发）
# 找的顺序：环境变量 SOUNDFONT_SF2 → assets/sf/GeneralUser-GS.sf2 → 同级旧项目里的那份
SF2 = Path(os.environ.get("SOUNDFONT_SF2") or ROOT / "assets" / "sf" / "GeneralUser-GS.sf2")
if not SF2.exists() and (ROOT.parent / "sunk-cost" / "assets" / "sf" / "GeneralUser-GS.sf2").exists():
    SF2 = ROOT.parent / "sunk-cost" / "assets" / "sf" / "GeneralUser-GS.sf2"
VOICE = ROOT / "assets" / "voice"
OUT = ROOT / "out"
SR = 44100
N = int((TOTAL + 2.0) * SR)
LOG = "[music 配乐混音]"

# 混音目标（LUFS，按分轨整段测）
VOICE_LUFS = -16.0
MUSIC_LUFS = -17.0      # 配乐让位之前的整体电平（也就是没人说话时配乐的音量）
FINAL_LUFS = -14.0
CEILING = 0.74          # 限幅上限（约 -2.6 dBFS），给 AAC 编码留余量
# 说话时旁白比配乐至少高多少（LU，每句话整句测全频段响度）：用户在《间歇性强化》里选定的 10 LU
TARGET_LD = 10.0
# 各织体在 TARGET_LD 上再多让多少 LU：保住段落之间的起伏
TEXTURE_EXTRA_LD = {"L": 2.0, "M": 2.5, "G": 1.0, "C": 1.0, "O": 3.0}
MAX_DUCK_DB = 20.0
# 说话时配乐额外的频段倾斜（dB）：中频再让一点，字更清楚
SPEECH_EQ_DB = {"low": 0.0, "mid": -4.0, "high": -2.0}
DUCK_PRE, DUCK_POST, DUCK_BRIDGE = 0.10, 0.15, 0.9
DUCK_ATTACK_DBS, DUCK_RELEASE_DBS = 120.0, 25.0
HIT_PASS = 0.10
# 频段能量统计的分界（Hz）：只用来检查声音亮不亮，不做修正
BANDS = [120, 500, 2000, 6000]

# 声部：音源里的音色；room/hall = 两种混响的发送量；hp = 高通（Hz）
PARTS = {
    "kick": dict(prog=0, bank=128, drums=True, vol=1.1, pan=0.0, room=0.05, hall=0.0, hp=30),
    "stick": dict(prog=0, bank=128, drums=True, vol=0.55, pan=0.08, room=0.2, hall=0.08, hp=200),
    "clap": dict(prog=25, bank=128, drums=True, vol=0.5, pan=-0.05, room=0.2, hall=0.1, hp=250),
    "perc": dict(prog=0, bank=128, drums=True, vol=0.7, pan=0.25, room=0.14, hall=0.05, hp=300),
    "timp": dict(prog=47, vol=0.75, pan=0.0, room=0.1, hall=0.3, hp=35),
    "bass": dict(prog=32, vol=0.85, pan=0.0, room=0.0, hall=0.05, hp=35),
    "piano": dict(prog=0, vol=0.72, pan=-0.12, room=0.05, hall=0.25, hp=70),
    "lead": dict(prog=0, vol=0.78, pan=0.05, room=0.05, hall=0.3, hp=150),
    "flute": dict(prog=73, vol=0.42, pan=0.18, room=0.05, hall=0.32, hp=250),
    "marimba": dict(prog=12, vol=0.55, pan=0.28, room=0.1, hall=0.2, hp=150),
    "harp": dict(prog=46, vol=0.42, pan=-0.3, room=0.04, hall=0.32, hp=180),
    "pizz": dict(prog=45, vol=0.55, pan=-0.22, room=0.08, hall=0.22, hp=140),
    "pad": dict(prog=89, vol=0.38, pan=0.0, room=0.0, hall=0.35, hp=150),
    "strings": dict(prog=49, vol=0.55, pan=0.12, room=0.0, hall=0.4, hp=110),
    "lowstr": dict(prog=48, vol=0.5, pan=-0.08, room=0.0, hall=0.35, hp=55),
    "stab": dict(prog=48, vol=0.55, pan=-0.1, room=0.05, hall=0.3, hp=120),
    "choir": dict(prog=52, vol=0.45, pan=0.0, room=0.0, hall=0.5, hp=160),
}
ALL = set(PARTS)
KICK, STICK, CLAP = 36, 37, 39
SHAKER = 82

NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def m(name):
    base = NOTE[name[0]]
    i = 1
    if i < len(name) and name[i] in "#b":
        base += 1 if name[i] == "#" else -1
        i += 1
    return base + (int(name[i:]) + 1) * 12


# 和弦：(低音, 钢琴/垫子的和弦音)；加九音让声音更亮更现代
CH = {
    "D": ("D2", ["D4", "F#4", "A4", "E5"]),
    "A/C#": ("C#2", ["C#4", "E4", "A4", "B4"]),
    "Bm": ("B1", ["D4", "F#4", "B4", "C#5"]),
    "G": ("G1", ["D4", "G4", "B4", "A4"]),
    "A": ("A1", ["C#4", "E4", "A4", "B4"]),
    "F#m": ("F#1", ["C#4", "F#4", "A4", "E5"]),
    "Em": ("E2", ["D4", "E4", "G4", "B4"]),
    "D/F#": ("F#1", ["D4", "F#4", "A4", "E5"]),
}
# 主旋律：按和弦写，一小节一句；(音, 拍, 时值拍数)
MEL = {
    "D": [("A5", 0, .75), ("F#5", .75, .25), ("E5", 1, .5), ("D5", 1.5, .5), ("E5", 2, 1), ("F#5", 3, 1)],
    "A/C#": [("E5", 0, .75), ("C#5", .75, .25), ("E5", 1, .5), ("A5", 1.5, .5), ("A5", 2, 1), ("B5", 3, 1)],
    "Bm": [("B5", 0, 1), ("A5", 1, .5), ("F#5", 1.5, .5), ("D5", 2, 1), ("F#5", 3, 1)],
    "G": [("G5", 0, .75), ("F#5", .75, .25), ("E5", 1, .5), ("D5", 1.5, .5), ("E5", 2, .5), ("F#5", 2.5, .5),
          ("A5", 3, 1)],
    "D/F#": [("A5", 0, .75), ("F#5", .75, .25), ("E5", 1, .5), ("D5", 1.5, .5), ("F#5", 2, 1), ("A5", 3, 1)],
    "Em": [("G5", 0, 1), ("F#5", 1, .5), ("E5", 1.5, .5), ("B4", 2, 1), ("E5", 3, 1)],
    "A": [("C#5", 0, 1), ("E5", 1, 1), ("A5", 2, 2)],
    "F#m": [("A5", 0, 1), ("F#5", 1, 1), ("E5", 2, 1), ("C#5", 3, 1)],
}

rng = np.random.default_rng(2026)


class Score:
    def __init__(self):
        self.ev = {k: [] for k in PARTS}
        self.cuts = []          # (声部集合, 开始秒, 结束秒)：这段时间里这些声部不出声，跨进来的长音在开始处截断
        self.gates = []         # (开始秒, 结束秒)：整条配乐（含混响尾巴）静音

    def note(self, part, t, dur, key, vel=80, human=0.004, force=False):
        if t < 0 or t > TOTAL + 1.5 or dur <= 0:
            return
        key = m(key) if isinstance(key, str) else key
        t = max(0.0, t + (rng.normal(0, human) if human else 0))
        if not force:
            for parts, a, b in self.cuts:
                if part in parts:
                    if a <= t < b:
                        return
                    if t < a < t + dur:
                        dur = a - t
        vel = int(max(1, min(127, vel + rng.normal(0, 3))))
        self.ev[part].append((t, 1, "on", key, vel))
        self.ev[part].append((t + dur, 0, "off", key, 0))

    def cc(self, part, t, ctrl, val):
        self.ev[part].append((max(0.0, t), 0, "cc", ctrl, int(val)))

    def swell(self, part, t0, t1, v0, v1, steps=24):
        for i in range(steps + 1):
            k = i / steps
            self.cc(part, t0 + (t1 - t0) * k, 11, v0 + (v1 - v0) * k)


def chord_of(name):
    bass, keys = CH[name]
    return m(bass), [m(k) for k in keys]


def arp_tones(name):
    _b, keys = chord_of(name)
    ks = sorted(keys)
    return ks + [k + 12 for k in ks]


def compose():
    S = Score()

    def tb(bar, beat=0.0):
        return (bar + beat / 4) * BAR

    # ---------- 静音窗口：蓄力小节最后半拍全体停住；片尾最后一个和弦之后收掉 ----------
    for b0, b1, kind, *_ in A.PLAN:
        if kind == "R":
            S.cuts.append((ALL, tb(b1) - BEAT / 2, tb(b1)))
            S.gates.append((tb(b1) - BEAT / 2 + 0.03, tb(b1)))
    end_bar = A.PLAN[-1][1]
    S.cuts.append((ALL, tb(end_bar), TOTAL + 2))

    def inside(blk, t):
        return blk[0] * BAR - 1e-6 <= t < blk[1] * BAR - 1e-6

    def bars_of(blk):
        b0, b1, _k, prog, _lv, _c = blk
        seq = A.PROG[prog]
        return [(b, seq[(b - b0) % len(seq)]) for b in range(b0, b1)]

    def ev(blk, part, bar, beat, dur_beats, key, vel, human=0.004):
        t = tb(bar, beat)
        if inside(blk, t):
            S.note(part, t, dur_beats * BEAT, key, vel, human=human)

    # ---------- 各声部的写法 ----------
    def shaker(blk, b, lv, sixteenth=False, base=40):
        step = 0.25 if sixteenth else 0.5
        for i in range(int(4 / step)):
            strong = (i * step) % 1 == 0.5 if not sixteenth else (i % 2 == 1)
            ev(blk, "perc", b, i * step, 0.1, SHAKER, (base + 18 if strong else base) * lv, human=0.002)

    def groove_drums(blk, b, lv, full):
        kv = 108 if full else 92
        hits = [(0, kv), (1.5, kv - 22), (2.0, kv - 10)]
        if full and b % 2 == 1:
            hits.append((3.5, kv - 28))
        for bt, v in hits:
            ev(blk, "kick", b, bt, 0.4, KICK, v * lv, human=0.002)
        for bt in (1, 3):
            ev(blk, "stick", b, bt, 0.3, STICK, (96 if full else 84) * lv, human=0.002)
            if full:
                ev(blk, "clap", b, bt + 0.01, 0.3, CLAP, 88 * lv, human=0.002)
        shaker(blk, b, lv, sixteenth=full, base=42 if full else 38)

    def bassline(blk, b, ch, nxt, lv, full):
        root, _ = chord_of(ch)
        nroot, _ = chord_of(nxt)
        root += 12
        nroot += 12
        pat = [(0, .9, root, 98), (1.5, .4, root, 80), (2.0, .9, root + 7, 86), (3.5, .4, nroot, 74)]
        if full:
            pat += [(1.0, .4, root + 12, 72), (3.0, .4, root + 7, 80)]
        for bt, d, k, v in pat:
            ev(blk, "bass", b, bt, d, k, v * lv)

    def piano_broken(blk, b, ch, lv):
        """钢琴分解和弦：低音 – 五度 – 和弦音往上走，八分音符"""
        root, keys = chord_of(ch)
        seq = [root + 24, root + 31] + sorted(keys)[1:] + [sorted(keys)[-1] + 12, sorted(keys)[2]]
        for i in range(8):
            ev(blk, "piano", b, i * 0.5, 0.9, seq[i % len(seq)], (60 if i % 2 == 0 else 50) * lv)

    def piano_offbeat(blk, b, ch, lv, drive=False):
        _r, keys = chord_of(ch)
        for i in range(8):
            if i % 2 == 0 and not drive:
                continue
            v = (56 if i % 2 else 64) + (8 if drive else 0)
            for k in keys[1:]:
                ev(blk, "piano", b, i * 0.5, 0.4, k, v * lv, human=0.003)
        ev(blk, "piano", b, 0, 1.8, keys[0], 58 * lv)

    def arps(blk, b, ch, lv, step, part="harp", vel=44, octave=12):
        tones = arp_tones(ch)
        order = [0, 2, 4, 6, 7, 5, 3, 1]
        for i in range(int(4 / step)):
            ev(blk, part, b, i * step, step * 0.95, tones[order[i % 8]] + octave - 12,
               (vel + (6 if i % 2 == 0 else 0)) * lv)

    def cycle_figure(blk, b, ch, lv):
        """循环音型：马林巴十六分音符一圈一圈地转（上去、下来、再上去）"""
        tones = arp_tones(ch)
        order = [0, 1, 2, 3, 4, 3, 2, 1, 0, 2, 4, 5, 6, 5, 4, 2]
        for i in range(16):
            ev(blk, "marimba", b, i * 0.25, 0.22, tones[order[i]] + 12, (62 if i % 4 == 0 else 48) * lv, human=0.002)
        for bt in (0.5, 1.5, 2.5, 3.5):
            ev(blk, "pizz", b, bt, 0.3, tones[(int(bt) * 2 + b) % len(tones)], 64 * lv)

    def pad_chord(blk, b, ch, lv, part="pad", vel=56, octave=0):
        _r, keys = chord_of(ch)
        for k in keys[:3]:
            ev(blk, part, b, 0, 4.02, k + octave, vel * lv)

    def melody(blk, b, ch, lv, part="lead", vel=86):
        for nm, bt, d in MEL.get(ch, MEL["D"]):
            ev(blk, part, b, bt, d * 0.92, nm, vel * lv)

    def big_hit(blk, b, ch, lv):
        """段落第一拍的重音：定音鼓 + 钢琴低八度 + 弦乐短奏"""
        root, keys = chord_of(ch)
        ev(blk, "timp", b, 0, 1.0, root + 12, 112 * lv)
        for k in (root + 12, root + 24):
            ev(blk, "piano", b, 0, 2.0, k, 96 * lv)
        for k in keys[:3]:
            ev(blk, "stab", b, 0, 0.5, k, 96 * lv)

    def dew(blk, b, ch, lv):
        """暗场里零星落下的钢琴高音，像水滴"""
        tones = sorted(chord_of(ch)[1])
        for bt, k in ((1.5, tones[-1] + 12), (3.0, tones[1] + 24)):
            ev(blk, "lead", b, bt, 1.2, k, 46 * lv)

    # ---------- 逐段写 ----------
    plan = A.PLAN
    for idx, blk in enumerate(plan):
        b0, b1, kind, prog, lv, color = blk
        prev_kind = plan[idx - 1][2] if idx else ""
        bars = bars_of(blk)
        for j, (b, ch) in enumerate(bars):
            if j + 1 < len(bars):
                nxt = bars[j + 1][1]
            elif idx + 1 < len(plan):
                nxt = A.PROG[plan[idx + 1][3]][0]
            else:
                nxt = ch
            root, keys = chord_of(ch)
            if kind == "L":
                piano_broken(blk, b, ch, lv)
                for bt in (1.5, 3.5):
                    ev(blk, "pizz", b, bt, 0.3, sorted(keys)[-1] + 12 if bt == 1.5 else sorted(keys)[2] + 12, 62 * lv)
                arps(blk, b, ch, lv, 1.0, vel=46)
                pad_chord(blk, b, ch, lv, vel=46)
                ev(blk, "kick", b, 0, 0.3, KICK, 70 * lv, human=0.001)
                ev(blk, "kick", b, 2, 0.3, KICK, 58 * lv, human=0.001)
                shaker(blk, b, lv, base=30)
                ev(blk, "bass", b, 0, 3.8, root + 12, 64 * lv)
            elif kind == "G":
                groove_drums(blk, b, lv, full=False)
                bassline(blk, b, ch, nxt, lv, full=False)
                piano_offbeat(blk, b, ch, lv)
                arps(blk, b, ch, lv, 0.5, part="marimba", vel=50)
                pad_chord(blk, b, ch, lv, vel=50)
                if b % 2 == 1:
                    # 每两小节一次小回应：长笛三个音往上走
                    top = sorted(keys)[-3:]
                    for i, k in enumerate(top):
                        ev(blk, "flute", b, 2.5 + i * 0.5, 0.45, k + 12, (66 + 4 * i) * lv)
                if j == 0 and prev_kind in ("R", "L", "M"):
                    big_hit(blk, b, ch, lv * 0.7)
            elif kind == "M":
                pad_chord(blk, b, ch, lv, vel=64)
                pad_chord(blk, b, ch, lv, part="lowstr", vel=70, octave=-12)
                arps(blk, b, ch, lv, 1.0, vel=48)
                dew(blk, b, ch, lv)
                ev(blk, "bass", b, 0, 3.9, root + 12, 72 * lv)
                ev(blk, "kick", b, 0, 0.3, KICK, 74 * lv, human=0.001)
                ev(blk, "kick", b, 0.75, 0.3, KICK, 52 * lv, human=0.001)
                shaker(blk, b, lv, base=24)
            elif kind == "R":
                # 一小节的蓄力：定音鼓从八分滚到十六分，弦乐渐强，竖琴往上扫，最后半拍停住
                for i in range(14):
                    bt = i * 0.5 if i < 4 else 2 + (i - 4) * 0.15
                    if bt < 3.5:
                        ev(blk, "timp", b, bt, 0.3, root + 12, (60 + 4 * i) * lv, human=0.001)
                for k in keys[:3]:
                    ev(blk, "strings", b, 0, 3.4, k + 12, 70 * lv)
                S.swell("strings", tb(b), tb(b, 3.45), 40, 127)
                S.cc("strings", tb(b + 1) - 0.01, 11, 100)
                ev(blk, "bass", b, 0, 3.4, root + 12, 84 * lv)
                pad_chord(blk, b, ch, lv, vel=56)
                for i, k in enumerate(arp_tones(ch)):
                    ev(blk, "harp", b, 1.6 + i * 0.22, 0.4, k + 12, (40 + 6 * i) * lv, human=0)
                for i in range(8):
                    ev(blk, "perc", b, i * 0.25 + 1.5, 0.1, SHAKER, (40 + 6 * i) * lv, human=0.001)
            elif kind == "F":
                if j == 0:
                    big_hit(blk, b, ch, lv)
                groove_drums(blk, b, lv, full=True)
                bassline(blk, b, ch, nxt, lv, full=True)
                piano_offbeat(blk, b, ch, lv * 0.9, drive=True)
                melody(blk, b, ch, lv)
                melody(blk, b, ch, lv * 0.8, part="flute", vel=80)
                arps(blk, b, ch, lv, 0.25, vel=40)
                pad_chord(blk, b, ch, lv, vel=64)
                pad_chord(blk, b, ch, lv, part="strings", vel=74, octave=12)
                pad_chord(blk, b, ch, lv, part="lowstr", vel=66, octave=-12)
                if color == "choir":
                    pad_chord(blk, b, ch, lv, part="choir", vel=86, octave=12)
            elif kind == "C":
                groove_drums(blk, b, lv, full=False)
                bassline(blk, b, ch, nxt, lv, full=False)
                cycle_figure(blk, b, ch, lv)
                pad_chord(blk, b, ch, lv, vel=54)
                if b >= 34:
                    # 后半段（糖连成长链）弦乐进来，往上长
                    pad_chord(blk, b, ch, lv, part="strings", vel=60 + 4 * (b - 34), octave=12)
                if b % 4 == 3:
                    ev(blk, "flute", b, 2, 2, sorted(keys)[-1] + 12, 64 * lv)
            elif kind == "O":
                fade = [1.0, 0.85, 0.75][min(2, b - b0)]
                for kk in keys:
                    ev(blk, "piano", b, 0, 3.9 if b < b1 - 1 else 7.0, kk, 54 * lv * fade)
                ev(blk, "bass", b, 0, 3.8, root + 12, 54 * lv * fade)
                pad_chord(blk, b, ch, lv * fade, vel=46)
                if b == b0:
                    for nm, bt, d in (("B5", 0, 1), ("A5", 1, 1), ("F#5", 2, 2)):
                        ev(blk, "lead", b, bt, d, nm, 58 * lv)
                elif b == b0 + 1:
                    for nm, bt, d in (("E5", 0, 1), ("C#5", 1, 1), ("E5", 2, 2)):
                        ev(blk, "lead", b, bt, d, nm, 52 * lv)
                else:
                    ev(blk, "lead", b, 0, 4, "F#5", 46 * lv)
    return S


# ---------------------------------------------------------------- 渲染
def render_part(spec, events, sfbytes):
    syn = tsf.Synth(samplerate=SR)
    sfid = syn.sfload(sfbytes)
    syn.program_select(0, sfid, spec.get("bank", 0), spec["prog"], spec.get("drums", False))
    syn.control_change(0, 11, 100)
    out = np.zeros((N, 2), np.float32)
    pos = 0
    for t, _, kind, a, b in sorted(events, key=lambda x: (x[0], x[1])):
        smp = min(N, int(t * SR))
        if smp > pos:
            out[pos:smp] = np.frombuffer(syn.generate_simple(smp - pos), np.float32).reshape(-1, 2)
            pos = smp
        if kind == "on":
            syn.noteon(0, a, b)
        elif kind == "off":
            syn.noteoff(0, a)
        elif kind == "cc":
            syn.control_change(0, a, b)
    if pos < N:
        out[pos:] = np.frombuffer(syn.generate_simple(N - pos), np.float32).reshape(-1, 2)
    return out


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "lowpass", fs=SR, output="sos"), x, axis=0)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "highpass", fs=SR, output="sos"), x, axis=0)


def reverb_ir(seconds, predelay, seed, tone=6000):
    n = int(seconds * SR)
    g = np.random.default_rng(seed)
    x = np.arange(n) / SR
    ir = np.stack([lp(g.standard_normal(n), tone, 1) * np.exp(-x * 6.9 / seconds) for _ in range(2)], 1)
    ir = np.concatenate([np.zeros((int(predelay * SR), 2)), ir])
    return ir / np.sqrt((ir ** 2).sum(0))


def convolve(x, ir):
    return np.stack([fftconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1).astype(np.float32)


def split_bands(x, edges):
    """互补分频：各频段相加严格等于原信号"""
    out = []
    rest = x
    for f in edges:
        low = lp(rest, f, 4).astype(np.float32)
        out.append(low)
        rest = rest - low
    out.append(rest)
    return out


def band_db(x):
    return np.array([10 * np.log10((b ** 2).mean() + 1e-12) for b in split_bands(x, BANDS)])


def envelope(x, attack, release, step=32):
    pk = maximum_filter1d(x, size=step)[::step]
    aa, ar = math.exp(-step / (attack * SR)), math.exp(-step / (release * SR))
    env = np.zeros_like(pk)
    prev = 0.0
    for i, val in enumerate(pk):
        coef = aa if val > prev else ar
        prev = coef * prev + (1 - coef) * val
        env[i] = prev
    return np.interp(np.arange(len(x)), np.arange(len(env)) * step, env)


def compress(x, ratio, pct, attack, release):
    """压缩（立体声联动）：超过整段第 pct 百分位电平的部分按 ratio 压下去"""
    a = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    env = envelope(a, attack, release)
    thr = np.percentile(env[env > 1e-4], pct)
    g = np.where(env > thr, (np.maximum(env, 1e-9) / thr) ** (1 / ratio - 1), 1.0).astype(np.float32)
    return x * (g[:, None] if x.ndim == 2 else g)


def limiter(x, ceiling=0.8, look=0.005, release=0.12):
    """前视限幅：增益先取前视窗口内的最小值，再用同样长度的窗口平滑，避免一个采样内的硬阶跃"""
    a = np.abs(x).max(axis=1)
    L = 2 * int(look * SR) + 1
    pk = maximum_filter1d(a, size=L)
    g = np.minimum(1.0, ceiling / np.maximum(pk, 1e-9))
    step = 32
    gs = g[::step].copy()
    coef = math.exp(-step / (release * SR))
    for i in range(1, len(gs)):
        gs[i] = min(gs[i], coef * gs[i - 1] + (1 - coef) * gs[i])
    g2 = np.minimum(np.interp(np.arange(len(g)), np.arange(len(gs)) * step, gs), g)
    g2 = np.convolve(g2, np.ones(L) / L, mode="same")
    g2 = np.minimum(g2, minimum_filter1d(g, size=L))
    return x * g2[:, None], g2


def voice_track():
    v = np.zeros(N, np.float32)
    for lid, (a, _b) in T.items():
        sr, x = wavfile.read(VOICE / f"{lid}.wav")
        x = x.astype(np.float32) / 32768.0
        i = int(round(a * SR))
        v[i:i + len(x)] += x[: max(0, min(len(x), N - i))]
    return hp(v, 80).astype(np.float32)


def speech_lines():
    """每句旁白的 (起, 止) 秒，按时间排序"""
    return sorted(T.values())


def duck_curve(lines, levels, n, step=32):
    """配乐让位的增益曲线（dB，每 step 个采样一个点）：每句话一个固定值；
    两句之间空档短就不回升（从上一句的值平滑过渡到下一句），空档长就回到 0；
    砸点紧挨着话头时，砸下前把配乐放回 0，砸下的那一下过去后再压"""
    m = n // step + 2
    tgt = np.zeros(m)
    drops = list(A.drop_times())
    wins = []
    for (a, b), g in zip(lines, levels):
        hits = [d for d in drops if a - 0.3 <= d <= a + 0.05]
        s = max(a - DUCK_PRE, max(hits) + HIT_PASS) if hits else a - DUCK_PRE
        wins.append((s, b + DUCK_POST, g, max(hits) if hits else None))
    idx = lambda t: min(m, max(0, int(round(t * SR / step))))
    for i, (s, e, g, _hit) in enumerate(wins):
        i0, i1 = idx(s), idx(e)
        tgt[i0:i1] = np.minimum(tgt[i0:i1], g)
        if i + 1 < len(wins):
            s2, _e2, g2, hit2 = wins[i + 1]
            j1 = idx(s2)
            if s2 - e < DUCK_BRIDGE and j1 > i1:
                if hit2 is None:
                    tgt[i1:j1] = np.linspace(g, g2, j1 - i1)
                else:
                    # 砸点前：在空档里放回 0（多数砸点前乐队本来就停半拍，放回来听不出）
                    jh = max(i1 + 1, idx(hit2 - 0.02))
                    tgt[i1:jh] = np.linspace(g, 0.0, jh - i1)

    # 压下去要提前到位（倒着走，限速 DUCK_ATTACK_DBS），放回来要慢（顺着走，限速 DUCK_RELEASE_DBS）
    up = DUCK_RELEASE_DBS * step / SR
    down = DUCK_ATTACK_DBS * step / SR
    y = tgt.copy()
    for i in range(1, m):
        y[i] = min(tgt[i], y[i - 1] + up)
    for i in range(m - 2, -1, -1):
        y[i] = min(y[i], y[i + 1] + down)
    return np.interp(np.arange(n), np.arange(m) * step, y).astype(np.float32)


def duck_music(music, vo, target_ld, log=True):
    """配乐让位：说话时中高频先按 SPEECH_EQ_DB 倾斜，再逐句算压低量，
    让每句旁白比配乐高 target_ld（再加上这句所在织体的 TEXTURE_EXTRA_LD）；返回 (让位后的配乐, 每句的差)"""
    n = len(music)
    lines = speech_lines()
    kinds = [A.texture((a + b) / 2) for a, b in lines]
    want = np.array([target_ld + TEXTURE_EXTRA_LD.get(k, 0.0) for k in kinds])
    low = lp(music, 250, 4).astype(np.float32)
    high = hp(music, 5000, 4).astype(np.float32)
    mid = music - low - high

    # 说话时的频段倾斜：曲线用同样的时间规则（按压低 10 dB 的速度走），换算成 0（没说话）~ 1（说话）
    act = np.clip(-duck_curve(lines, [-10.0] * len(lines), n) / 10.0, 0, 1)[:, None]
    shaped = (low * (10 ** (SPEECH_EQ_DB["low"] * act / 20)) + mid * (10 ** (SPEECH_EQ_DB["mid"] * act / 20))
              + high * (10 ** (SPEECH_EQ_DB["high"] * act / 20))).astype(np.float32)

    # 逐句求压低量：压低量和这句的响度差基本是一一对应的，迭代几次把相邻句子过渡段的影响也算进去
    meter = pyln.Meter(SR)
    spans = [slice(int(a * SR), int(b * SR)) for a, b in lines]
    lv = np.array([meter.integrated_loudness(vo[s]) for s in spans])
    g = np.zeros(len(lines))
    for _ in range(5):
        bed = shaped * (10 ** (duck_curve(lines, g, n) / 20))[:, None]
        lb = np.array([meter.integrated_loudness(bed[s]) for s in spans])
        ld = lv - lb
        g = np.clip(g + np.where(np.isfinite(ld), ld - want, 0.0), -MAX_DUCK_DB, 0.0)
    bed = shaped * (10 ** (duck_curve(lines, g, n) / 20))[:, None]
    lb = np.array([meter.integrated_loudness(bed[s]) for s in spans])
    ld = lv - lb
    if log:
        fin = ld[np.isfinite(ld)]
        by = "，".join(f"{k} {np.median([d for d, kk in zip(ld, kinds) if kk == k and np.isfinite(d)]):.1f}"
                      for k in "LGMRFCO" if any(kk == k and np.isfinite(d) for d, kk in zip(ld, kinds)))
        print(f"{LOG} 让位：下限旁白比配乐高 {target_ld:.0f} LU；实际每句最小 {fin.min():.1f} LU，按织体的中位（LU）：{by}；"
              f"压低量最多 {-g.min():.1f} dB，中位 {-np.median(g):.1f} dB", flush=True)
    return bed.astype(np.float32), ld


def true_peak_db(x):
    from scipy.signal import resample_poly
    up = resample_poly(x, 4, 1, axis=0)
    return 20 * math.log10(np.abs(up).max() + 1e-12)


def to_lufs(meter, x, target):
    return x * (10 ** ((target - meter.integrated_loudness(x)) / 20))


def texture_slices(kind):
    return [slice(int(b0 * BAR * SR), int(b1 * BAR * SR)) for b0, b1, k, *_ in A.PLAN if k == kind]



def main():
    S = compose()
    sfbytes = SF2.read_bytes()
    dry = np.zeros((N, 2), np.float32)
    room = np.zeros((N, 2), np.float32)
    hall = np.zeros((N, 2), np.float32)
    for name, evs in S.ev.items():
        if not evs:
            continue
        spec = PARTS[name]
        y = render_part(spec, evs, sfbytes)
        y = hp(y, spec["hp"]).astype(np.float32)
        p = spec["pan"]
        y[:, 0] *= math.sqrt((1 - p) / 2) * 1.414
        y[:, 1] *= math.sqrt((1 + p) / 2) * 1.414
        y *= spec["vol"]
        dry += y
        room += y * spec["room"]
        hall += y * spec["hall"]
        print(f"{LOG}[{name}] 音符 {sum(1 for e in evs if e[2] == 'on')}", flush=True)
    music = dry + convolve(room, reverb_ir(0.8, 0.012, 7, 7000)) * 0.6 + convolve(hall, reverb_ir(2.4, 0.03, 9, 6000)) * 0.75
    music = hp(music, 30).astype(np.float32)
    # 总线：温和的胶水压缩 + 轻微的软削波
    music = compress(music, 2.0, 92, 0.01, 0.2)
    music = music / (np.abs(music).max() + 1e-9)
    music = (np.tanh(music * 1.2) / np.tanh(1.2)).astype(np.float32)

    # 静音窗口（含混响尾巴）：两头 15ms 淡入淡出；片尾最后一个和弦自然衰减后淡出
    tt = np.arange(N) / SR
    gate = np.ones(N, np.float32)
    for a_, b_ in S.gates:
        gate *= np.clip(np.maximum((a_ - tt) / 0.015, (tt - b_) / 0.015 + 1), 0, 1).astype(np.float32)
    gate *= np.clip((TOTAL + 0.3 - tt) / 1.5, 0, 1).astype(np.float32)
    music *= gate[:, None]

    meter = pyln.Meter(SR)
    for kind in "LGMCF":
        sl = texture_slices(kind)
        if sl:
            seg = np.concatenate([music[s] for s in sl])
            bd = band_db(seg)
            print(f"{LOG} 织体 {kind} 的响度 {meter.integrated_loudness(seg):.1f} LUFS，频段能量（dB，<120/120-500/500-2k/2k-6k/>6k）"
                  + " ".join(f"{x - bd.max():+.1f}" for x in bd))
    OUT.mkdir(exist_ok=True)
    mo = music / (np.abs(music).max() + 1e-9) * 0.85
    wavfile.write(OUT / "music_only.wav", SR, (mo * 32767).astype(np.int16))
    mix = mixdown(music, TARGET_LD)
    wavfile.write(OUT / "mix.wav", SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    print(f"{LOG} 输出 {OUT / 'mix.wav'}")


def mixdown(music, target_ld):
    """配乐（让位前）+ 旁白 → 成片音轨：配乐定电平、逐句让位、叠旁白、归一响度、限幅"""
    meter = pyln.Meter(SR)
    music = to_lufs(meter, music, MUSIC_LUFS)
    v = compress(voice_track(), 3.0, 60, 0.004, 0.08)
    vo = to_lufs(meter, np.stack([v, v], 1), VOICE_LUFS)
    bed, _ld = duck_music(music, vo, target_ld)
    mix = bed + vo
    for nm, part in (("旁白", vo), ("配乐（让位前）", music), ("配乐（让位后）", bed)):
        print(f"{LOG} 分轨 {nm} {meter.integrated_loudness(part):.1f} LUFS")
    loud = meter.integrated_loudness(mix)
    mix, g1 = limiter(pyln.normalize.loudness(mix, loud, FINAL_LUFS), ceiling=CEILING)
    loud2 = meter.integrated_loudness(mix)
    mix, g2 = limiter(pyln.normalize.loudness(mix, loud2, FINAL_LUFS), ceiling=CEILING)
    gr = 20 * np.log10(np.minimum(g1, 1) * np.minimum(g2, 1) + 1e-9)
    body = slice(int(0.05 * SR), int((TOTAL - 0.05) * SR))
    print(f"{LOG} 限幅压缩量（正片）：最大 {-gr[body].min():.1f} dB，超过 1 dB 的时间占 {(gr[body] < -1).mean() * 100:.2f}%")
    print(f"{LOG} 响度 {loud:.1f} → {meter.integrated_loudness(mix):.1f} LUFS，真峰值 {true_peak_db(mix):.2f} dBTP")
    return mix


if __name__ == "__main__":
    main()

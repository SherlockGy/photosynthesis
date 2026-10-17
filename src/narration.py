# 旁白合成（温柔学姐）：逐句合成，裁首尾静音，记录时长；文本没变的句子不重复合成
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from minimax import tts
from script import LINES, SPEED, TONE, VOICE

ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = ROOT / "assets" / "voice"
META = VOICE_DIR / "durations.json"
LOG = "[narration 旁白合成]"


def trim(path):
    sr, x = wavfile.read(path)
    xf = x.astype(np.float32) / 32768.0
    idx = np.where(np.abs(xf) > 0.012)[0]
    a = max(0, idx[0] - int(0.03 * sr))
    b = min(len(xf), idx[-1] + int(0.08 * sr))
    xf = xf[a:b]
    n = int(0.008 * sr)
    xf[:n] *= np.linspace(0, 1, n)
    xf[-n:] *= np.linspace(1, 0, n)
    wavfile.write(path, sr, (xf * 32767).astype(np.int16))
    return len(xf) / sr


def main(force=()):
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    for lid, text, _sec, _gap in LINES:
        out = VOICE_DIR / f"{lid}.wav"
        sig = f"{VOICE}|{text}|{SPEED}|{'/'.join(TONE)}"
        if out.exists() and meta.get(lid, {}).get("sig") == sig and lid not in force:
            continue
        print(f"{LOG}[{lid}] {text}", flush=True)
        tts(text, out, VOICE, speed=SPEED, tone=TONE)
        meta[lid] = {"sig": sig, "dur": round(trim(out), 3)}
        META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{LOG} 完成，纯旁白 {sum(meta[l[0]]['dur'] for l in LINES):.1f}s")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))

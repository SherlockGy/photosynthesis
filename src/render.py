# 渲染：preview 抽帧拼图；video 分段并行渲染后与混音合成（24 fps，每一帧都单独渲染，推镜才顺）
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from timeline import FPS, H, NFRAMES, TOTAL, W

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
LOG = "[render 光合作用渲染]"
NAME = "光合作用"


def preview(times, name="sheet", cols=4):
    from scenes import render_frame
    (OUT / "preview").mkdir(parents=True, exist_ok=True)
    thumbs = []
    for t in times:
        img = render_frame(t)
        Image.fromarray(img).save(OUT / "preview" / f"t_{t:07.2f}.png")
        thumbs.append(Image.fromarray(img).resize((480, 270)))
    sheet = Image.new("RGB", (480 * cols, 270 * ((len(thumbs) + cols - 1) // cols)))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 480, (i // cols) * 270))
    sheet.save(OUT / "preview" / f"{name}.jpg", quality=88)


def chunk(args):
    a, b, path = args
    from scenes import render_frame
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(a, b):
        p.stdin.write(np.ascontiguousarray(render_frame(f / FPS)).tobytes())
    p.stdin.close()
    p.wait()
    return path


def video(workers=4, parts=16, t0=0.0, t1=None, name=NAME):
    (OUT / "chunks").mkdir(parents=True, exist_ok=True)
    f0 = int(t0 * FPS)
    f1 = NFRAMES if t1 is None else min(NFRAMES, int(t1 * FPS))
    size = (f1 - f0 + parts - 1) // parts
    jobs = [(f0 + i * size, min(f1, f0 + (i + 1) * size), OUT / "chunks" / f"{name}_{i:02d}.mp4") for i in range(parts)
            if f0 + i * size < f1]
    st = time.time()
    with ProcessPoolExecutor(workers) as ex:
        paths = list(ex.map(chunk, jobs))
    lst = OUT / "chunks" / f"{name}_list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in paths), encoding="utf-8")
    master = OUT / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-ss", f"{t0:.3f}",
                    "-i", str(OUT / "mix.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a",
                    "256k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(master)], check=True)
    print(f"{LOG} 完成 {master}，用时 {time.time() - st:.0f}s，时长 {(f1 - f0) / FPS:.1f}s")


def redo(times, workers=2, parts=16):
    """局部修改后只重渲染包含这些时刻的分段，再用已有分段重新合成（分段划分必须和上次整片渲染一致）"""
    size = (NFRAMES + parts - 1) // parts
    idx = sorted({min(parts - 1, int(t * FPS) // size) for t in times})
    jobs = [(i * size, min(NFRAMES, (i + 1) * size), OUT / "chunks" / f"{NAME}_{i:02d}.mp4") for i in idx]
    print(f"{LOG} 重渲染分段 {idx}")
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(chunk, jobs))
    lst = OUT / "chunks" / f"{NAME}_list.txt"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i",
                    str(OUT / "mix.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-ar", "48000", "-shortest", "-movflags", "+faststart", str(OUT / f"{NAME}.mp4")], check=True)
    print(f"{LOG} 已重新合成 {OUT / f'{NAME}.mp4'}")


if __name__ == "__main__":
    if sys.argv[1] == "preview":
        preview([float(x) for x in sys.argv[2:]])
    elif sys.argv[1] == "redo":
        # 只重渲染包含这些时刻的分段：render.py redo 52.5 89
        redo([float(x) for x in sys.argv[2:]])
    elif sys.argv[1] == "clip":
        # 只渲染一段：render.py clip 起 止 [进程数]
        video(int(sys.argv[4]) if len(sys.argv) > 4 else 4, 8, float(sys.argv[2]), float(sys.argv[3]),
              f"片段_{sys.argv[2]}-{sys.argv[3]}")
    else:
        video(int(sys.argv[2]) if len(sys.argv) > 2 else 4)

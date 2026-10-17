# 逐帧检查：标签出画、标签压字幕、标签互相重叠（按 4 fps 走完全片，只看透明度 > 0.3 的标签）
import sys
from concurrent.futures import ProcessPoolExecutor

from timeline import TOTAL

MARGIN = 16
SUB_BAR = (978, 1046)


def boxes_at(t):
    import scenes as S
    from draw import font
    S.render_frame(t)
    out = []
    for lb in S.LAST_LABELS:
        if lb["a"] < 0.3:
            continue
        f = font(lb["size"], 600)
        w = f.measureText(lb["text"])
        cx, by = lb["x"] + lb["dx"], lb["y"] + lb["dy"]
        h = lb["size"]
        out.append((lb["text"], (cx - w / 2, by - h * 0.9, cx + w / 2, by + h * 0.25)))
    sub = None
    for t0, t1, txt in S._sub_chunks():
        if t0 <= t < t1 and t < S.Q["question"]:
            w = font(44, 600).measureText(txt)
            sub = (960 - w / 2 - 28, SUB_BAR[0], 960 + w / 2 + 28, SUB_BAR[1])
    return t, out, sub


def inter(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def main(workers=6):
    ts = [i / 4 for i in range(int(TOTAL * 4))]
    issues = []
    with ProcessPoolExecutor(workers) as ex:
        for t, boxes, sub in ex.map(boxes_at, ts, chunksize=8):
            for name, b in boxes:
                if b[0] < MARGIN or b[1] < MARGIN or b[2] > 1920 - MARGIN or b[3] > 1080 - MARGIN:
                    issues.append((t, "出画", name))
                if sub and inter(b, sub):
                    issues.append((t, "压字幕", name))
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if inter(boxes[i][1], boxes[j][1]):
                        issues.append((t, "标签重叠", boxes[i][0] + " / " + boxes[j][0]))
    # 合并连续时间段
    merged = {}
    for t, kind, name in issues:
        merged.setdefault((kind, name), []).append(t)
    for (kind, name), tl in sorted(merged.items(), key=lambda x: x[1][0]):
        print(f"[audit 逐帧检查] {kind} 「{name}」 {tl[0]:.2f}s ~ {tl[-1]:.2f}s，共 {len(tl)} 帧")
    print(f"[audit 逐帧检查] 共检查 {len(ts)} 帧，问题 {len(issues)} 处")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)

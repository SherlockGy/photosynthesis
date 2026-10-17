# 绘图工具：手绘描边、排线纹理、纸张底、显微暗场底、字体与标签
# 风格依据：程序化矢量插画——墨线略带抖动，形状内填短排线，浅色场景用米色纸底，显微场景用深绿暗场加屏幕斜纹
# 所有随机都由种子决定，任意时刻的画面只取决于时间 t
import functools
import math
import os
from pathlib import Path

import numpy as np
import skia

ROOT = Path(__file__).resolve().parent.parent
W, H = 1920, 1080

# ---------- 配色 ----------
INK = (0.17, 0.12, 0.10)          # 墨线：深棕黑
PAPER = (0.925, 0.905, 0.855)     # 米色纸底
PAPER_WARM = (0.965, 0.895, 0.765)
CREAM = (0.97, 0.955, 0.91)
WHITE = (0.99, 0.985, 0.96)

LEAF = (0.47, 0.71, 0.28)         # 叶片绿
LEAF_DK = (0.33, 0.56, 0.20)
LEAF_LT = (0.80, 0.90, 0.58)      # 叶脉、排线亮色
VEIN = (0.86, 0.93, 0.66)
POT = (0.72, 0.33, 0.24)          # 陶盆
POT_DK = (0.50, 0.20, 0.15)
SUN = (0.96, 0.66, 0.22)
SUN_RAY = (0.93, 0.47, 0.20)
BRASS = (0.83, 0.62, 0.20)
WOOD = (0.45, 0.28, 0.16)
WATER = (0.55, 0.80, 0.86)        # 水流浅蓝
WATER_DK = (0.33, 0.62, 0.70)

EPI = (0.87, 0.91, 0.78)          # 表皮细胞浅绿
EPI_LINE = (0.70, 0.79, 0.58)
GUARD = (0.50, 0.72, 0.32)        # 保卫细胞

DARK = (0.075, 0.19, 0.125)       # 显微暗场
DARK_MID = (0.11, 0.27, 0.17)
DARK_HI = (0.16, 0.36, 0.23)
MEM = (0.34, 0.70, 0.36)          # 膜的亮绿
MEM_LT = (0.58, 0.85, 0.58)
GRANA = (0.30, 0.66, 0.30)
GRANA_TOP = (0.55, 0.88, 0.42)
PROT = (0.10, 0.12, 0.10)         # 蛋白复合体：近黑
CHL = (0.42, 0.80, 0.40)          # 叶绿素
O_RED = (0.84, 0.40, 0.29)        # 氧原子
C_GRAY = (0.24, 0.23, 0.24)       # 碳原子
H_WHITE = (0.95, 0.93, 0.88)      # 氢原子
GOLD = (0.95, 0.76, 0.30)         # 能量颗粒、光子
SUGAR = (0.97, 0.94, 0.84)

# 当前图层在屏幕上的缩放倍数：描边宽度按它的平方根回收，推镜时墨线不会粗得离谱
ZK = {"z": 1.0}


def lw(w):
    return w / math.sqrt(max(ZK["z"], 1e-3))


# ---------- 数学与时间 ----------

def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, k):
    if isinstance(a, tuple):
        return tuple(x + (y - x) * k for x, y in zip(a, b))
    return a + (b - a) * k


def ease_io(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3):
    x = clamp(x)
    return 1 - (1 - x) ** p


def ease_in(x, p=2):
    return clamp(x) ** p


def ramp(t, a, b, fn=ease_io):
    if b <= a:
        return 1.0 if t >= b else 0.0
    return fn(clamp((t - a) / (b - a)))


def hrand(*key):
    """由键值决定的伪随机数 0~1"""
    h = 2166136261
    for k in key:
        for ch in str(k):
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFFFF) / 0xFFFFFF


def srand(*key):
    """-1~1 的伪随机数"""
    return hrand(*key) * 2 - 1


def wave(x, seed, octaves=3):
    """平滑的一维噪声：几组随机相位的正弦叠加，范围约 -1~1"""
    v = 0.0
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        f = (1.3 + hrand(seed, o, "f")) * (2 ** o)
        v += amp * math.sin(x * f + hrand(seed, o, "p") * 6.283)
        tot += amp
        amp *= 0.5
    return v / tot


# ---------- 颜色与画笔 ----------

def C(c, a=1.0):
    return skia.Color4f(c[0], c[1], c[2], clamp(a))


def fill(c, a=1.0):
    return skia.Paint(AntiAlias=True, Color=C(c, a))


def stroke(c, w, a=1.0, cap=skia.Paint.kRound_Cap):
    return skia.Paint(AntiAlias=True, Color=C(c, a), Style=skia.Paint.kStroke_Style, StrokeWidth=w,
                      StrokeCap=cap, StrokeJoin=skia.Paint.kRound_Join)


def shade(c, k):
    """k<0 变暗，k>0 变亮"""
    if k < 0:
        return tuple(x * (1 + k) for x in c)
    return tuple(x + (1 - x) * k for x in c)


# ---------- 路径 ----------

def smooth_path(pts, closed=True):
    """过点的平滑曲线（中点二次贝塞尔）"""
    p = skia.Path()
    n = len(pts)
    if n < 3:
        p.moveTo(*pts[0])
        for q in pts[1:]:
            p.lineTo(*q)
        return p
    if closed:
        mid = ((pts[-1][0] + pts[0][0]) / 2, (pts[-1][1] + pts[0][1]) / 2)
        p.moveTo(*mid)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            p.quadTo(a[0], a[1], (a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        p.close()
    else:
        p.moveTo(*pts[0])
        for i in range(1, n - 1):
            a = pts[i]
            b = pts[i + 1]
            p.quadTo(a[0], a[1], (a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        p.lineTo(*pts[-1])
    return p


def resample(pts, step, closed=True):
    """把折线按固定步长重新取点"""
    seq = list(pts) + ([pts[0]] if closed else [])
    out = [seq[0]]
    carry = 0.0
    for i in range(len(seq) - 1):
        x0, y0 = seq[i]
        x1, y1 = seq[i + 1]
        L = math.hypot(x1 - x0, y1 - y0)
        d = step - carry
        while d <= L:
            k = d / L
            out.append((x0 + (x1 - x0) * k, y0 + (y1 - y0) * k))
            d += step
        carry = L - (d - step)
    if closed and len(out) > 2 and math.hypot(out[-1][0] - out[0][0], out[-1][1] - out[0][1]) < step * 0.5:
        out.pop()
    return out


def wobble(pts, seed, amp=2.0, step=14.0, closed=True, freq=0.09):
    """手绘抖动：沿法线方向加平滑噪声位移"""
    rs = resample(pts, step, closed)
    n = len(rs)
    out = []
    s = 0.0
    for i in range(n):
        a = rs[i - 1] if (closed or i > 0) else rs[i]
        b = rs[(i + 1) % n] if (closed or i < n - 1) else rs[i]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        d = amp * wave(s * freq, seed)
        out.append((rs[i][0] + nx * d, rs[i][1] + ny * d))
        s += step
    return out


def ellipse_pts(cx, cy, rx, ry, n=48, rot=0.0):
    cr, sr = math.cos(rot), math.sin(rot)
    out = []
    for i in range(n):
        a = i / n * 2 * math.pi
        x, y = rx * math.cos(a), ry * math.sin(a)
        out.append((cx + x * cr - y * sr, cy + x * sr + y * cr))
    return out


def rrect_pts(x, y, w, h, r, n=8):
    """圆角矩形的点列"""
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def xform(path, m):
    """路径套上矩阵，返回新路径"""
    q = skia.Path(path)
    q.transform(m)
    return q


def poly_path(pts, closed=True):
    p = skia.Path()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    if closed:
        p.close()
    return p


# ---------- 手绘元素 ----------

def soft_shadow(c, path, dx=10, dy=12, blur=10, a=0.16, color=(0.25, 0.18, 0.10)):
    p = fill(color, a)
    p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    c.save()
    c.translate(dx, dy)
    c.drawPath(path, p)
    c.restore()


@functools.lru_cache(maxsize=8192)
def _hatch_geom(seed, bx0, by0, bx1, by1, angle, spacing, length, density, gap, jitter):
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    R = math.hypot(bx1 - bx0, by1 - by0) / 2 + length
    ang = math.radians(angle)
    ux, uy = math.cos(ang), math.sin(ang)      # 线段方向
    vx, vy = -uy, ux                           # 行方向
    hp = skia.Path()
    rows = int(2 * R / spacing) + 1
    pitch = length * (1 + gap)
    for r in range(rows):
        v = -R + r * spacing
        off = hrand(seed, r, "o") * pitch
        cols = int(2 * R / pitch) + 2
        for k in range(cols):
            if hrand(seed, r, k, "d") > density:
                continue
            u = -R + off + k * pitch + srand(seed, r, k, "u") * pitch * jitter * 0.5
            L = length * (0.6 + 0.6 * hrand(seed, r, k, "l"))
            vv = v + srand(seed, r, k, "v") * spacing * jitter * 0.5
            x0 = cx + ux * u + vx * vv
            y0 = cy + uy * u + vy * vv
            hp.moveTo(x0, y0)
            hp.lineTo(x0 + ux * L, y0 + uy * L)
    return hp


def hatch(c, path, color, a=0.5, angle=-35, spacing=9.0, length=26.0, width=1.6, seed="h", density=0.85,
          gap=0.6, jitter=0.45):
    """排线纹理：在路径内部按旋转网格撒短线段（裁剪到路径）"""
    if a <= 0:
        return
    b = path.computeTightBounds()
    q = 8.0
    hp = _hatch_geom(str(seed), math.floor(b.left() / q) * q, math.floor(b.top() / q) * q,
                     math.ceil(b.right() / q) * q, math.ceil(b.bottom() / q) * q, angle, spacing, length, density,
                     gap, jitter)
    c.save()
    c.clipPath(path, doAntiAlias=True)
    c.drawPath(hp, stroke(color, lw(width) if ZK["z"] > 1 else width, a))
    c.restore()


def ink(c, path, w=4.2, a=1.0, color=INK):
    c.drawPath(path, stroke(color, lw(w), a))


def blob(c, pts, color, seed, outline=4.2, wob=1.6, shadow=True, hatch_kw=None, ink_color=INK, ink_a=1.0):
    """手绘色块：抖动外形 + 软阴影 + 底色 + 排线 + 墨线"""
    path = smooth_path(wobble(pts, seed, wob))
    if shadow:
        soft_shadow(c, path)
    c.drawPath(path, fill(color))
    if hatch_kw is not None:
        hatch(c, path, **{"seed": seed, **hatch_kw})
    if outline:
        ink(c, path, outline, ink_a, ink_color)
    return path


def line(c, pts, color=INK, w=4.0, a=1.0, seed=None, wob=1.2):
    if seed is not None:
        pts = wobble(pts, seed, wob, closed=False)
    c.drawPath(smooth_path(pts, closed=False), stroke(color, lw(w), a))


# ---------- 背景 ----------

def view_rect(c, pad=4.0):
    """当前画布可见区域（局部坐标）"""
    b = c.getLocalClipBounds()
    return skia.Rect(b.left() - pad, b.top() - pad, b.right() + pad, b.bottom() + pad)


@functools.lru_cache(maxsize=None)
def grain_img(seed=3, strength=1.0):
    """纸张颗粒：低频斑驳 + 高频细颗粒 + 零星墨点，做成半透明叠加层"""
    rng = np.random.default_rng(seed)
    hi = rng.normal(0, 1, (H, W)).astype(np.float32)
    lo = rng.normal(0, 1, (H // 24 + 2, W // 24 + 2)).astype(np.float32)
    lo = np.kron(lo, np.ones((24, 24), np.float32))[:H, :W]
    # 简单的盒式模糊让低频更柔
    k = 9
    for ax in (0, 1):
        lo = sum(np.roll(lo, i, axis=ax) for i in range(-k, k + 1)) / (2 * k + 1)
    g = hi * 0.55 + lo * 1.4
    g = (g - g.min()) / (g.max() - g.min())
    a = (np.abs(g - 0.5) * 2) ** 1.6 * 0.10 * strength
    dark = g < 0.5
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = np.where(dark[..., None], np.array([70, 55, 40], np.uint8), np.array([255, 252, 240], np.uint8))
    rgba[..., 3] = (a * 255).astype(np.uint8)
    # 零星墨点
    n = 420
    xs = rng.integers(0, W, n)
    ys = rng.integers(0, H, n)
    for x, y in zip(xs, ys):
        r = 1 if rng.random() < 0.8 else 2
        rgba[max(0, y - r // 2):y + r, max(0, x - r // 2):x + r] = (60, 45, 35, 90)
    # 预乘透明度
    rgba[..., :3] = (rgba[..., :3].astype(np.float32) * rgba[..., 3:4] / 255).astype(np.uint8)
    return skia.Image.fromarray(np.ascontiguousarray(rgba), colorType=skia.ColorType.kRGBA_8888_ColorType,
                                alphaType=skia.AlphaType.kPremul_AlphaType)


def paper_bg(c, glow=(330, 160), glow_a=0.55):
    """米色纸底，左上方有一团暖光（太阳那一侧）"""
    r = view_rect(c)
    c.drawRect(r, fill(PAPER))
    if glow_a > 0:
        sh = skia.GradientShader.MakeRadial(skia.Point(*glow), 1100, [C(PAPER_WARM, glow_a), C(PAPER_WARM, 0)])
        c.drawRect(r, skia.Paint(Shader=sh))


@functools.lru_cache(maxsize=None)
def streak_path(seed="rain", n=1500, angle=-62, Lmin=10, Lmax=24):
    """暗场里的屏幕斜纹：一层固定在屏幕上的短斜线"""
    p = skia.Path()
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)
    for i in range(n):
        x = hrand(seed, i, "x") * (W + 40) - 20
        y = hrand(seed, i, "y") * (H + 40) - 20
        L = Lmin + (Lmax - Lmin) * hrand(seed, i, "l")
        p.moveTo(x, y)
        p.lineTo(x + ux * L, y + uy * L)
    return p


def dark_bg(c, center=(960, 480), glow=1.0, base=DARK, hi=DARK_HI):
    """显微暗场：中心略亮的径向渐变"""
    r = view_rect(c)
    c.drawRect(r, fill(base))
    sh = skia.GradientShader.MakeRadial(skia.Point(*center), 1000, [C(hi, 0.85 * glow), C(hi, 0)])
    c.drawRect(r, skia.Paint(Shader=sh))


def dark_streaks(c, a=0.10, color=(0.55, 0.80, 0.55)):
    c.drawPath(streak_path(), stroke(color, 1.5, a))


def finish_paper(c, strength=1.0):
    c.drawImage(grain_img(3, strength), 0, 0)


# ---------- 文字 ----------

def _font_file(env, name):
    for cand in (os.environ.get(env), str(ROOT / "assets" / "fonts" / name), "C:/Windows/Fonts/" + name):
        if cand and Path(cand).exists():
            return cand
    return "C:/Windows/Fonts/" + name


FONT_FILES = {
    "sans": _font_file("NOTO_SANS_SC", "NotoSansSC-VF.ttf"),
    "serif": _font_file("NOTO_SERIF_SC", "NotoSerifSC-VF.ttf"),
}


@functools.lru_cache(maxsize=None)
def typeface(kind, weight=400):
    tf = skia.Typeface.MakeFromFile(FONT_FILES[kind])
    tag = (ord("w") << 24) | (ord("g") << 16) | (ord("h") << 8) | ord("t")
    coords = skia.FontArguments.VariationPosition.Coordinates(
        [skia.FontArguments.VariationPosition.Coordinate(tag, weight)])
    args = skia.FontArguments()
    args.setVariationDesignPosition(skia.FontArguments.VariationPosition(coords))
    return tf.makeClone(args)


def font(size, weight=500, kind="sans"):
    f = skia.Font(typeface(kind, weight), size)
    f.setSubpixel(True)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    return f


def text_w(s, f):
    return f.measureText(s)


def text(c, s, x, y, f, color=INK, a=1.0, align="left", outline=None, outline_w=0.0):
    """单行文字；y 是基线。outline 给描边色时先描边再填色"""
    w = f.measureText(s)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    if outline is not None and outline_w > 0:
        c.drawString(s, x, y, f, stroke(outline, outline_w, a))
    c.drawString(s, x, y, f, fill(color, a))
    return w


def chem(c, s, x, y, size, color=INK, a=1.0, align="center", weight=600, outline=None, outline_w=0.0):
    """化学式：数字自动下标，例如 CO2、H2O、O2"""
    f = font(size, weight)
    fs = font(size * 0.62, weight)
    parts = []
    for ch in s:
        parts.append((ch, fs if ch.isdigit() else f))
    w = sum(ff.measureText(ch) for ch, ff in parts)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    for ch, ff in parts:
        dy = size * 0.18 if ff is fs else 0
        if outline is not None and outline_w > 0:
            c.drawString(ch, x, y + dy, ff, stroke(outline, outline_w, a))
        c.drawString(ch, x, y + dy, ff, fill(color, a))
        x += ff.measureText(ch)
    return w


def tag(c, s, x, y, tx, ty, size=34, color=INK, a=1.0, line_color=None, dark=False):
    """引线标签：从 (x, y) 指向文字位置 (tx, ty)"""
    if a <= 0:
        return
    lc = line_color or (CREAM if dark else INK)
    f = font(size, 600)
    c.drawCircle(x, y, 5, fill(lc, a))
    w = f.measureText(s)
    end_x = tx - w / 2 - 12 if tx > x else tx + w / 2 + 12
    if abs(tx - x) < w / 2:
        end_x = tx
    c.drawPath(smooth_path([(x, y), (lerp(x, end_x, 0.5), lerp(y, ty - size * 0.35, 0.8)), (end_x, ty - size * 0.35)],
                           closed=False), stroke(lc, 2.4, a * 0.9))
    if dark:
        text(c, s, tx, ty, f, CREAM, a, "center", outline=DARK, outline_w=7)
    else:
        text(c, s, tx, ty, f, INK, a, "center", outline=PAPER, outline_w=7)


# ---------- 屏幕空间的标签 ----------
# 条目：dict(kind, text, x, y（锚点屏幕坐标）, dx, dy（文字相对锚点的屏幕偏移）, a, dark, size)
LABELS = []


def label_at(c, kind, s, x, y, dx, dy, a=1.0, dark=False, size=40):
    """登记一个标签：kind = "tag"（引线 + 文字）或 "chem"（化学式，文字放在锚点 + 偏移处）"""
    if a <= 0.01:
        return
    p = c.getTotalMatrix().mapXY(x, y)
    LABELS.append(dict(kind=kind, text=s, x=p.x(), y=p.y(), dx=dx, dy=dy, a=a, dark=dark, size=size))


def flush_labels(c):
    for L in LABELS:
        if L["kind"] == "tag":
            tag(c, L["text"], L["x"], L["y"], L["x"] + L["dx"], L["y"] + L["dy"], L["size"], a=L["a"], dark=L["dark"])
        else:
            col, oc = (CREAM, (0.06, 0.15, 0.10)) if L["dark"] else (INK, PAPER)
            chem(c, L["text"], L["x"] + L["dx"], L["y"] + L["dy"], L["size"], col, L["a"], outline=oc, outline_w=8)
    LABELS.clear()


# ---------- 画布 ----------

def new_canvas():
    surf = skia.Surface(W, H)
    return surf, surf.getCanvas()


def snapshot(surf):
    arr = surf.makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
    return arr[..., :3]

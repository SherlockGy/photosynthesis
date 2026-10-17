# 各层场景：每层都在自己的 1920×1080 设计坐标里绘制，由镜头系统（scenes.py）负责嵌套与推进
# 层级：earth 地球 → room 盆栽与式子 → leaf 叶面叶脉 → epi 叶背表皮与气孔 → cell 叶肉细胞 → chloro 叶绿体
#       → mem 类囊体膜（光反应；膜的上方是基质，卡尔文循环画在同一层的 y < 0 区域）
# 每层的状态（动画进度、标签透明度）由 scenes.py 按时间线算好，放在 st 字典里传进来
import functools
import math

import numpy as np
import skia
from scipy.spatial import Voronoi

from draw import (BRASS, C_GRAY, CHL, CREAM, DARK, DARK_MID, EPI, EPI_LINE, GOLD, GRANA, GRANA_TOP, GUARD, H,
                  H_WHITE, INK, LEAF, LEAF_DK, LEAF_LT, MEM, MEM_LT, O_RED, POT, POT_DK, PROT, SUN, SUN_RAY, VEIN, W,
                  WATER, WATER_DK, WHITE, WOOD, clamp, dark_bg, ease_io, ellipse_pts, fill, font, hatch, hrand, ink,
                  label_at, lerp, line, lw, paper_bg, poly_path, rrect_pts, shade, smooth_path, soft_shadow, srand,
                  stroke, text, view_rect, wobble, xform)

SUGAR_C = (0.95, 0.89, 0.72)      # 糖、淀粉粒的米黄
RED_LIGHT = (0.92, 0.28, 0.22)
BLUE_LIGHT = (0.38, 0.42, 0.95)
GREEN_LIGHT = (0.45, 0.95, 0.45)
OCEAN = (0.36, 0.62, 0.74)
OCEAN_DK = (0.22, 0.45, 0.58)
SAND = (0.86, 0.74, 0.50)
NIGHT = (0.07, 0.09, 0.16)


def ease_out_q(x):
    x = clamp(x)
    return 1 - (1 - x) ** 2


def glow(c, x, y, r, color, a=1.0):
    if a <= 0:
        return
    sh = skia.GradientShader.MakeRadial(skia.Point(x, y), r, [skia.Color4f(*color, 0.55 * a), skia.Color4f(*color, 0)])
    c.drawCircle(x, y, r, skia.Paint(Shader=sh, AntiAlias=True))


# ======================================================================
# 通用小部件
# ======================================================================

def leaf_poly(L, Wd, n=36, asym=0.08):
    """叶片轮廓（局部坐标：叶柄在原点，叶尖在 (L, 0)）"""
    up, dn = [], []
    for i in range(n + 1):
        s = i / n
        f = math.sin(math.pi * s ** 0.72) ** 0.95 * (1 - 0.12 * s)
        up.append((L * s, -Wd * f * (1 + asym)))
        dn.append((L * s, Wd * f * (1 - asym)))
    return up + dn[::-1][1:-1]


@functools.lru_cache(maxsize=None)
def leaf_path(L, Wd, seed):
    return smooth_path(wobble(leaf_poly(L, Wd), seed, 1.4, step=12))


def leaf(c, x, y, ang, L, Wd, seed, sparkle=0.0, t=0.0):
    """一片手绘叶子：底色、两组顺着侧脉的排线、主脉侧脉、墨线"""
    c.save()
    c.translate(x, y)
    c.rotate(ang)
    path = leaf_path(L, Wd, seed)
    soft_shadow(c, path, 8, 10, 9, 0.14)
    c.drawPath(path, fill(LEAF))
    for half, ang_h in ((-1, -48), (1, 48)):
        c.save()
        c.clipRect(skia.Rect(-10, -Wd * 2 if half < 0 else 0, L + 10, 0 if half < 0 else Wd * 2))
        hatch(c, path, LEAF_LT, 0.55, ang_h, spacing=max(6, L / 46), length=max(10, L / 16), width=1.7,
              seed=f"{seed}{half}", density=0.8)
        c.restore()
    c.drawPath(smooth_path([(0, 0), (L * 0.5, -Wd * 0.03), (L * 0.97, 0)], closed=False), stroke(VEIN, max(2.5, L / 90)))
    for i in range(7):
        s = 0.12 + i * 0.115
        for side in (-1, 1):
            f2 = math.sin(math.pi * min(1, s + 0.15) ** 0.72) ** 0.95
            pts = [(L * s, 0), (L * (s + 0.08), side * Wd * f2 * 0.45), (L * (s + 0.17), side * Wd * f2 * 0.82)]
            c.drawPath(smooth_path(pts, closed=False), stroke(VEIN, max(1.6, L / 170)))
    ink(c, path, 4.4)
    if sparkle > 0:
        for k in range(3):
            sx, sy = L * (0.35 + 0.2 * k), (-1) ** k * Wd * 0.35
            ph = (math.sin(t * 2.2 + k * 2.1) + 1) / 2
            r = 7 + 4 * ph
            sp = skia.Path()
            sp.moveTo(sx, sy - r)
            sp.quadTo(sx, sy, sx + r, sy)
            sp.quadTo(sx, sy, sx, sy + r)
            sp.quadTo(sx, sy, sx - r, sy)
            sp.quadTo(sx, sy, sx, sy - r)
            c.drawPath(sp, fill(GOLD, sparkle * (0.5 + 0.5 * ph)))
    c.restore()


def molecule(c, x, y, kind, r=10.0, ang=0.0, a=1.0, mark=0.0):
    """小分子：CO2（红-灰-红一字排开）、H2O（红 + 两个白）、O2（两个红）、H（一个白）
    mark > 0 时给氧原子套一圈金色标记（同位素标记）"""
    if a <= 0:
        return
    c.save()
    c.translate(x, y)
    c.rotate(ang)
    if kind == "CO2":
        atoms = [(-1.75 * r, 0, r, O_RED), (1.75 * r, 0, r, O_RED), (0, 0, r * 1.05, C_GRAY)]
    elif kind == "H2O":
        atoms = [(-1.05 * r, 0.85 * r, r * 0.62, H_WHITE), (1.05 * r, 0.85 * r, r * 0.62, H_WHITE), (0, 0, r, O_RED)]
    elif kind == "O2":
        atoms = [(-0.8 * r, 0, r, O_RED), (0.8 * r, 0, r, O_RED)]
    else:
        atoms = [(0, 0, r, H_WHITE)]
    for ax, ay, ar, col in atoms:
        c.drawCircle(ax, ay, ar + lw(2.2), fill(INK, a))
        c.drawCircle(ax, ay, ar, fill(col, a))
        c.drawCircle(ax - ar * 0.3, ay - ar * 0.32, ar * 0.32, fill(WHITE, a * 0.35))
        if mark > 0 and col == O_RED:
            c.drawCircle(ax, ay, ar * 1.55, stroke(GOLD, lw(3.2), a * mark))
            c.drawCircle(ax, ay, ar * 1.9, stroke(GOLD, lw(1.6), a * mark * 0.5))
    c.restore()


def hexagon(c, x, y, r, a=1.0, ang=0.0, glow_a=0.0, color=SUGAR_C):
    """糖分子（六元环）"""
    if a <= 0:
        return
    if glow_a > 0:
        glow(c, x, y, r * 2.4, (1, 0.95, 0.6), glow_a * a)
    pts = [(x + r * math.cos(math.radians(ang + 60 * k + 30)), y + r * math.sin(math.radians(ang + 60 * k + 30)))
           for k in range(6)]
    p = poly_path(pts)
    c.drawPath(p, fill(color, a))
    c.drawPath(p, stroke(INK, lw(2.6), a))
    c.drawCircle(x, y, r * 0.42, stroke(shade(color, -0.35), lw(2.2), a))


def atp(c, x, y, r=15, a=1.0, glow_a=0.8):
    """ATP：三个磷酸连起来 + 光晕（表示存着能量）"""
    if a <= 0:
        return
    glow(c, x, y, r * 4.4, (0.6, 1, 0.5), glow_a * a)
    for k in range(3):
        c.drawCircle(x + (k - 1) * r * 2, y, r, fill(SUGAR_C, a))
        c.drawCircle(x + (k - 1) * r * 2, y, r, stroke(INK, lw(2.4), a * 0.8))


def h_carrier(c, x, y, r=12, a=1.0):
    """带着能量的氢（发光的白色小球）"""
    if a <= 0:
        return
    glow(c, x, y, r * 3.4, (0.9, 1, 0.85), 0.9 * a)
    molecule(c, x, y, "H", r, a=a)


# ======================================================================
# room：盆栽、太阳、放大镜、式子
# ======================================================================

ROOM_GROUND = 830
BIG_LEAF = (800, 500, -18, 430, 118)
LENS = (1010, 455, 150, 32)            # 放大镜停在大叶子上的位置：镜片中心、半径、手柄角度
FORMULA_POS = (1330, 205, 0.9)         # 式子中心、基线、缩放
FORMULA_ITEMS = ["二氧化碳", "+", "水", "→", "有机物", "+", "氧气"]


def sun(c, x, y, r, t, a=1.0):
    path = smooth_path(wobble(ellipse_pts(x, y, r, r, 40), "sun", 1.2))
    for i in range(12):
        ang = i / 12 * 2 * math.pi + 0.08 * math.sin(t * 0.8)
        L0 = r + 18
        L1 = r + 40 + 8 * math.sin(t * 2 + i)
        line(c, [(x + math.cos(ang) * L0, y + math.sin(ang) * L0), (x + math.cos(ang) * L1, y + math.sin(ang) * L1)],
             SUN_RAY, 3.6, a)
    c.drawPath(path, fill(SUN, a))
    hatch(c, path, SUN_RAY, 0.55 * a, -40, spacing=7, length=16, width=1.5, seed="sunh")
    ink(c, path, 4.0, a)


def sun_beam(c, x0, y0, x1, y1, t, a=0.5):
    """从太阳斜照下来的一束细排线，缓缓流动"""
    ang = math.atan2(y1 - y0, x1 - x0)
    L = math.hypot(x1 - x0, y1 - y0)
    p = skia.Path()
    for i in range(26):
        u = (hrand("beam", i) * L * 0.85 + t * 40) % (L * 0.85) + L * 0.08
        v = srand("beam", i, "v") * (18 + u * 0.10)
        l = 30 + 50 * hrand("beam", i, "l")
        x = x0 + math.cos(ang) * u - math.sin(ang) * v
        y = y0 + math.sin(ang) * u + math.cos(ang) * v
        p.moveTo(x, y)
        p.lineTo(x + math.cos(ang) * l, y + math.sin(ang) * l)
    c.drawPath(p, stroke(SUN_RAY, 1.6, a))


def pot(c, x, y):
    body = [(x - 80, y + 30), (x + 80, y + 30), (x + 64, ROOM_GROUND), (x - 64, ROOM_GROUND)]
    rim = rrect_pts(x - 94, y, 188, 38, 6, 3)
    for pts, seed, k in ((body, "potb", 0), (rim, "potr", 0.06)):
        path = smooth_path(wobble(pts, seed, 1.2, step=10))
        soft_shadow(c, path, 8, 6, 8, 0.12)
        c.drawPath(path, fill(shade(POT, k)))
        hatch(c, path, POT_DK, 0.5, 60, spacing=8, length=18, width=1.6, seed=seed + "1", density=0.6)
        hatch(c, path, POT_DK, 0.35, -30, spacing=9, length=14, width=1.4, seed=seed + "2", density=0.4)
        ink(c, path, 4.2)
    c.drawPath(smooth_path(wobble(ellipse_pts(x, y + 4, 82, 9, 24), "soil", 0.8)), fill((0.35, 0.24, 0.17)))


def magnifier(c, x, y, R, ang, inner=None):
    """放大镜：inner(c) 画镜片里的内容（已裁剪到镜片圆内）"""
    c.save()
    c.translate(x, y)
    c.rotate(ang)
    hp = smooth_path(wobble(rrect_pts(R + 6, -11, 46, 22, 5, 2), "neck", 0.6, step=8))
    c.drawPath(hp, fill(BRASS))
    ink(c, hp, 3.6)
    handle = smooth_path(wobble(rrect_pts(R + 48, -17, 300, 34, 15, 4), "hdl", 1.0, step=10))
    soft_shadow(c, handle, 10, 12, 10, 0.14)
    c.drawPath(handle, fill(WOOD))
    hatch(c, handle, (0.30, 0.17, 0.09), 0.6, 4, spacing=6, length=40, width=1.5, seed="hdl", density=0.7)
    ink(c, handle, 4.2)
    c.restore()
    lens = skia.Path()
    lens.addCircle(x, y, R)
    c.save()
    c.clipPath(lens, doAntiAlias=True)
    if inner is not None:
        inner(c)
    else:
        c.drawCircle(x, y, R, fill((0.93, 0.94, 0.92)))
    c.drawArc(skia.Rect(x - R * 0.78, y - R * 0.78, x + R * 0.78, y + R * 0.78), 200, 50, False,
              stroke(WHITE, R * 0.07, 0.55))
    c.drawCircle(x, y, R, fill(WHITE, 0.06))
    c.restore()
    c.drawCircle(x, y, R + 11, stroke(BRASS, 22))
    c.drawArc(skia.Rect(x - R - 11, y - R - 11, x + R + 11, y + R + 11), 210, 70, False, stroke(shade(BRASS, 0.45), 6, 0.8))
    ink(c, smooth_path(wobble(ellipse_pts(x, y, R + 22, R + 22, 60), "ring_o", 0.8)), 4.4)
    ink(c, smooth_path(wobble(ellipse_pts(x, y, R, R, 60), "ring_i", 0.6)), 3.4)


def plant(c, t):
    sw = 0.8 * math.sin(t * 0.9)
    px = 760
    pot(c, px, 700)
    stem = [(px, 706), (px + 6, 600), (px + 14, 480), (px + 12, 360), (px + 4, 250)]
    stem = [(x + sw * (706 - y) / 120, y) for x, y in stem]
    c.drawPath(smooth_path(stem, closed=False), stroke(INK, 13))
    c.drawPath(smooth_path(stem, closed=False), stroke(LEAF_DK, 6.5))
    leaf(c, px + 2, 620, 162 + sw, 200, 64, "lf1")
    leaf(c, px + 10, 400, 196 + sw, 230, 72, "lf2")
    leaf(c, px + 4, 262, -70 + sw * 1.5, 170, 54, "lf3")
    bx, by, ba, bl, bw = BIG_LEAF
    leaf(c, bx, by, ba, bl, bw, "big", sparkle=1.0, t=t)


def formula_layout(scale):
    """式子里每一项的（左边 x 偏移、宽度），相对式子中心"""
    s = scale
    f = font(54 * s, 700)
    widths = [220 * s if it == "→" else f.measureText(it) for it in FORMULA_ITEMS]
    gap = 28 * s
    total = sum(widths) + gap * (len(widths) - 1)
    out = []
    x = -total / 2
    for wd in widths:
        out.append((x, wd))
        x += wd + gap
    return out


def _quad(pts, u):
    (x0, y0), (x1, y1), (x2, y2) = pts
    a, b, cc = (1 - u) ** 2, 2 * (1 - u) * u, u * u
    return (a * x0 + b * x1 + cc * x2, a * y0 + b * y1 + cc * y2)


def _arc_note(c, x0, x1, y0, bulge, k, color, s, label, label_dy):
    """式子上的弧形箭头 + 注释，k=0~1 为画出的进度"""
    pts = [(x0, y0), ((x0 + x1) / 2, y0 + bulge), (x1, y0)]
    prog = ease_io(k)
    n = 30
    curve = [_quad(pts, i / n * prog) for i in range(n + 1)]
    line(c, curve, color, 4.2 * s, 1.0)
    if prog > 0.95:
        # 箭头沿曲线终点的切线方向：二次贝塞尔在 u 处的导数 2(1-u)(P1-P0) + 2u(P2-P1)
        ex, ey = curve[-1]
        u = prog
        (px0, py0), (px1, py1), (px2, py2) = pts
        tx = 2 * (1 - u) * (px1 - px0) + 2 * u * (px2 - px1)
        ty = 2 * (1 - u) * (py1 - py0) + 2 * u * (py2 - py1)
        tl = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tl, ty / tl
        wing = 20 * s
        wings = []
        for sign in (1, -1):
            a = math.radians(152 * sign)
            wings.append((ex + wing * (tx * math.cos(a) - ty * math.sin(a)),
                          ey + wing * (tx * math.sin(a) + ty * math.cos(a))))
        # 两翼用直线折线画（平滑曲线会把尖角磨圆往回收，箭杆末端就会戳出来）
        c.drawPath(poly_path([wings[0], (ex, ey), wings[1]], closed=False), stroke(color, lw(4.2 * s)))
    fs = font(38 * s, 700)
    text(c, label, (x0 + x1) / 2, y0 + bulge + label_dy * s, fs, color, clamp(k * 2 - 1), "center")


def formula(c, x, y, scale=1.0, alphas=None, hi=None, notes=None):
    """光合作用的式子：二氧化碳 + 水 —(光能 / 叶绿体)→ 有机物 + 氧气
    alphas：每一项的透明度（逐项出现）；hi：{项序号或"光能": 高亮强度}；notes：两条注释箭头的进度"""
    s = scale
    f = font(54 * s, 700)
    fs = font(30 * s, 600)
    hi = hi or {}
    lay = formula_layout(scale)
    alphas = alphas or [1.0] * len(FORMULA_ITEMS)
    for i, (it, (ox, wd), a) in enumerate(zip(FORMULA_ITEMS, lay, alphas)):
        if a <= 0:
            continue
        cx = x + ox
        k = hi.get(i, 0.0)
        if it == "→":
            yy = y - 18 * s
            kl = hi.get("光能", 0.0)
            if kl > 0:
                hp = smooth_path(wobble(rrect_pts(cx + wd / 2 - 46 * s, yy - 58 * s, 92 * s, 40 * s, 10 * s, 2), "hlg", 1.2))
                c.drawPath(hp, fill(GOLD, 0.6 * a * kl))
            line(c, [(cx + 6 * s, yy), (cx + wd * 0.5, yy - 3 * s), (cx + wd - 8 * s, yy)], INK, 4.4 * s, a, seed="arw")
            line(c, [(cx + wd - 30 * s, yy - 14 * s), (cx + wd - 6 * s, yy), (cx + wd - 30 * s, yy + 14 * s)], INK, 4.4 * s, a)
            text(c, "光能", cx + wd / 2, yy - 22 * s, fs, SUN_RAY, a, "center")
            text(c, "叶绿体", cx + wd / 2, yy + 46 * s, fs, LEAF_DK, a, "center")
        else:
            if k > 0:
                hp = smooth_path(wobble(rrect_pts(cx - 8 * s, y - 22 * s, wd + 16 * s, 30 * s, 8 * s, 2), f"hl{i}", 1.5))
                c.drawPath(hp, fill(GOLD, 0.6 * a * k))
            text(c, it, cx, y, f, INK, a)
    notes = notes or {}
    k = notes.get("o_from_w", 0.0)
    if k > 0:
        (wx, ww), (ox_, ow) = lay[2], lay[6]
        _arc_note(c, x + wx + ww / 2, x + ox_ + ow / 2, y + 30 * s, 100 * s, k, (0.20, 0.50, 0.75), s, "氧来自水", 44)
    k = notes.get("c_to_s", 0.0)
    if k > 0:
        (cx_, cw), (sx_, sw_) = lay[0], lay[4]
        _arc_note(c, x + cx_ + cw / 2, x + sx_ + sw_ / 2, y - 66 * s, -66 * s, k, LEAF_DK, s, "碳合成糖", -12)


def draw_room(c, t, st):
    """st: mag=(x, y, R, 角度) 或 None；lens(c, x, y, R)；title_a；f_alpha（每项）；hi；notes"""
    paper_bg(c)
    sun(c, 300, 175, 58, t)
    sun_beam(c, 340, 215, 820, 470, t, 0.4)
    c.drawPath(smooth_path(wobble([(140, ROOM_GROUND), (1780, ROOM_GROUND)], "gnd", 1.0, closed=False), closed=False),
               stroke((0.55, 0.50, 0.44), 2.2, 0.7))
    plant(c, t)
    fx, fy, fsc = FORMULA_POS
    ta = st.get("title_a", 0.0)
    if ta > 0:
        f = font(76, 800)
        w = text(c, "光合作用", fx, fy - 105, f, INK, ta, "center")
        k = st.get("title_line", 1.0)
        if k > 0:
            line(c, [(fx - w / 2, fy - 88), (fx - w / 2 + w * k, fy - 86)], LEAF_DK, 6, ta, seed="ttl")
    formula(c, fx, fy, fsc, st.get("f_alpha"), st.get("hi"), st.get("notes"))
    mag = st.get("mag")
    if mag is not None:
        mx, my, R, ang = mag
        lens = st.get("lens")
        magnifier(c, mx, my, R, ang, (lambda cc: lens(cc, mx, my, R)) if lens else None)


# ======================================================================
# leaf：叶面特写，叶脉里有水在流（内容覆盖范围大于画面，镜头经过镜片和拉远时不露底）
# ======================================================================

LEAF_MAIN = [(-500, 960), (0, 820), (500, 660), (1100, 480), (1700, 330), (2400, 150)]


def _main_y(x):
    return float(np.interp(x, [p[0] for p in LEAF_MAIN], [p[1] for p in LEAF_MAIN]))


@functools.lru_cache(maxsize=None)
def leaf_veins():
    """侧脉 + 细脉网（细脉用 Voronoi 边，像真实叶片的网眼）"""
    sides = []
    for i in range(-2, 9):
        x0 = -300 + i * 300
        y0 = _main_y(x0)
        for side in (-1, 1):
            pts = [(x0, y0)]
            for j in range(1, 7):
                u = j / 6
                pts.append((x0 + 560 * u + 80 * u * u, y0 + side * (900 * u - 200 * u * u) - 160 * u * (side < 0)))
            sides.append(smooth_path(pts, closed=False))
    rng = np.random.default_rng(11)
    pts = rng.uniform([-700, -900], [2700, 2000], (900, 2))
    vor = Voronoi(pts)
    minor = skia.Path()
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        p, q = vor.vertices[a], vor.vertices[b]
        if np.all(np.abs(p) < 5000) and np.all(np.abs(q) < 5000) and np.hypot(*(p - q)) < 400:
            pp = wobble([(float(p[0]), float(p[1])), (float(q[0]), float(q[1]))], f"mn{a}_{b}", 2.0, step=20, closed=False)
            minor.addPath(smooth_path(pp, closed=False))
    return sides, minor


# 推进叶背表皮的位置：两条侧脉之间的网眼里
EPI_ANCHOR = (1225, 760)


def water(c, path, w, t, seed, a=1.0):
    """叶脉中的水：浅蓝芯 + 沿管道移动的亮短划"""
    c.drawPath(path, stroke(WATER_DK, w + 4, 0.55 * a))
    c.drawPath(path, stroke(WATER, w, 0.9 * a))
    dash = stroke(WHITE, max(2.0, w * 0.28), 0.85 * a)
    period = 70.0
    dash.setPathEffect(skia.DashPathEffect.Make([18, period - 18], -(t * 160 + hrand(seed) * period) % period))
    c.drawPath(path, dash)


def draw_leaf(c, t, st):
    r = view_rect(c)
    c.drawRect(r, fill(LEAF))
    bg = skia.Path()
    bg.addRect(skia.Rect(-800, -1000, 2800, 2100))
    hatch(c, bg, LEAF_DK, 0.45, -60, spacing=9, length=20, width=1.8, seed="lfbg", density=0.8)
    hatch(c, bg, LEAF_LT, 0.30, 25, spacing=12, length=14, width=1.5, seed="lfbg2", density=0.55)
    sides, minor = leaf_veins()
    c.drawPath(minor, stroke(LEAF_DK, 7.0, 0.45))
    c.drawPath(minor, stroke(VEIN, 3.0, 0.6))
    for i, sp in enumerate(sides):
        if not sp.getBounds().intersects(r):
            continue
        c.drawPath(sp, stroke(LEAF_DK, 44, 0.55))
        c.drawPath(sp, stroke(VEIN, 30))
        water(c, sp, 12, t, f"s{i}")
    mpth = smooth_path(LEAF_MAIN, closed=False)
    c.drawPath(mpth, stroke(LEAF_DK, 96, 0.55))
    c.drawPath(mpth, stroke(VEIN, 76))
    c.drawPath(mpth, stroke(shade(VEIN, 0.35), 18, 0.7))
    water(c, mpth, 22, t, "main")
    label_at(c, "tag", "叶脉", 1400, _main_y(1400), 150, -190, st.get("lab_vein", 0.0))
    label_at(c, "chem", "H2O", 820, _main_y(820) - 40, 0, -40, st.get("lab_h2o", 0.0))


# ======================================================================
# epi：叶背表皮，拼图状的表皮细胞和气孔；正中的气孔是镜头要钻进去的那个
# ======================================================================

MAIN_STOMA = (960, 560, 0, 1.25)
STOMATA = [MAIN_STOMA, (1500, 250, 20, 0.9), (1560, 860, -14, 1.0), (330, 200, 30, 0.85), (380, 820, -8, 0.9),
           (1880, 560, -40, 0.85), (-200, 500, 15, 0.9), (2200, 150, 25, 0.9), (2300, 1000, -20, 0.95),
           (-300, -150, 10, 0.85), (900, -200, -30, 0.9), (1000, 1300, 20, 0.9), (-250, 1250, 35, 0.9),
           (2050, -250, 5, 0.85), (2600, 520, -25, 0.9), (-650, 700, 0, 0.9)]


@functools.lru_cache(maxsize=None)
def epi_cells():
    """拼图状表皮细胞：Voronoi 边加正弦波纹"""
    rng = np.random.default_rng(5)
    gx, gy = np.meshgrid(np.arange(-1000, 2950, 165), np.arange(-700, 1800, 150))
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    pts += rng.uniform(-55, 55, pts.shape)
    keep = np.ones(len(pts), bool)
    for sx, sy, _a, sc in STOMATA:
        keep &= np.hypot(pts[:, 0] - sx, pts[:, 1] - sy) > 125 * sc
    pts = pts[keep]
    vor = Voronoi(pts)
    edges = skia.Path()
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        p, q = vor.vertices[a], vor.vertices[b]
        L = float(np.hypot(*(p - q)))
        if L < 4 or np.any(np.abs(p) > 4000) or np.any(np.abs(q) > 4000):
            continue
        n = max(6, int(L / 9))
        d = (q - p) / L
        nrm = np.array([-d[1], d[0]])
        waves = 1 + int(L / 70)
        ph = hrand("ep", a, b) * 0.6
        amp = min(16, L * 0.12)
        seg = []
        for i in range(n + 1):
            u = i / n
            off = amp * math.sin(u * math.pi * waves * 2 + ph) * math.sin(u * math.pi)
            pt = p + (q - p) * u + nrm * off
            seg.append((float(pt[0]), float(pt[1])))
        edges.addPath(smooth_path(seg, closed=False))
    regions = []
    for i, ri in enumerate(vor.point_region):
        reg = vor.regions[ri]
        if -1 in reg or len(reg) < 3:
            continue
        poly = [(float(vor.vertices[v][0]), float(vor.vertices[v][1])) for v in reg]
        if max(abs(x) for x, y in poly) > 4000 or max(abs(y) for x, y in poly) > 4000:
            continue
        regions.append((poly_path(poly), hrand("cellc", i)))
    return edges, regions


def pore_dims(openness):
    return 62 + 8 * openness, 6 + 20 * openness


def pore_path(openness):
    """正中气孔的孔口（表皮坐标）"""
    sx, sy, ang, sc = MAIN_STOMA
    prx, pry = pore_dims(openness)
    m = skia.Matrix()
    m.setTranslate(sx, sy)
    m.preRotate(ang)
    m.preScale(sc, sc)
    p = skia.Path()
    p.addOval(skia.Rect(-(prx - 2), -max(1.5, pry - 2), prx - 2, max(1.5, pry - 2)))
    return xform(p, m)


def stoma(c, x, y, ang, sc, openness, seed):
    """气孔：两个肾形保卫细胞围成，中间是孔"""
    c.save()
    c.translate(x, y)
    c.rotate(ang)
    c.scale(sc, sc)
    rx, ry = 105, 64
    c.drawPath(smooth_path(ellipse_pts(0, 0, rx + 26, ry + 24, 40)), fill(shade(EPI, 0.45)))
    ink(c, smooth_path(wobble(ellipse_pts(0, 0, rx + 26, ry + 24, 40), seed + "h", 1.0)), 2.6, 0.35, EPI_LINE)
    prx, pry = pore_dims(openness)
    for side in (-1, 1):
        outer = [(rx * math.cos(u), side * ry * abs(math.sin(u))) for u in np.linspace(0, math.pi, 22)]
        inner = [(prx * math.cos(u), side * pry * abs(math.sin(u))) for u in np.linspace(math.pi, 0, 18)]
        path = smooth_path(wobble(outer + inner, f"{seed}g{side}", 1.0, step=9))
        soft_shadow(c, path, 4, 6, 6, 0.12)
        c.drawPath(path, fill(GUARD))
        hatch(c, path, LEAF_LT, 0.45, 20 * side, spacing=7, length=14, width=1.4, seed=f"{seed}{side}")
        for k in range(7):
            u = 0.15 + 0.7 * k / 6
            cx = math.cos(math.pi * u) * (rx + prx) / 2
            cy = side * (ry + pry) / 2 * math.sin(math.pi * u) * 0.95
            c.drawCircle(cx, cy, 6.5, fill(LEAF_DK))
            c.drawCircle(cx - 1.5, cy - 1.5, 2.4, fill(LEAF_LT, 0.7))
        ink(c, path, 3.6)
    c.drawPath(smooth_path(ellipse_pts(0, 0, prx - 2, max(1.5, pry - 2), 30)), fill(DARK))
    c.restore()


def draw_epi(c, t, st):
    """st: open_main（正中气孔开度）, co2（二氧化碳透明度）, lab_stoma, lab_guard, lab_co2"""
    r = view_rect(c)
    c.drawRect(r, fill(EPI))
    edges, regions = epi_cells()
    for path, k in regions:
        if path.getBounds().intersects(r):
            c.drawPath(path, fill(shade(EPI, (k - 0.5) * 0.05)))
    c.drawPath(edges, stroke(EPI_LINE, 7, 0.75))
    c.drawPath(edges, stroke(shade(EPI, 0.5), 2.2, 0.8))
    bg = skia.Path()
    bg.addRect(skia.Rect(-1100, -800, 3050, 1900))
    hatch(c, bg, EPI_LINE, 0.25, -60, spacing=22, length=9, width=1.4, seed="epih", density=0.5)
    base_open = 0.55 + 0.35 * math.sin(t * 0.9)
    for i, (sx, sy, ang, sc) in enumerate(STOMATA):
        op = st.get("open_main", 0.9) if i == 0 else clamp(base_open + 0.2 * math.sin(t * 0.7 + i))
        stoma(c, sx, sy, ang, sc, op, f"st{i}")
    co2 = st.get("co2", 1.0)
    if co2 > 0:
        # 二氧化碳分子从四周飘向气孔，到孔口时缩小消失（钻进去了）
        for i in range(30):
            sx, sy, ang, sc = STOMATA[i % 6]
            ph = (t * 0.16 + hrand("co2", i)) % 1.0
            a0 = hrand("co2a", i) * 6.283
            d = 520 * (1 - ph) + 10
            x = sx + math.cos(a0) * d + 40 * math.sin(t * 0.7 + i) * (1 - ph)
            y = sy + math.sin(a0) * d * 0.8
            fade = clamp(ph * 5) * clamp((1 - ph) * 7)
            molecule(c, x, y, "CO2", 9 * (0.4 + 0.6 * clamp((1 - ph) * 4)), a0 * 57 + 90 + t * 20, co2 * fade)
    sx, sy, _a, sc = MAIN_STOMA
    label_at(c, "tag", "气孔", sx + 40, sy, 330, 200, st.get("lab_stoma", 0.0))
    label_at(c, "tag", "保卫细胞", sx - 70, sy - 52, -320, -210, st.get("lab_guard", 0.0))
    label_at(c, "chem", "CO2", sx - 470, sy - 180, 0, -40, st.get("lab_co2", 0.0) * co2)


# ======================================================================
# cell：叶肉细胞，叶绿体沿着细胞壁排开；正中细胞顶部那颗是镜头要推进的叶绿体
# ======================================================================

CELLS = [(900, 600, 600, 330, 0), (-260, 120, 420, 300, 1), (-200, 1080, 460, 280, 2), (2100, 180, 430, 320, 3),
         (2100, 1060, 450, 300, 4), (900, -150, 560, 240, 5), (930, 1400, 540, 230, 6), (2900, 620, 380, 400, 7),
         (-1050, 600, 400, 420, 8)]
CHL_RX, CHL_RY = 64, 31


def chl_positions(cx, cy, rx, ry, n, k0):
    """沿细胞壁内侧排开的叶绿体：(x, y, 切线角度)"""
    out = []
    for i in range(n):
        a = math.radians(k0 + 360 * i / n)
        x = cx + rx * 0.86 * math.cos(a)
        y = cy + ry * 0.86 * math.sin(a)
        tx, ty = -rx * math.sin(a), ry * math.cos(a)
        out.append((x, y, math.degrees(math.atan2(ty, tx))))
    return out


def chl_target():
    """镜头要推进的那颗叶绿体：正中细胞顶部，横着放"""
    cx, cy, rx, ry, _i = CELLS[0]
    return cx, cy - ry * 0.86, 0.0


def small_chloroplast(c, x, y, ang, seed, hi=0.0):
    c.save()
    c.translate(x, y)
    c.rotate(ang)
    path = smooth_path(wobble(ellipse_pts(0, 0, CHL_RX, CHL_RY, 30), seed, 1.0, step=8))
    if hi > 0:
        glow(c, 0, 0, CHL_RX * 1.8, (0.6, 1.0, 0.5), hi)
    c.drawPath(path, fill(lerp(GRANA, MEM_LT, 0.15 * hi)))
    hatch(c, path, MEM_LT, 0.35, -60, spacing=6, length=10, width=1.2, seed=seed)
    for k in range(5):
        gx = (k - 2) * CHL_RX * 0.34
        gy = srand(seed, k) * CHL_RY * 0.3
        c.drawPath(smooth_path(ellipse_pts(gx, gy, CHL_RX * 0.12, CHL_RY * 0.2, 12)), fill(GRANA_TOP))
    c.drawPath(path, stroke(MEM_LT, lw(2.4)))
    c.restore()


def draw_cell(c, t, st):
    """st: lab_cell, lab_chl, hi_chl（目标叶绿体发光）, co2"""
    dark_bg(c, (900, 500), glow=0.7, base=(0.05, 0.14, 0.09), hi=(0.10, 0.25, 0.16))
    vr = view_rect(c)
    tx, ty, _ = chl_target()
    for cx, cy, rx, ry, i in CELLS:
        if not skia.Rect(cx - rx - 40, cy - ry - 40, cx + rx + 40, cy + ry + 40).intersects(vr):
            continue
        pts = [(cx + rx * math.cos(a) * (1 + 0.04 * math.sin(3 * a + i)), cy + ry * math.sin(a)) for a in
               np.linspace(0, 2 * math.pi, 48, endpoint=False)]
        path = smooth_path(wobble(pts, f"cell{i}", 3, step=24))
        soft_shadow(c, path, 10, 14, 18, 0.35, (0, 0, 0))
        c.drawPath(path, fill((0.11, 0.29, 0.17)))
        hatch(c, path, MEM_LT, 0.12, -62, spacing=15, length=18, width=1.5, seed=f"ch{i}", density=0.6)
        c.drawPath(smooth_path(wobble(ellipse_pts(cx, cy, rx * 0.62, ry * 0.5, 36), f"vac{i}", 4, step=24)),
                   fill((0.15, 0.36, 0.22), 0.7))
        c.drawPath(path, stroke(MEM, lw(9)))
        c.drawPath(path, stroke(INK, lw(2.4), 0.6))
        n = max(10, int((rx + ry) / 55))
        for j, (x, y, ang) in enumerate(chl_positions(cx, cy, rx, ry, n, -90 if i == 0 else 17 * i)):
            if i == 0 and j == 0:
                small_chloroplast(c, tx, ty, 0.0, "target", st.get("hi_chl", 0.0))
            else:
                small_chloroplast(c, x, y, ang, f"sc{i}_{j}")
    co2 = st.get("co2", 1.0)
    for k in range(14):
        ph = (t * 0.07 + hrand("cc", k)) % 1.0
        x = -200 + 2300 * hrand("ccx", k) + 160 * math.sin(t * 0.5 + k)
        y = 380 * srand("ccy", k) + (ph - 0.5) * 300 + (1100 if k % 2 else 60)
        molecule(c, x, y, "CO2", 9, k * 40 + t * 25, co2 * clamp(ph * 4) * clamp((1 - ph) * 4))
    cx, cy, rx, ry, _i = CELLS[0]
    label_at(c, "tag", "叶肉细胞", cx - rx * 0.97, cy - ry * 0.15, -40, -250, st.get("lab_cell", 0.0), dark=True)
    label_at(c, "tag", "叶绿体", tx + 50, ty - 12, 260, -150, st.get("lab_chl", 0.0), dark=True)


# ======================================================================
# chloro：一个叶绿体，内部是基粒（类囊体叠成的“硬币摞”）；光谱演示：红光蓝紫光被吸收，绿光被反射
# ======================================================================

GRANA_POS = [(520, 520, 6), (760, 380, 5), (1010, 330, 4), (1260, 400, 5), (1480, 560, 4), (700, 690, 5),
             (980, 700, 7), (1240, 690, 6), (420, 650, 3)]
GRANUM_TARGET = 6         # 推进类囊体膜的那一摞
G_STEP = 18


def granum(c, x, y, n, rx=78, ry=21, seed="g", hi=0.0, disc_hi=-1, disc_a=0.0):
    """一摞类囊体：n 片扁圆盘叠起来；disc_hi 号盘子单独描亮（“一个类囊体”）"""
    for k in range(n):
        yy = y + (n - 1 - k) * G_STEP - (n - 1) * G_STEP / 2
        side = skia.Path()
        side.addRRect(skia.RRect.MakeRectXY(skia.Rect(x - rx, yy - 2, x + rx, yy + G_STEP + 3), 10, 10))
        c.drawPath(side, fill(lerp(shade(GRANA, -0.15), GRANA_TOP, 0.25 * hi)))
        hatch(c, side, (0.15, 0.35, 0.15), 0.6, -70, spacing=5, length=8, width=1.2, seed=f"{seed}{k}")
        c.drawPath(side, stroke(MEM_LT, lw(2.0), 0.8))
        if k == disc_hi and disc_a > 0:
            c.drawPath(side, stroke(GOLD, lw(5), disc_a))
    top_y = y - (n - 1) * G_STEP / 2 - 2
    top = smooth_path(ellipse_pts(x, top_y, rx, ry, 30))
    if hi > 0:
        glow(c, x, top_y, rx * 2.2, (0.7, 1, 0.5), hi * 0.9)
    c.drawPath(top, fill(lerp(GRANA_TOP, (0.85, 1, 0.6), hi * 0.5)))
    c.drawPath(smooth_path(ellipse_pts(x, top_y - 1, rx * 0.45, ry * 0.4, 20)), fill(shade(GRANA_TOP, 0.35), 0.8))
    for k in range(10):
        c.drawCircle(x + srand(seed, k) * rx * 0.75, top_y + srand(seed, k, 1) * ry * 0.6, 1.6, fill(LEAF_DK, 0.7))
    c.drawPath(top, stroke(MEM_LT, lw(2.2)))
    return top_y


def disc_y(n, k, y):
    """一摞里第 k 片盘子的中线"""
    return y + (n - 1 - k) * G_STEP - (n - 1) * G_STEP / 2 + G_STEP / 2


def ray(c, x0, y0, x1, y1, color, k, a=1.0, w=10):
    """一道彩色光线：从 (x0,y0) 射向 (x1,y1)，k=0~1 为射出的长度"""
    if a <= 0 or k <= 0:
        return
    xe, ye = lerp(x0, x1, k), lerp(y0, y1, k)
    c.drawLine(x0, y0, xe, ye, stroke(color, w * 3.2, 0.22 * a))
    c.drawLine(x0, y0, xe, ye, stroke(color, w, 0.95 * a))
    c.drawLine(x0, y0, xe, ye, stroke(WHITE, w * 0.3, 0.5 * a))
    glow(c, xe, ye, w * 4, color, a * 0.8)


def draw_chloro(c, t, st):
    """st: lab_chl, lab_grana, lab_thyl, lab_stroma, disc_hi, spec（光谱演示进度，见 scenes.py）"""
    dark_bg(c, (960, 500))
    cx, cy, rx, ry = 960, 540, 820, 400
    outer = smooth_path(wobble(ellipse_pts(cx, cy, rx, ry, 64), "chl", rx * 0.008, step=rx * 0.05, freq=0.004))
    soft_shadow(c, outer, 16, 20, 24, 0.35, (0, 0, 0))
    c.drawPath(outer, fill(DARK_MID))
    hatch(c, outer, MEM_LT, 0.12, -62, spacing=16, length=20, width=1.5, seed="chlh", density=0.6)
    c.drawPath(outer, stroke(MEM, lw(12)))
    inner = smooth_path(wobble(ellipse_pts(cx, cy, rx - 26, ry - 24, 64), "chl2", 6, step=40, freq=0.004))
    c.drawPath(inner, stroke(MEM_LT, lw(4), 0.8))
    c.drawPath(outer, stroke(INK, lw(3), 0.8))
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 4), (1, 5), (2, 6), (3, 7), (8, 5), (8, 0)):
        x0, y0, n0 = GRANA_POS[i]
        x1, y1, n1 = GRANA_POS[j]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 30 * srand("lam", i, j)
        c.drawPath(smooth_path([(x0, y0), (mx, my), (x1, y1)], closed=False), stroke(MEM, lw(3.2), 0.85))
    sp = smooth_path(ellipse_pts(1560, 400, 66, 40, 30, 0.3))
    c.drawPath(sp, fill(SUGAR_C))
    c.drawPath(sp, stroke(INK, lw(2.5), 0.6))
    for k in range(26):
        x = cx + srand("rib", k) * rx * 0.75
        y = cy + srand("rib", k, 1) * ry * 0.7
        c.drawCircle(x, y + 4 * math.sin(t + k), 5, fill(GOLD, 0.85))
    spec = st.get("spec") or {}
    tops = {}
    for i, (x, y, n) in enumerate(GRANA_POS):
        hi = spec.get("absorb", 0.0) if i == GRANUM_TARGET else 0.0
        hi = max(hi, spec.get("green_all", 0.0) * 0.5)
        tops[i] = granum(c, x, y, n, seed=f"gr{i}", hi=hi, disc_hi=3 if i == GRANUM_TARGET else -1,
                         disc_a=st.get("disc_hi", 0.0) if i == GRANUM_TARGET else 0.0)
    if spec:
        gx, gy, gn = GRANA_POS[GRANUM_TARGET]
        hx, hy = gx, tops[GRANUM_TARGET]
        k_in = spec.get("in", 0.0)
        a_in = spec.get("a", 0.0)
        # 三道光平行射入，间隔拉开，能分清颜色
        for d, col in ((-1, RED_LIGHT), (0, GREEN_LIGHT), (1, BLUE_LIGHT)):
            ox, oy = d * 46, -d * 56
            ray(c, hx - 560 + ox, hy - 460 + oy, hx + ox * 0.35, hy - 6 + oy * 0.2, col, k_in, a_in, w=9)
        ray(c, hx, hy - 6, hx + 520, hy - 430, GREEN_LIGHT, spec.get("out", 0.0), a_in, w=9)
        lab_rb = spec.get("lab_rb", 0.0) * a_in
        # 标签挂在光线中段：红光在左上一道，蓝紫光在右下一道
        label_at(c, "tag", "红光", hx - 560 - 46 + 0.45 * 560, hy - 460 + 56 + 0.45 * 454, -170, -10, lab_rb, dark=True)
        label_at(c, "tag", "蓝紫光", hx - 560 + 46 + 0.45 * 560, hy - 460 - 56 + 0.45 * 454, 190, 40, lab_rb, dark=True)
        label_at(c, "tag", "绿光", hx + 360, hy - 304, 120, 30, spec.get("lab_g", 0.0) * a_in, dark=True)
    label_at(c, "tag", "叶绿体", cx - rx * 0.62, cy - ry * 0.80, -230, -100, st.get("lab_chl", 0.0), dark=True)
    gx, gy, gn = GRANA_POS[2]
    label_at(c, "tag", "基粒", gx + 60, tops[2] - 4, 150, -110, st.get("lab_grana", 0.0), dark=True)
    gx, gy, gn = GRANA_POS[GRANUM_TARGET]
    label_at(c, "tag", "类囊体", gx + 78, disc_y(gn, 3, gy), 230, 120, st.get("lab_thyl", 0.0), dark=True)


def granum_anchor():
    """推进类囊体膜的位置：目标基粒第 3 片盘子的侧面"""
    gx, gy, gn = GRANA_POS[GRANUM_TARGET]
    return gx - 40, disc_y(gn, 3, gy)


# ======================================================================
# mem：类囊体膜截面（光反应）；膜的上方是基质（y < 0 区域画卡尔文循环、糖链）
# ======================================================================

MEM_Y = 520
MEM_HALF = 58
PSX = 860
ATX = 1560
ANTS = [(260, 1.0), (520, 0.9), (1180, 0.95), (1300, 0.8), (1780, 0.9), (-200, 0.9), (-560, 1.0), (2150, 0.95),
        (2480, 0.85), (-900, 0.9)]


def mem_curve(x):
    return MEM_Y + 18 * math.sin(x / 520)


def lipid_row(c, x0, x1, side):
    heads = skia.Path()
    tails = skia.Path()
    step = 26
    for k in range(math.floor(x0 / step), int(x1 / step) + 2):
        x = k * step
        yc = mem_curve(x)
        yh = yc + side * MEM_HALF
        heads.addCircle(x, yh, 11)
        tails.moveTo(x - 4, yh - side * 10)
        tails.lineTo(x - 4, yc + side * 4)
        tails.moveTo(x + 4, yh - side * 10)
        tails.lineTo(x + 4, yc + side * 4)
    c.drawPath(tails, stroke(MEM_LT, lw(2.0), 0.55))
    c.drawPath(heads, fill(MEM_LT))
    c.drawPath(heads, stroke(INK, lw(1.6), 0.5))


@functools.lru_cache(maxsize=None)
def complex_path(x, w, h_up, h_dn, seed):
    yc = mem_curve(x)
    return smooth_path(wobble(ellipse_pts(x, yc + (h_dn - h_up) / 2, w / 2, (h_up + h_dn) / 2, 40), seed, 4, step=16))


def complex_body(c, x, w, h_up, h_dn, seed, helices=5):
    """膜蛋白复合体：贯穿膜的深色团块，里面是绿色螺旋和叶绿素"""
    yc = mem_curve(x)
    path = complex_path(x, w, h_up, h_dn, seed)
    soft_shadow(c, path, 6, 10, 14, 0.4, (0, 0, 0))
    c.drawPath(path, fill(PROT))
    hatch(c, path, MEM, 0.35, -65, spacing=9, length=16, width=1.5, seed=seed)
    c.save()
    c.clipPath(path, doAntiAlias=True)
    for k in range(helices):
        hx = x - w * 0.34 + k * w * 0.68 / max(1, helices - 1) + srand(seed, k, "x") * 10
        top = yc - h_up * (0.35 + 0.35 * hrand(seed, k, "t"))
        bot = yc + h_dn * (0.35 + 0.35 * hrand(seed, k, "b"))
        c.save()
        c.translate(hx, (top + bot) / 2)
        c.rotate(srand(seed, k, "a") * 14)
        hh = (bot - top) / 2
        hp = skia.Path()
        hp.addRRect(skia.RRect.MakeRectXY(skia.Rect(-10, -hh, 10, hh), 10, 10))
        c.drawPath(hp, fill(shade(GRANA, -0.1)))
        for j in range(int(hh / 9)):
            yy = -hh + 8 + j * 18
            c.drawLine(-9, yy, 9, yy + 8, stroke(GRANA_TOP, lw(1.6), 0.6))
        c.drawPath(hp, stroke(INK, lw(1.8), 0.6))
        c.restore()
    for k in range(5):
        c.save()
        c.translate(x + srand(seed, k, "cx") * w * 0.3, yc + srand(seed, k, "cy") * 30)
        c.rotate(45)
        c.drawRect(skia.Rect(-7, -7, 7, 7), fill(CHL))
        c.restore()
    c.restore()
    ink(c, path, 3.4, 1, (0.03, 0.05, 0.03))


def antenna(c, x, y, r, seed, hi=0.0):
    """捕光天线：一圈叶绿素围成的八边形；hi 为受光激发的亮度"""
    c.save()
    c.translate(x, y)
    c.rotate(15 * srand(seed))
    body = poly_path([(r * math.cos(k * math.pi / 4 + math.pi / 8), r * math.sin(k * math.pi / 4 + math.pi / 8))
                      for k in range(8)])
    if hi > 0:
        glow(c, 0, 0, r * 2.6, (0.6, 1.0, 0.5), hi)
    c.drawPath(body, fill(lerp(GRANA, MEM_LT, hi)))
    for k in range(4):
        ang = k * math.pi / 2 + math.pi / 4
        px, py = r * 0.95 * math.cos(ang), r * 0.95 * math.sin(ang)
        pp = poly_path([(px + r * 0.32 * math.cos(ang + j * math.pi / 3), py + r * 0.32 * math.sin(ang + j * math.pi / 3))
                        for j in range(6)])
        c.drawPath(pp, fill(lerp(CHL, (0.85, 1, 0.7), hi)))
        c.drawPath(pp, stroke(CREAM, lw(2), 0.8))
    c.drawCircle(0, 0, r * 0.28, fill(lerp(MEM_LT, (1, 1, 0.8), hi)))
    c.drawPath(body, stroke(CREAM, lw(2.4), 0.85))
    c.restore()


def photon(c, x0, y0, x1, y1, k, a=1.0):
    """光子：一条从 (x0,y0) 射向 (x1,y1) 的金色波浪线，k=0~1 为飞行进度"""
    if a <= 0 or k <= 0:
        return
    ang = math.atan2(y1 - y0, x1 - x0)
    L = math.hypot(x1 - x0, y1 - y0)
    head = L * k
    tail = max(0, head - 260)
    pts = []
    for i in range(61):
        u = tail + (head - tail) * i / 60
        off = 13 * math.sin(u / 22)
        pts.append((x0 + math.cos(ang) * u - math.sin(ang) * off, y0 + math.sin(ang) * u + math.cos(ang) * off))
    c.save()
    c.translate(x0, y0)
    c.rotate(math.degrees(ang))
    sh = skia.GradientShader.MakeLinear([skia.Point(tail, 0), skia.Point(head + 1, 0)],
                                        [skia.Color4f(1, 0.85, 0.4, 0), skia.Color4f(1, 0.85, 0.4, 0.22 * a)])
    c.drawRect(skia.Rect(tail, -46, head, 46), skia.Paint(Shader=sh))
    c.restore()
    p = smooth_path(pts, closed=False)
    c.drawPath(p, stroke(GOLD, 9, 0.25 * a))
    c.drawPath(p, stroke(GOLD, 4.2, a))
    glow(c, pts[-1][0], pts[-1][1], 34, (1, 0.9, 0.5), a)


def big_antenna_pos():
    return PSX, mem_curve(PSX) - MEM_HALF - 190


def water_pos():
    return PSX - 40, mem_curve(PSX) + MEM_HALF + 120


def draw_membrane(c, t, st):
    r = view_rect(c)
    x0, x1 = r.left() - 30, r.right() + 30
    lumen = skia.Path()
    lumen.moveTo(x0, mem_curve(x0))
    x = x0
    while x <= x1 + 40:
        lumen.lineTo(x, mem_curve(x))
        x += 40
    lumen.lineTo(x1 + 40, max(r.bottom(), MEM_Y) + 10)
    lumen.lineTo(x0, max(r.bottom(), MEM_Y) + 10)
    lumen.close()
    c.drawPath(lumen, fill((0.05, 0.13, 0.09), 0.55))
    band = skia.Path()
    xs = [x0 + i * 40 for i in range(int((x1 - x0) / 40) + 2)]
    band.moveTo(xs[0], mem_curve(xs[0]) - MEM_HALF)
    for x in xs:
        band.lineTo(x, mem_curve(x) - MEM_HALF)
    for x in xs[::-1]:
        band.lineTo(x, mem_curve(x) + MEM_HALF)
    band.close()
    c.drawPath(band, fill((0.10, 0.25, 0.15)))
    lipid_row(c, x0, x1, -1)
    lipid_row(c, x0, x1, 1)
    for i, (ax, s) in enumerate(ANTS):
        if ax < x0 - 100 or ax > x1 + 100:
            continue
        ay = mem_curve(ax) - MEM_HALF - 120 * s
        c.drawPath(smooth_path([(ax, mem_curve(ax) - MEM_HALF), (ax + 8, ay + 50), (ax, ay)], closed=False),
                   stroke(CREAM, lw(2.2), 0.6))
        antenna(c, ax, ay, 44 * s, f"an{i}", hi=st.get("amb_hit", {}).get(i, 0.0))
    complex_body(c, PSX, 380, 135, 115, "ps2", helices=7)
    bx, by = big_antenna_pos()
    c.drawPath(smooth_path([(PSX, mem_curve(PSX) - 120), (PSX + 10, by + 70), (PSX, by)], closed=False),
               stroke(CREAM, lw(2.4), 0.6))
    antenna(c, bx, by, 70, "anbig", hi=st.get("hit", 0.0))
    # 能量从天线传到光系统中心
    hop = st.get("hop", -1.0)
    if 0 <= hop <= 1:
        hx, hy = bx + 10 * math.sin(hop * 3), lerp(by, mem_curve(PSX) + 10, ease_io(hop))
        glow(c, hx, hy, 50, (0.8, 1, 0.6), 1.0)
        c.drawCircle(hx, hy, 9, fill((0.95, 1, 0.8)))
    # ATP 合酶
    yc = mem_curve(ATX)
    stalk = skia.Path()
    stalk.addRRect(skia.RRect.MakeRectXY(skia.Rect(ATX - 34, yc - 150, ATX + 34, yc + 70), 16, 16))
    c.drawPath(stalk, fill(PROT))
    hatch(c, stalk, MEM, 0.3, -65, spacing=8, length=12, width=1.3, seed="atps")
    head = smooth_path(wobble(ellipse_pts(ATX, yc - 200, 110, 70, 36), "atph", 3))
    c.drawPath(head, fill(PROT))
    hatch(c, head, MEM, 0.35, -65, spacing=9, length=14, width=1.4, seed="atph")
    ink(c, head, 3.0, 1, (0.03, 0.05, 0.03))
    spin = st.get("spin", 0.0)
    for k in range(6):
        ang = k / 6 * 2 * math.pi + spin
        c.drawCircle(ATX + 70 * math.cos(ang), yc - 200 + 40 * math.sin(ang), 15, fill(shade(GRANA, -0.1)))


def _split(c, t, wx, wy, sk, mark, seed, show_label):
    """水的分解：sk=0~1；先发光，再拆成一个氧气分子和四个氢离子"""
    if sk < 0.35:
        molecule(c, wx - 60, wy + 30 * (1 - sk / 0.35), "H2O", 20, 10, mark=mark)
        molecule(c, wx + 60, wy + 40 * (1 - sk / 0.35), "H2O", 20, -15, mark=mark)
        if sk > 0.2:
            glow(c, wx, wy, 120, (1, 0.95, 0.6), (sk - 0.2) / 0.15)
        return
    k2 = ease_out_q((sk - 0.35) / 0.65)
    ox, oy = wx + 20 + 380 * k2, wy + 60 * k2 - 50 * math.sin(k2 * 3)
    c.drawCircle(ox, oy, 52, fill(CREAM, 0.12))
    c.drawCircle(ox, oy, 52, stroke(CREAM, lw(2.4), 0.55))
    molecule(c, ox, oy, "O2", 22, 20 * k2, mark=mark)
    if show_label:
        label_at(c, "chem", "O2", ox, oy, 76, -46, clamp(k2 * 3), dark=True)
    for k in range(4):
        ang = -2.4 + k * 0.55
        hx = wx - 30 + math.cos(ang) * 200 * k2
        hy = wy + 60 + math.sin(ang) * -120 * k2 + 60 * k2
        molecule(c, hx, hy, "H", 11)
        text(c, "+", hx + 12, hy - 8, font(22, 700), CREAM, 0.9)


def draw_light(c, t, st):
    """光反应的动画元素（画在膜上下两侧）"""
    bx, by = big_antenna_pos()
    pk = st.get("photon", 0.0)
    if 0 < pk < 1:
        photon(c, bx - 520, by - 640, bx - 30, by - 20, pk)
    for i, k in st.get("amb_photons", {}).items():
        ax, s = ANTS[i]
        ay = mem_curve(ax) - MEM_HALF - 120 * s
        photon(c, ax - 300, ay - 520, ax - 10, ay - 10, k, 0.75)
    wx, wy = water_pos()
    if st.get("split", 0.0) < 1.0 or st.get("split2", -1.0) < 0:
        _split(c, t, wx, wy, st.get("split", 0.0), 0.0, "s1", True)
    sk2 = st.get("split2", -1.0)
    if sk2 >= 0:
        _split(c, t, wx - 230, wy + 60, sk2, st.get("mark", 0.0), "s2", False)
    else:
        molecule(c, wx - 230, wy + 120 + 10 * math.sin(t * 1.3), "H2O", 20, -20, mark=st.get("mark", 0.0))
    molecule(c, wx - 440, wy + 40 + 10 * math.sin(t * 1.1 + 2), "H2O", 20, 25)
    molecule(c, wx + 330, wy + 130 + 10 * math.sin(t * 1.1 + 1), "H2O", 20, 15)
    label_at(c, "chem", "H2O", wx - 440, wy + 40, 0, 90, st.get("lab_h2o", 0.0), dark=True)
    label_at(c, "tag", "做了标记的氧", wx - 230, wy + 120, -260, -130, st.get("lab_mark", 0.0), dark=True)
    label_at(c, "tag", "光", bx - 177, by - 206, -150, 40, st.get("lab_light", 0.0), dark=True)
    label_at(c, "tag", "类囊体膜", 300, mem_curve(300) + 30, -20, 220, st.get("lab_mem", 0.0), dark=True)
    for i, k in enumerate(st.get("atps", [])):
        if 0 < k < 1:
            x = ATX + 60 + 120 * math.sin(i * 1.7) * k
            y = mem_curve(ATX) - 260 - 520 * k
            atp(c, x, y, 15, clamp(k * 6) * clamp((1 - k) * 4))
    for i, k in enumerate(st.get("hs", [])):
        if 0 < k < 1:
            x = PSX + 160 + 90 * i + 60 * math.sin(k * 4 + i)
            y = mem_curve(PSX) - 120 - 600 * k
            h_carrier(c, x, y, 11, clamp(k * 6) * clamp((1 - k) * 4))
    label_at(c, "chem", "ATP", ATX + 60, mem_curve(ATX) - 330, 0, -40, st.get("lab_atp", 0.0), dark=True)


CALVIN = (960, -560, 300)      # 卡尔文循环：圆环中心与半径（膜上方的基质里）


def calvin_ring(c, t, st):
    cx, cy, R = CALVIN
    rot = st.get("ring_rot", 0.0)
    ring = skia.Path()
    ring.addCircle(cx, cy, R + 38)
    hole = skia.Path()
    hole.addCircle(cx, cy, R - 38)
    ring = skia.Op(ring, hole, skia.PathOp.kDifference_PathOp)
    glow(c, cx, cy, R + 140, (0.5, 0.9, 0.5), 0.25 + 0.25 * st.get("ring_glow", 0.0))
    soft_shadow(c, ring, 8, 12, 16, 0.4, (0, 0, 0))
    c.drawPath(ring, fill(PROT))
    c.save()
    c.translate(cx, cy)
    c.rotate(math.degrees(rot))
    c.translate(-cx, -cy)
    hatch(c, ring, MEM, 0.4, -65, spacing=9, length=16, width=1.5, seed="calv")
    for k in range(8):
        a = k / 8 * 2 * math.pi
        x, y = cx + R * math.cos(a), cy + R * math.sin(a)
        c.drawCircle(x, y, 17, fill((0.22, 0.24, 0.22)))
        c.drawCircle(x, y, 17, stroke(MEM_LT, lw(2.2), 0.8))
        c.drawCircle(x, y, 7, fill(MEM_LT, 0.8))
    for k in range(24):
        a = k / 24 * 2 * math.pi
        c.drawLine(cx + (R + 40) * math.cos(a), cy + (R + 40) * math.sin(a), cx + (R + 52) * math.cos(a + 0.05),
                   cy + (R + 52) * math.sin(a + 0.05), stroke(CREAM, lw(2), 0.5))
    c.restore()
    ink(c, ring, 3.0, 1, (0.03, 0.05, 0.03))
    # 圆环内侧的循环箭头
    aa = st.get("arrow_a", 0.0)
    if aa > 0:
        for k in range(3):
            a0 = math.degrees(rot) + k * 120
            arc = skia.Path()
            arc.addArc(skia.Rect(cx - R + 80, cy - R + 80, cx + R - 80, cy + R - 80), a0, 80)
            c.drawPath(arc, stroke(MEM_LT, lw(5), 0.7 * aa))
            ea = math.radians(a0 + 80)
            rr = R - 80
            ex, ey = cx + rr * math.cos(ea), cy + rr * math.sin(ea)
            tx, ty = -math.sin(ea), math.cos(ea)
            nx, ny = math.cos(ea), math.sin(ea)
            c.drawPath(poly_path([(ex + tx * 16, ey + ty * 16), (ex - tx * 4 + nx * 12, ey - ty * 4 + ny * 12),
                                  (ex - tx * 4 - nx * 12, ey - ty * 4 - ny * 12)]), fill(MEM_LT, 0.7 * aa))


def draw_stroma(c, t, st):
    """基质：卡尔文循环 + 糖链（淀粉、纤维素）"""
    cx, cy, R = CALVIN
    # 基质里漂着的核糖体和小分子（让画面不空）
    for k in range(40):
        x = cx - 1100 + 2900 * hrand("str", k) + 30 * math.sin(t * 0.4 + k)
        y = cy - 640 + 1150 * hrand("str", k, 1) + 24 * math.cos(t * 0.5 + k * 1.3)
        c.drawCircle(x, y, 4 + 3 * hrand("str", k, 2), fill(GOLD, 0.55))
    calvin_ring(c, t, st)
    for i, k in enumerate(st.get("co2s", [])):
        if k <= 0:
            continue
        a_end = math.radians(200 + 25 * i)
        ex, ey = cx + (R + 10) * math.cos(a_end), cy + (R + 10) * math.sin(a_end)
        sx, sy = cx - 1000 - 120 * i, cy - 300 + 140 * i
        kk = ease_out_q(min(k, 1.0))
        x, y = lerp(sx, ex, kk), lerp(sy, ey, kk) + 40 * math.sin(kk * 3 + i) * (1 - kk)
        ma = clamp((1.25 - k) * 6) if k > 1 else 1.0
        molecule(c, x, y, "CO2", 19, 30 * i + 90 * kk, ma)
        if i == 0:
            label_at(c, "chem", "CO2", x, y, 0, -56, st.get("lab_co2", 0.0) * ma, dark=True)
        if 0.95 < k < 1.25:
            glow(c, ex, ey, 70, (1, 0.95, 0.6), clamp((1.25 - k) * 4))
    for i, k in enumerate(st.get("feeds", [])):
        if 0 < k < 1:
            a_end = math.radians(70 - 22 * i)
            ex, ey = cx + R * math.cos(a_end), cy + R * math.sin(a_end)
            sx, sy = cx + 400 + 60 * i, cy + 760
            kk = ease_io(k)
            x, y = lerp(sx, ex, kk), lerp(sy, ey, kk)
            if i % 2 == 0:
                atp(c, x, y, 17, clamp((1 - k) * 5))
            else:
                h_carrier(c, x, y, 14, clamp((1 - k) * 5))
    for i, k in enumerate(st.get("sugars", [])):
        if 0 < k < 1:
            a0 = math.radians(20)
            sx, sy = cx + (R + 40) * math.cos(a0), cy + (R + 40) * math.sin(a0)
            x, y = sx + 300 * ease_out_q(k), sy + 50 * ease_out_q(k)
            sa = clamp(k * 5) * clamp((1 - k) * 5)
            hexagon(c, x, y, 30, sa, 20 * k, glow_a=clamp((0.3 - k) * 4))
            if i == 0:
                label_at(c, "chem", "糖", x, y, 0, -62, st.get("lab_sugar", 0.0) * sa, dark=True)
    label_at(c, "tag", "卡尔文循环", cx - R * 0.72, cy - R * 0.72, -330, -20, st.get("lab_calvin", 0.0), dark=True)
    if st.get("chain", 0.0) > 0:
        _sugar_chains(c, t, st)


def _sugar_chains(c, t, st):
    """糖链：一个接一个连起来，一部分卷成淀粉粒，一部分拉成直的纤维素"""
    cx, cy, R = CALVIN
    k = st.get("chain", 0.0)
    x0, y0 = cx + R + 90, cy + 130
    n = int(10 * k)
    pts = [(x0 + 64 * i, y0 + 30 * math.sin(i * 0.9 + t * 0.6)) for i in range(n + 1)]
    for i in range(1, len(pts)):
        c.drawLine(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], stroke(SUGAR_C, lw(5), 0.9))
    for i, (x, y) in enumerate(pts):
        hexagon(c, x, y, 26, clamp(10 * k - i), i * 15)
    ks = st.get("starch", 0.0)
    if ks > 0:
        sx, sy = cx + R + 700, cy - 230
        for i in range(int(46 * ks)):
            a = i * 0.55
            rr = 14 + 7.5 * a
            hexagon(c, sx + rr * math.cos(a), sy + rr * math.sin(a) * 0.85, 18, clamp(46 * ks - i), a * 40,
                    color=shade(SUGAR_C, -0.04))
        label_at(c, "tag", "淀粉", sx + 180, sy - 60, 190, -50, st.get("lab_starch", 0.0), dark=True)
    kc = st.get("cellulose", 0.0)
    if kc > 0:
        bx, by = cx + R + 250, cy + 330
        for j in range(4):
            yy = by + j * 46
            m = int(18 * kc)
            for i in range(m):
                hexagon(c, bx + i * 46 + (j % 2) * 23, yy, 19, clamp(18 * kc - i), 0, color=shade(SUGAR_C, 0.05))
        label_at(c, "tag", "纤维素", bx + 780, by + 70, 170, -40, st.get("lab_cellu", 0.0), dark=True)


def draw_mem(c, t, st):
    dark_bg(c, (900, 260), glow=0.9)
    r = view_rect(c)
    if r.top() < 150:
        draw_stroma(c, t, st)
    if r.bottom() > MEM_Y - 420:
        draw_membrane(c, t, st)
        draw_light(c, t, st)


# ======================================================================
# earth：地球（正射投影 + 手绘描边），昼夜交替
# ======================================================================

CONTINENTS = {
    "eurasia": [(-9, 43), (-9, 37), (-5, 36), (3, 43), (8, 44), (12, 44), (16, 41), (18, 40), (19, 42), (20, 40),
                (23, 37), (26, 40), (28, 41), (36, 36), (35, 33), (34, 28), (43, 13), (52, 16), (57, 22), (60, 25),
                (57, 26), (51, 24), (48, 30), (50, 30), (56, 27), (62, 25), (67, 24), (72, 21), (77, 8), (80, 13),
                (80, 16), (87, 21), (92, 22), (94, 17), (98, 16), (98, 9), (101, 3), (104, 1), (103, 10), (106, 10),
                (109, 12), (108, 17), (106, 20), (110, 21), (117, 23), (121, 28), (122, 31), (120, 35), (122, 37),
                (118, 38), (121, 40), (125, 40), (129, 35), (129, 40), (135, 43), (140, 48), (141, 53), (135, 55),
                (141, 59), (150, 59), (156, 51), (163, 59), (165, 62), (180, 65), (180, 70), (160, 70), (140, 72),
                (130, 71), (113, 74), (105, 77), (95, 76), (80, 73), (70, 73), (67, 69), (60, 69), (50, 68), (44, 68),
                (41, 67), (33, 69), (25, 71), (15, 68), (10, 63), (5, 61), (5, 58), (8, 57), (10, 54), (8, 54),
                (4, 52), (2, 51), (-2, 48), (-4, 48), (-1, 46), (-2, 44)],
    "africa": [(-17, 21), (-17, 15), (-15, 11), (-8, 4), (-4, 5), (5, 6), (9, 4), (9, -1), (12, -5), (13, -12), (12, -17),
               (15, -27), (18, -33), (20, -35), (26, -34), (33, -28), (35, -24), (35, -18), (40, -15), (40, -10),
               (39, -5), (42, 0), (51, 12), (44, 11), (43, 13), (38, 18), (34, 28), (32, 31), (25, 32), (20, 31),
               (10, 34), (10, 37), (3, 37), (-6, 35), (-10, 30), (-13, 27)],
    "australia": [(114, -22), (114, -34), (118, -35), (124, -34), (131, -31), (135, -35), (138, -35), (140, -38),
                  (147, -38), (150, -37), (153, -31), (153, -25), (146, -19), (145, -15), (142, -11), (141, -15),
                  (136, -12), (130, -12), (125, -14), (122, -18)],
    "sumatra": [(95, 5), (98, 4), (104, -2), (106, -6), (103, -5), (99, 0)],
    "borneo": [(109, 1), (113, 4), (117, 7), (119, 1), (116, -4), (110, -3)],
    "newguinea": [(131, -1), (138, -2), (141, -3), (150, -10), (143, -9), (138, -8)],
    "japan": [(130, 31), (135, 34), (140, 35), (142, 40), (141, 45), (139, 40), (133, 35)],
    "madagascar": [(44, -25), (47, -25), (50, -15), (49, -12), (44, -17)],
    "britain": [(-5, 50), (1, 51), (-2, 56), (-5, 58), (-6, 55), (-3, 54)],
    "philippines": [(120, 18), (122, 18), (126, 7), (122, 7)],
    "java": [(105, -6), (114, -7), (114, -8), (106, -7)],
}
DESERT = [(-15, 18), (-5, 30), (10, 32), (30, 31), (35, 29), (45, 28), (55, 23), (50, 18), (40, 15), (30, 16),
          (15, 15), (0, 15)]
EARTH = (960, 580, 360)
CHINA = (112.0, 30.0)


def _densify(poly, step=2.0):
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        m = max(1, int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / step))
        for j in range(m):
            u = j / m
            out.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
    return out


DENSE = {k: _densify(v) for k, v in CONTINENTS.items()}
DENSE_DESERT = _densify(DESERT)


def project(lon, lat, lon0, lat0, R, cx, cy):
    """正射投影；背面的点压到地球边缘上"""
    la, lo = math.radians(lat), math.radians(lon - lon0)
    la0 = math.radians(lat0)
    x = math.cos(la) * math.sin(lo)
    y = math.cos(la0) * math.sin(la) - math.sin(la0) * math.cos(la) * math.cos(lo)
    z = math.sin(la0) * math.sin(la) + math.cos(la0) * math.cos(la) * math.cos(lo)
    if z < 0:
        d = math.hypot(x, y) or 1.0
        x, y = x / d, y / d
    return cx + R * x, cy - R * y, z


def earth_view(t):
    """地球的中心经纬度（缓慢自转：地面从左往右走）"""
    return 112.0 - 1.2 * (t - 100.0), 22.0


def china_point(t):
    lon0, lat0 = earth_view(t)
    cx, cy, R = EARTH
    x, y, _z = project(*CHINA, lon0, lat0, R, cx, cy)
    return x, y


@functools.lru_cache(maxsize=8)
def _night_mask(sx, sy, sz, R):
    """昼夜遮罩：地球圆盘上每个点的法线和太阳方向点乘，小于 0 的是夜晚（晨昏线带一点柔和过渡）"""
    n = int(2 * R) + 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x = (xx - n / 2) / R
    y = -(yy - n / 2) / R
    rr = x * x + y * y
    z = np.sqrt(np.clip(1 - rr, 0, 1))
    d = x * sx + y * sy + z * sz
    a = np.clip((0.06 - d) / 0.16, 0, 1) * (rr <= 1.0)
    rgba = np.zeros((n, n, 4), np.float32)
    rgba[..., 0] = NIGHT[0] * 255 * a * 0.88
    rgba[..., 1] = NIGHT[1] * 255 * a * 0.88
    rgba[..., 2] = NIGHT[2] * 255 * a * 0.88
    rgba[..., 3] = a * 225
    return skia.Image.fromarray(np.ascontiguousarray(rgba.astype(np.uint8)), colorType=skia.ColorType.kRGBA_8888_ColorType,
                                alphaType=skia.AlphaType.kPremul_AlphaType)


def draw_earth(c, t, st):
    """st: night（0 白天 ~ 1 夜晚，太阳方向转到背后）, sparkle（陆地上的叶子闪光）, question（片尾问题的透明度）"""
    night = st.get("night", 0.0)
    paper_bg(c, glow=(300, 200), glow_a=0.5 * (1 - night))
    vr = view_rect(c)
    if night > 0:
        c.drawRect(vr, fill(NIGHT, 0.92 * night))
        for k in range(90):
            x = hrand("star", k) * W
            y = hrand("star", k, 1) * H
            tw = 0.5 + 0.5 * math.sin(t * 2 + k)
            c.drawCircle(x, y, 1.6 + 1.4 * hrand("star", k, 2), fill(CREAM, night * (0.3 + 0.5 * tw)))
    cx, cy, R = EARTH
    lon0, lat0 = earth_view(t)
    disk = skia.Path()
    disk.addCircle(cx, cy, R)
    soft_shadow(c, disk, 14, 18, 22, 0.18 * (1 - night))
    c.drawPath(disk, fill(OCEAN))
    c.save()
    c.clipPath(disk, doAntiAlias=True)
    hatch(c, disk, OCEAN_DK, 0.45, -30, spacing=8, length=18, width=1.5, seed="ocean", density=0.6)
    land = skia.Path()
    coast = skia.Path()     # 只描正面的海岸线：背面压到边缘上的那部分不描边
    for name, poly in DENSE.items():
        pts = [project(lo, la, lon0, lat0, R, cx, cy) for lo, la in poly]
        if max(p[2] for p in pts) <= 0:
            continue
        land.addPath(poly_path([(p[0], p[1]) for p in pts]))
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            if a[2] > 0.03 and b[2] > 0.03:
                coast.moveTo(a[0], a[1])
                coast.lineTo(b[0], b[1])
    c.drawPath(land, fill(LEAF))
    c.save()
    c.clipPath(land, doAntiAlias=True)
    dpts = [project(lo, la, lon0, lat0, R, cx, cy) for lo, la in DENSE_DESERT]
    if max(p[2] for p in dpts) > 0:
        c.drawPath(smooth_path([(p[0], p[1]) for p in dpts]), fill(SAND))
    hatch(c, land, LEAF_DK, 0.5, 50, spacing=7, length=14, width=1.4, seed="land", density=0.7)
    c.restore()
    c.drawPath(coast, stroke(INK, 2.6, 0.85))
    # 陆地上闪动的小亮点：每一片叶子都在做光合作用
    sp = st.get("sparkle", 0.0)
    if sp > 0:
        for k in range(90):
            lo = -15 + 165 * hrand("lf", k)
            la = -35 + 90 * hrand("lf", k, 1)
            x, y, z = project(lo, la, lon0, lat0, R, cx, cy)
            if z < 0.15 or not land.contains(x, y):
                continue
            ph = (math.sin(t * 3 + k * 1.7) + 1) / 2
            glow(c, x, y, 18, (0.75, 1.0, 0.4), sp * ph * (1 - night))
    for k in range(7):
        lo = lon0 - 60 + 120 * hrand("cl", k)
        la = -40 + 80 * hrand("cl", k, 1)
        pts = [project(lo + d, la + 3 * math.sin(d / 8), lon0, lat0, R, cx, cy) for d in range(0, 26, 2)]
        if min(p[2] for p in pts) > 0.1:
            c.drawPath(smooth_path([(p[0], p[1]) for p in pts], closed=False), stroke(WHITE, 3.0, 0.75))
    if night > 0:
        ang = math.pi * 0.62 * night
        s0 = np.array([-0.55, 0.25, 0.80])
        s0 /= np.linalg.norm(s0)
        # 太阳方向绕竖轴转到背后：晨昏线从右往左扫过
        sx = s0[0] * math.cos(ang) - s0[2] * math.sin(ang)
        sz = s0[0] * math.sin(ang) + s0[2] * math.cos(ang)
        img = _night_mask(round(sx, 3), round(float(s0[1]), 3), round(sz, 3), R)
        c.drawImage(img, cx - img.width() / 2, cy - img.height() / 2)
        for k in range(40):
            lo = 70 + 70 * hrand("city", k)
            la = 10 + 35 * hrand("city", k, 1)
            x, y, z = project(lo, la, lon0, lat0, R, cx, cy)
            if z > 0.1 and land.contains(x, y):
                c.drawCircle(x, y, 1.8, fill((1.0, 0.85, 0.5), 0.8 * clamp(night * 2 - 1)))
    sh = skia.GradientShader.MakeRadial(skia.Point(cx - R * 0.4, cy - R * 0.45), R * 1.6,
                                        [skia.Color4f(1, 1, 1, 0.16), skia.Color4f(0, 0, 0, 0.0),
                                         skia.Color4f(0, 0, 0, 0.18)], [0.0, 0.6, 1.0])
    c.drawPath(disk, skia.Paint(Shader=sh, AntiAlias=True))
    c.restore()
    ink(c, smooth_path(wobble(ellipse_pts(cx, cy, R, R, 90), "earth", 1.2, step=16)), 4.6)
    qa = st.get("question", 0.0)
    if qa > 0:
        text(c, "到了晚上，植物还会放出氧气吗？", W / 2, 130, font(62, 800), CREAM, qa, "center")

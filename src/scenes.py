# 镜头系统与动作表：一个连续的镜头从盆栽一路推进到类囊体膜，再一口气拉回，最后拉远到地球
#
# 层级嵌套：每层在自己的 1920×1080 设计坐标里画；子层的设计中心 (960, 540) 对应父层里的锚点，子层 1 个单位 = 父层 s 个单位
#   earth ─(淡入淡出)→ room ─(放大镜镜片)→ leaf ─(淡入淡出)→ epi ─(气孔孔口)→ cell ─(淡入淡出)→ chloro ─(淡入淡出)→ mem
# 镜头：关键帧写成（时刻, 层, 层内中心 x, y, 层内缩放）；两帧之间在统一的根坐标里按“对数缩放 + 不动点”插值，
# 推进时画面上有一个点始终不动，看起来就是朝那个点钻进去
import functools
import math

import numpy as np
import skia

import arrange as A
import levels as L
from draw import (CREAM, H, INK, LABELS, W, ZK, clamp, dark_streaks, ease_io, ease_out, fill, finish_paper,
                  flush_labels, font, lerp, new_canvas, ramp, snapshot, text, xform)
from script import LINES
from timeline import BAR, BEAT, PUNCT, SECTION, T, TEXT, TOTAL, anchors, e, s, wt

C0 = (960.0, 540.0)


# ======================================================================
# 动作表（全部由时间线推出来，配音变了这里跟着变）
# ======================================================================

def _w(lid, sub, frac=0.0):
    return wt(lid, sub, frac)


Q = dict(
    mag_in=(8.3, 10.0), lens_leaf=(10.0, 10.6), mag_out=(98.0, 99.2),
    f_items=[_w("H01", "二氧化碳"), _w("H01", "加水"), _w("H01", "加水", 0.5), _w("H01", "在光照下"),
             _w("H01", "有机物"), _w("H01", "和氧气"), _w("H01", "氧气")],
    vein=_w("L01", "沿着叶脉"), h2o=_w("L01", "水从根部"),
    stoma=_w("L02", "气孔进来"), guard=_w("L02", "保卫细胞"), open_=_w("L02", "张开"), close_=_w("L02", "关上"),
    co2_epi=_w("L02", "二氧化碳"),
    cell=_w("C01", "叶肉细胞"), chl=_w("C01", "叶绿体"),
    grana=_w("C02", "基粒"), thyl=_w("C02", "类囊体"),
    chl_mem=_w("C03", "类囊体膜"), rb=_w("C03", "红光"), green=_w("C03", "绿光"), so=_w("C03", "所以"),
    hit=SECTION["light"], split=_w("P01", "把水分解"),
    oxy=_w("P02", "氧气从"), iso=_w("P03", "同位素"), atp=_w("P04", "ATP"), stroma_word=_w("P04", "基质"),
    k_co2=_w("K01", "二氧化碳"), k_fix=_w("K01", "固定"), k_energy=_w("K01", "能量"), k_sugar=_w("K01", "转化成糖"),
    k_loop=_w("K02", "一圈一圈"), k_calvin=_w("K02", "卡尔文循环"),
    k_chain=_w("K03", "连成长链"), k_starch=_w("K03", "淀粉"), k_cellu=_w("K03", "纤维素"),
    z_light=_w("Z01", "光提供"), z_water=_w("Z01", "把水拆开"), z_oxy=_w("Z01", "放出氧气"), z_co2=_w("Z01", "二氧化碳"),
    earth=SECTION["earth"], night=SECTION["end"] - 0.6, question=e("Q01") + 0.1,
)

KF = [
    (0.0, "room", 960, 540, 1.0),
    (10.6, "room", 1000, 500, 1.12),
    (12.5, "leaf", 960, 540, 1.0),
    (18.7, "leaf", 1180, 470, 1.22),
    (20.6, "epi", 960, 540, 1.0),
    (26.5, "epi", 962, 556, 1.35),
    (28.2, "cell", 960, 540, 1.0),
    (33.4, "cell", 925, 420, 1.35),
    (35.0, "chloro", 960, 540, 1.0),
    (40.3, "chloro", 940, 520, 1.1),
    (41.3, "chloro", 900, 480, 1.2),
    (48.9, "chloro", 905, 485, 1.25),
    (50.0, "mem", 960, 540, 1.0),
    (54.0, "mem", 900, 600, 1.12),
    (60.2, "mem", 1000, 620, 1.12),
    (66.6, "mem", 900, 590, 1.05),
    (71.8, "mem", 1250, 420, 0.95),
    (75.0, "mem", 900, -540, 1.08),
    (87.9, "mem", 960, -540, 1.12),
    (89.6, "mem", 1560, -470, 0.9),
    (94.7, "mem", 1580, -470, 0.93),
    (97.5, "room", 960, 540, 1.0),
    (99.6, "room", 1250, 330, 1.35),
    (105.0, "room", 1250, 330, 1.38),
    (107.5, "earth", 960, 560, 1.0),
    (TOTAL, "earth", 960, 575, 0.93),
]


# ======================================================================
# 层级树
# ======================================================================

ORDER = ["earth", "room", "leaf", "epi", "cell", "chloro", "mem"]
S_LENS = 1 / 7.5


def mag_state(t):
    """放大镜：(x, y, R, 角度) 或 None"""
    lx, ly, R, ang = L.LENS
    a, b = Q["mag_in"]
    if t < a:
        return None
    if t < b:
        k = ease_out(clamp((t - a) / (b - a)), 3)
        return lerp(1780, lx, k), lerp(1000, ly, k), R, ang
    a2, b2 = Q["mag_out"]
    if t < a2:
        return lx, ly, R, ang
    if t < b2:
        k = clamp((t - a2) / (b2 - a2)) ** 2
        return lerp(lx, 2350, k), lerp(ly, 1350, k), R, ang
    return None


def level_link(name, t):
    """子层在父层里的 (锚点 x, y, 缩放 s)"""
    if name == "room":
        x, y = L.china_point(t)
        return x, y, 1 / 40
    if name == "leaf":
        return L.LENS[0], L.LENS[1], S_LENS
    if name == "epi":
        return L.EPI_ANCHOR[0], L.EPI_ANCHOR[1], 0.1
    if name == "cell":
        return L.MAIN_STOMA[0], L.MAIN_STOMA[1], 1 / 24
    if name == "chloro":
        x, y, _a = L.chl_target()
        return x, y, L.CHL_RX / 820
    if name == "mem":
        x, y = L.granum_anchor()
        return x, y, 0.026
    raise KeyError(name)


# 淡入淡出的层：子层的层内缩放在 [a0, a1] 之间从 0 淡到 1（按对数）
FADE = {"room": (0.15, 0.5), "epi": (0.55, 0.9), "chloro": (0.5, 0.85), "mem": (0.45, 0.8)}


def chain(t):
    """每层到根坐标的变换：root = O + S·(local − C0)"""
    out = {"earth": (0.0, 0.0, 1.0)}
    ox, oy, S = 0.0, 0.0, 1.0
    for name in ORDER[1:]:
        ax, ay, sk = level_link(name, t)
        # 父层局部 → 根：root = O_p + S_p·(p − C0)
        ox, oy = ox + S * (ax - C0[0]), oy + S * (ay - C0[1])
        S *= sk
        out[name] = (ox, oy, S)
    # 根层 earth 的局部坐标就是根坐标：O 取 C0 让公式统一
    return {k: (v[0] + C0[0], v[1] + C0[1], v[2]) if k != "earth" else (C0[0], C0[1], 1.0) for k, v in out.items()}


def to_root(name, x, y, z, ch):
    ox, oy, S = ch[name]
    return ox + S * (x - C0[0]), oy + S * (y - C0[1]), z / S


def camera(t):
    """当前镜头：根坐标里的中心和缩放"""
    ch = chain(t)
    if t <= KF[0][0]:
        k = KF[0]
        return to_root(k[1], k[2], k[3], k[4], ch)
    for a, b in zip(KF, KF[1:]):
        if a[0] <= t <= b[0]:
            u = ease_io((t - a[0]) / max(1e-6, b[0] - a[0]))
            x0, y0, z0 = to_root(a[1], a[2], a[3], a[4], ch)
            x1, y1, z1 = to_root(b[1], b[2], b[3], b[4], ch)
            lz = math.log(z1 / z0)
            if abs(lz) < 0.02:
                return lerp(x0, x1, u), lerp(y0, y1, u), z0 * math.exp(lz * u)
            z = z0 * math.exp(lz * u)
            fx, fy = (x1 * z1 - x0 * z0) / (z1 - z0), (y1 * z1 - y0 * z0) / (z1 - z0)
            return fx + (x0 - fx) * z0 / z, fy + (y0 - fy) * z0 / z, z
    k = KF[-1]
    return to_root(k[1], k[2], k[3], k[4], ch)


def drop_pulse(t):
    """音乐重音上镜头轻推一下"""
    p = 0.0
    for td, k in A.drop_times().items():
        if td <= t < td + 1.2:
            p = max(p, k * math.exp(-(t - td) / 0.35))
    return 1 + 0.014 * p


def level_matrix(name, cam, ch):
    cx, cy, z = cam
    ox, oy, S = ch[name]
    zl = z * S
    m = skia.Matrix()
    m.setTranslate((ox - S * C0[0] - cx) * z + C0[0], (oy - S * C0[1] - cy) * z + C0[1])
    m.preScale(zl, zl)
    return m, zl


# ======================================================================
# 各层的状态
# ======================================================================

def env(t, a, b, fade=0.35):
    """a~b 之间为 1，两头淡入淡出"""
    return ramp(t, a - fade, a) * (1 - ramp(t, b, b + fade))


def st_room(t):
    fa = [ramp(t, q, q + 0.35) for q in Q["f_items"]]
    hi = {
        "光能": env(t, Q["z_light"], 105.0, 0.3),
        2: env(t, Q["z_water"], 105.0, 0.3),
        6: env(t, Q["z_oxy"], 105.0, 0.3),
        0: env(t, Q["z_co2"], 105.0, 0.3),
        4: env(t, Q["z_co2"] + 0.8, 105.0, 0.3),
    }
    notes = {"o_from_w": ramp(t, Q["z_oxy"] + 0.2, Q["z_oxy"] + 1.2, lambda x: x),
             "c_to_s": ramp(t, Q["z_co2"] + 0.3, Q["z_co2"] + 1.3, lambda x: x)}
    # 回到式子时标题淡出，给式子上方的注释箭头让位置
    title_a = ramp(t, 0.2, 0.55) * (1 - env(t, 98.4, 200.0, 0.5))
    return dict(title_a=title_a, title_line=ramp(t, 0.4, 1.2), f_alpha=fa, hi=hi, notes=notes,
                mag=mag_state(t))


def st_leaf(t):
    return dict(lab_vein=env(t, Q["vein"], min(Q["vein"] + 3.2, 18.3)), lab_h2o=env(t, Q["h2o"], Q["h2o"] + 3.0))


def st_epi(t):
    op = 0.95
    op = lerp(op, 1.0, ramp(t, Q["open_"] - 0.1, Q["open_"] + 0.3))
    op = lerp(op, 0.18, ramp(t, Q["close_"] - 0.05, Q["close_"] + 0.35))
    op = lerp(op, 1.0, ramp(t, Q["close_"] + 0.55, Q["close_"] + 0.95))
    return dict(open_main=op, co2=1.0, lab_stoma=env(t, Q["stoma"], Q["close_"] + 0.3),
                lab_guard=env(t, Q["guard"], Q["close_"] + 0.3), lab_co2=env(t, Q["co2_epi"], Q["co2_epi"] + 2.6))


def st_cell(t):
    return dict(lab_cell=env(t, Q["cell"], Q["cell"] + 3.2), lab_chl=env(t, Q["chl"], 33.6, 0.3),
                hi_chl=ramp(t, Q["chl"], Q["chl"] + 0.6) * (0.6 + 0.4 * math.sin(t * 4)), co2=1.0)


def st_chloro(t):
    spec = {}
    a0 = Q["chl_mem"]
    if a0 - 0.6 < t < 49.6:
        spec = dict(
            a=env(t, a0 - 0.2, 48.8, 0.4),
            **{"in": ramp(t, a0, a0 + 1.4, lambda x: ease_out(x, 2))},
            absorb=ramp(t, Q["rb"] - 0.2, Q["rb"] + 0.6) * (1 - 0.5 * ramp(t, Q["so"], Q["so"] + 1.0)),
            lab_rb=env(t, Q["rb"], Q["green"] - 0.2, 0.3),
            out=ramp(t, Q["green"], Q["green"] + 0.9, lambda x: ease_out(x, 2)),
            lab_g=env(t, Q["green"] + 0.3, 48.7, 0.3),
            green_all=ramp(t, Q["so"], Q["so"] + 1.0),
        )
    return dict(spec=spec, lab_chl=env(t, 35.1, 37.0), lab_grana=env(t, Q["grana"], Q["grana"] + 3.2),
                lab_thyl=env(t, Q["thyl"], Q["thyl"] + 3.0), disc_hi=env(t, Q["thyl"], Q["thyl"] + 3.0))


# 背景里零星打在别的天线上的光子：(天线序号, 出发时刻)
AMB = [(i % len(L.ANTS), 51.5 + 2.3 * i + 0.9 * math.sin(i * 1.3)) for i in range(10)]


def st_mem(t):
    hit_t = Q["hit"]
    st = dict(
        photon=clamp((t - (hit_t - 0.55)) / 0.55) if t < hit_t + 0.05 else 0.0,
        hit=(ramp(t, hit_t - 0.05, hit_t + 0.05) * math.exp(-max(0.0, t - hit_t) / 1.4)),
        hop=(t - hit_t - 0.1) / 0.8 if hit_t + 0.1 <= t <= hit_t + 0.9 else -1.0,
        split=clamp((t - Q["split"]) / 2.4),
        mark=ramp(t, Q["iso"] - 0.4, Q["iso"] + 0.1) * (1 - ramp(t, 67.0, 67.6)),
        split2=clamp((t - Q["iso"] - 0.5) / 2.4) if t >= Q["iso"] + 0.5 else -1.0,
        spin=0.0,
        lab_light=env(t, hit_t - 0.3, hit_t + 1.9),
        lab_h2o=env(t, Q["split"] - 0.4, Q["split"] + 2.6),
        lab_mem=env(t, 50.6, 53.0),
        lab_atp=env(t, Q["atp"], Q["atp"] + 3.0),
        lab_mark=env(t, Q["iso"] - 0.2, Q["iso"] + 3.2),
    )
    # ATP 合酶从 P04 开始越转越快
    a0 = Q["atp"] - 1.0
    st["spin"] = 0.6 * t + (0.0 if t < a0 else 2.2 * (t - a0) ** 1.3 / (1 + 0.15 * (t - a0)))
    st["atps"] = [(t - (a0 + 0.6 + 0.9 * i)) / 2.2 for i in range(7)]
    st["hs"] = [(t - (a0 + 1.2 + 1.1 * i)) / 2.4 for i in range(5)]
    amb, ah = {}, {}
    for i, t0 in AMB:
        k = (t - t0) / 0.6
        if 0 < k < 1:
            amb[i] = k
        if k >= 1:
            ah[i] = max(ah.get(i, 0.0), math.exp(-(t - t0 - 0.6) / 0.8))
    st["amb_photons"], st["amb_hit"] = amb, ah
    # 基质：卡尔文循环
    st["ring_rot"] = 0.5 * t
    st["ring_glow"] = env(t, Q["k_loop"], Q["k_loop"] + 2.0, 0.5)
    st["arrow_a"] = ramp(t, Q["k_loop"] - 0.2, Q["k_loop"] + 0.4)
    st["co2s"] = [(t - (Q["k_co2"] + 0.5 * i)) / 1.6 for i in range(3)]
    st["feeds"] = [(t - (Q["k_energy"] - 0.3 + 0.45 * i)) / 1.7 for i in range(4)]
    st["sugars"] = [(t - (Q["k_sugar"] - 0.3 + 2.2 * i)) / 1.8 for i in range(6)]
    st["lab_co2"] = env(t, Q["k_co2"] + 0.9, Q["k_co2"] + 3.0)
    st["lab_sugar"] = env(t, Q["k_sugar"], Q["k_sugar"] + 2.6)
    st["lab_calvin"] = env(t, Q["k_calvin"], 88.3)
    st["chain"] = ramp(t, Q["k_chain"], Q["k_chain"] + 2.0, lambda x: x)
    st["starch"] = ramp(t, Q["k_starch"], Q["k_starch"] + 1.6, lambda x: x)
    st["cellulose"] = ramp(t, Q["k_cellu"], Q["k_cellu"] + 1.6, lambda x: x)
    st["lab_starch"] = env(t, Q["k_starch"] + 0.3, 94.6)
    st["lab_cellu"] = env(t, Q["k_cellu"] + 0.3, 94.6)
    return st


def st_earth(t):
    return dict(sparkle=env(t, Q["earth"] + 0.3, Q["night"], 0.8), night=ramp(t, Q["night"], Q["night"] + 2.8),
                question=ramp(t, Q["question"], Q["question"] + 0.5))


DRAW = {"earth": (L.draw_earth, st_earth), "room": (L.draw_room, st_room), "leaf": (L.draw_leaf, st_leaf),
        "epi": (L.draw_epi, st_epi), "cell": (L.draw_cell, st_cell), "chloro": (L.draw_chloro, st_chloro),
        "mem": (L.draw_mem, st_mem)}
DARK_LEVELS = {"cell", "chloro", "mem"}


# ======================================================================
# 渲染
# ======================================================================

def _screen_covered_by(path_screen):
    """裁剪形状（屏幕坐标）是否盖住整个画面：四个角都在里面"""
    return all(path_screen.contains(x, y) for x, y in ((0, 0), (W, 0), (0, H), (W, H)))


def plan_levels(t, cam, ch):
    """算出这一帧要画哪些层、每层的透明度和裁剪：返回 [(层名, 透明度, 裁剪路径（父层坐标）或 None)]"""
    z = cam[2]
    info = []
    for i, name in enumerate(ORDER):
        zl = z * ch[name][2]
        if i == 0:
            info.append((name, 1.0, None, True))
            continue
        parent_vis = info[-1] is not None and info[-1][3] is not None
        if name in FADE:
            a0, a1 = FADE[name]
            a = clamp(math.log(max(zl, 1e-9) / a0) / math.log(a1 / a0))
            info.append((name, ease_io(a), None, a > 0))
        elif name == "leaf":
            mg = mag_state(t)
            if mg is None:
                info.append((name, 0.0, None, False))
                continue
            p = skia.Path()
            p.addCircle(mg[0], mg[1], mg[2])
            a = ramp(t, *Q["lens_leaf"]) if t < 50 else 1.0
            info.append((name, a, p, a > 0))
        elif name == "cell":
            op = st_epi(t)["open_main"]
            info.append((name, 1.0, L.pore_path(op), zl >= 0.08))
    # 可见的最深一层
    d = 0
    for i, (name, a, clip, vis) in enumerate(info):
        if not vis:
            break
        d = i
    # 从最深往上找第一个盖满画面的层：它上面的层都不用画
    s_ = 0
    for i in range(d, 0, -1):
        name, a, clip, vis = info[i]
        if clip is None:
            if a >= 0.999:
                s_ = i
                break
        else:
            pm, _ = level_matrix(ORDER[i - 1], cam, ch)
            if a >= 0.999 and _screen_covered_by(xform(clip, pm)):
                s_ = i
                break
    return info, s_, d


@functools.lru_cache(maxsize=None)
def _sub_chunks():
    """字幕：长句在句中标点处拆开，每段的起止时间按配音停顿插值"""
    out = []
    for lid, text_, _sec, _gap in LINES:
        a, b = T[lid]
        pts = anchors(lid)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        segs, cur, start = [], "", 0
        for i, ch_ in enumerate(text_):
            cur += ch_
            brk = ch_ in "。！？；：" or (ch_ == "，" and len(cur.rstrip(PUNCT)) >= 9)
            if brk or i == len(text_) - 1:
                segs.append((start, i + 1, cur))
                cur, start = "", i + 1
        # 太短的段并到前一段
        merged = []
        for sg in segs:
            if merged and len(sg[2].strip(PUNCT)) <= 4 and len(merged[-1][2]) + len(sg[2]) <= 22:
                p0 = merged.pop()
                merged.append((p0[0], sg[1], p0[2] + sg[2]))
            else:
                merged.append(sg)
        for i0, i1, txt in merged:
            t0 = a + float(np.interp(i0, xs, ys))
            t1 = a + float(np.interp(i1, xs, ys))
            shown = txt.strip().rstrip("，。：；")
            out.append((t0 - 0.05, t1 + 0.12, shown))
    return out


def subtitle(c, t):
    if t >= Q["question"]:
        return
    for t0, t1, txt in _sub_chunks():
        if t0 <= t < t1:
            f = font(44, 600)
            w = f.measureText(txt)
            bar = skia.RRect.MakeRectXY(skia.Rect(W / 2 - w / 2 - 28, 978, W / 2 + w / 2 + 28, 1046), 14, 14)
            c.drawRRect(bar, fill((0.10, 0.08, 0.07), 0.55))
            text(c, txt, W / 2, 1027, f, CREAM, 1.0, "center")
            return


LAST_LABELS = []      # 最近一帧的标签（逐帧检查用）


def render_frame(t, with_subs=True):
    surf, c = new_canvas()
    ch = chain(t)
    cx, cy, z = camera(t)
    cam = (cx, cy, z * drop_pulse(t))
    info, s_, d = plan_levels(t, cam, ch)
    LABELS.clear()

    def draw_from(i, alpha_acc):
        name = ORDER[i]
        fn, stf = DRAW[name]
        st = stf(t)
        m, zl = level_matrix(name, cam, ch)
        # 这一层被子层淡入覆盖时，它的标签跟着淡出
        child_a = info[i + 1][1] if (i + 1 <= d and info[i + 1][2] is None) else 0.0
        if name == "room" and i + 1 <= d:
            # 镜片里的内容：先是“真放大镜”（把房间本身放大），再换成叶面那一层
            def lens(cc, x, y, R):
                cc.save()
                cc.translate(x, y)
                cc.scale(2.2, 2.2)
                cc.translate(-x, -y)
                L.draw_room(cc, t, {**st, "mag": None, "title_a": 0.0, "f_alpha": [0.0] * 7})
                cc.restore()
                a = info[i + 1][1]
                if a > 0:
                    cc.saveLayerAlpha(None, int(255 * a))
                    draw_from(i + 1, alpha_acc * a)
                    cc.restore()
            st["lens"] = lens
        elif name == "room":
            st["lens"] = lambda cc, x, y, R: (cc.save(), cc.translate(x, y), cc.scale(2.2, 2.2), cc.translate(-x, -y),
                                              L.draw_room(cc, t, {**st, "mag": None, "title_a": 0.0,
                                                                  "f_alpha": [0.0] * 7}), cc.restore())
        c.save()
        c.setMatrix(m)
        ZK["z"] = zl
        n_before = len(LABELS)
        fn(c, t, st)
        for lb in LABELS[n_before:]:
            lb["a"] *= alpha_acc * (1 - child_a)
        c.restore()
        if i + 1 <= d and name != "room":
            nm, a, clip, vis = info[i + 1]
            if clip is not None:
                c.save()
                c.setMatrix(m)
                c.clipPath(clip, doAntiAlias=True)
                c.resetMatrix()
                draw_from(i + 1, alpha_acc)
                c.restore()
            elif a > 0:
                c.saveLayerAlpha(None, int(255 * a))
                draw_from(i + 1, alpha_acc * a)
                c.restore()

    draw_from(s_, 1.0)
    # 屏幕空间后期：暗场斜纹、纸张颗粒
    deepest = ORDER[d]
    dark_k = 0.0
    if ORDER[s_] in DARK_LEVELS:
        dark_k = 1.0
    elif deepest in DARK_LEVELS and info[d][2] is None:
        dark_k = info[d][1]
    if dark_k > 0:
        dark_streaks(c, 0.10 * dark_k)
    finish_paper(c, 1.0 - 0.45 * dark_k)
    LAST_LABELS[:] = [dict(lb) for lb in LABELS]
    flush_labels(c)
    if with_subs:
        subtitle(c, t)
    return snapshot(surf)


def debug_plan(t):
    ch = chain(t)
    cam = camera(t)
    info, s_, d = plan_levels(t, cam, ch)
    return [ORDER[i] for i in range(s_, d + 1)], round(cam[2], 3)


if __name__ == "__main__":
    for tt in (0, 9, 11, 12.5, 19.5, 22, 27, 28.5, 34, 36, 45, 49.5, 50.5, 60, 73, 76, 90, 95.5, 96.5, 97.6, 104, 106.3,
               108, 115, 118):
        print(tt, debug_plan(tt))

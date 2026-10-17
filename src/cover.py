# 封面：按平台独立设计，不取视频帧
#   主题词“光合作用”做最大的主标题；上方一句对观众说的话，只抛疑问，不写答案
#   插画：一片大叶子，放大镜压在叶子上，镜片里是叶绿体内部（基粒 + 三道光打进去），一眼看出“钻进叶子里面”
#   B 站横版 16:9（右下角留给时长角标）、小红书横版 4:3、抖音竖版 9:16（关键内容放中间 3:4，避开底部文案区和右侧按钮）
#   python cover.py [文案 a/b/c]
import math
import sys
from pathlib import Path

import skia
from PIL import Image

import levels as L
from draw import (CREAM, GOLD, INK, LEAF_DK, PAPER, PAPER_WARM, ZK, C, fill, font, grain_img, line, rrect_pts,
                  smooth_path, wobble)

OUT = Path(__file__).resolve().parent.parent / "out" / "cover"
LOG = "[cover 封面]"

TITLE = "光合作用"
# 对观众说的话：(第一行, 第二行, 要压色块强调的词)
LINES = {
    "a": ("你呼吸的每一口氧气，", "是从哪儿来的？", "氧气"),
    "b": ("叶子为什么是绿色的，", "你真的知道吗？", "绿色"),
    "c": ("一片叶子里面，", "你猜藏着什么？", "藏着什么"),
}

# 各尺寸的版式：画布、对话句（左上角 x、第一行基线 y、字号、对齐）、主标题（x、基线、字号、对齐）、
# 镜片（中心、半径、手柄角度）、大叶子（叶柄位置、角度、长、半宽）、太阳（中心、半径）
LAYOUT = {
    "bili_16x9": dict(size=(1920, 1080), ask=(110, 330, 74, "left"), title=(100, 660, 240, "left"),
                      lens=(1480, 560, 290, 12), leaf=(380, 1200, -30, 1500, 330), sun=(1760, 120, 64)),
    "xhs_4x3": dict(size=(1440, 1080), ask=(80, 330, 62, "left"), title=(70, 630, 190, "left"),
                    lens=(1150, 620, 235, 12), leaf=(250, 1200, -32, 1250, 290), sun=(1320, 115, 58)),
    "douyin_9x16": dict(size=(1080, 1920), ask=(540, 380, 66, "center"), title=(540, 700, 230, "center"),
                        lens=(540, 1200, 330, 125), leaf=(-60, 1700, -42, 1350, 330), sun=(900, 210, 70)),
}


def paper(c, w, h, glow):
    c.drawRect(skia.Rect(0, 0, w, h), fill(PAPER))
    sh = skia.GradientShader.MakeRadial(skia.Point(*glow), max(w, h) * 0.7, [C(PAPER_WARM, 0.75), C(PAPER_WARM, 0)])
    c.drawRect(skia.Rect(0, 0, w, h), skia.Paint(Shader=sh))


def grain(c, w, h):
    """纸张颗粒：同一张颗粒图按需要铺满画布"""
    img = grain_img(3, 1.0)
    for y in range(0, h, img.height()):
        for x in range(0, w, img.width()):
            c.drawImage(img, x, y)


def lens_inside(c, x, y, R):
    """镜片里：叶绿体内部，红、绿、蓝紫三道光打在一摞基粒上，绿光被反射出来"""
    gx, gy, gn = L.GRANA_POS[L.GRANUM_TARGET]
    s = R / 430
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    c.translate(-gx - 60, -gy + 60)
    ZK["z"] = s
    spec = {"a": 1.0, "in": 1.0, "absorb": 1.0, "out": 1.0, "green_all": 0.0}
    L.draw_chloro(c, 1.3, {"spec": spec})
    ZK["z"] = 1.0
    c.restore()


def ask(c, x, y, size, line1, line2, hot, align):
    """对话句：两行；关键词底下压一道金色马克笔"""
    f = font(size, 800)
    lh = size * 1.32
    for i, s in enumerate((line1, line2)):
        w = f.measureText(s)
        x0 = x - w / 2 if align == "center" else x
        yy = y + i * lh
        if hot in s:
            k = s.index(hot)
            hx = x0 + f.measureText(s[:k])
            hw = f.measureText(hot)
            hp = smooth_path(wobble(rrect_pts(hx - size * 0.08, yy - size * 0.42, hw + size * 0.16, size * 0.55,
                                              size * 0.12, 2), "hot" + s, 2.0))
            c.drawPath(hp, fill(GOLD, 0.75))
        c.drawString(s, x0, yy, f, fill(INK))


def title(c, x, y, size, align):
    """主标题：特大号黑字 + 一道绿色手绘下划线"""
    f = font(size, 900)
    w = f.measureText(TITLE)
    x0 = x - w / 2 if align == "center" else x
    # 字后面一层米白描边，压在插画上也清楚
    c.drawString(TITLE, x0, y, f, skia.Paint(AntiAlias=True, Color=C(CREAM), Style=skia.Paint.kStroke_Style,
                                               StrokeWidth=size * 0.09, StrokeJoin=skia.Paint.kRound_Join))
    c.drawString(TITLE, x0, y, f, fill(INK))
    line(c, [(x0 + size * 0.05, y + size * 0.16), (x0 + w * 0.5, y + size * 0.13), (x0 + w - size * 0.05, y + size * 0.17)],
         LEAF_DK, size * 0.075, 1.0, seed="ttl")


def render(name, copy):
    lay = LAYOUT[name]
    w, h = lay["size"]
    surf = skia.Surface(w, h)
    c = surf.getCanvas()
    sx, sy, sr = lay["sun"]
    paper(c, w, h, (sx, sy))
    L.sun(c, sx, sy, sr, 1.0)
    bx, by, ba, bl, bw = lay["leaf"]
    L.leaf(c, bx, by, ba, bl, bw, "cover", sparkle=1.0, t=0.6)
    lx, ly, R, ang = lay["lens"]
    L.magnifier(c, lx, ly, R, ang, lambda cc: lens_inside(cc, lx, ly, R))
    l1, l2, hot = LINES[copy]
    ax, ay, asz, aal = lay["ask"]
    ask(c, ax, ay, asz, l1, l2, hot, aal)
    tx, ty, tsz, tal = lay["title"]
    title(c, tx, ty, tsz, tal)
    grain(c, w, h)
    arr = surf.makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)[..., :3]
    return Image.fromarray(arr)


def main(copy="a"):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in LAYOUT:
        img = render(name, copy)
        path = OUT / f"封面_{name}_{copy}.png"
        img.save(path)
        print(f"{LOG} {path} {img.size}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "a")

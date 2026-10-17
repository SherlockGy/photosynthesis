# 风格样张：每层场景出一张静帧，确认画风后再做镜头系统和成片
import sys
from pathlib import Path

from PIL import Image

import levels as L
import skia

from draw import CREAM, INK, W, fill, finish_paper, font, new_canvas, snapshot, text

OUT = Path(__file__).resolve().parent.parent / "out" / "style"


def subtitle(c, s):
    """字幕：半透明深色底条 + 米白字"""
    f = font(44, 600)
    w = f.measureText(s)
    bar = skia.RRect.MakeRectXY(skia.Rect(W / 2 - w / 2 - 28, 978, W / 2 + w / 2 + 28, 1046), 14, 14)
    c.drawRRect(bar, fill((0.10, 0.08, 0.07), 0.55))
    text(c, s, W / 2, 1027, f, CREAM, 1.0, "center")


def lens_leaf(c, x, y, R, t):
    """镜片里看到的是放大后的叶面"""
    c.save()
    s = 2 * R / 1080 * 1.05
    c.translate(x - 960 * s, y - 540 * s)
    c.scale(s, s)
    L.draw_leaf(c, t)
    c.restore()


def frame(name, t):
    surf, c = new_canvas()
    if name == "room":
        L.draw_room(c, t, magnify=(1130, 560, 150, 32), lens=lambda cc, x, y, R: lens_leaf(cc, x, y, R, t))
        finish_paper(c)
        subtitle(c, "光合作用可以写成一个式子")
    elif name == "leaf":
        L.draw_leaf(c, t)
        finish_paper(c, 0.6)
        subtitle(c, "水从根部吸上来，沿着叶脉，送到叶片的每个角落")
    elif name == "epi":
        L.draw_epi(c, t, openness=0.8, labels=1.0)
        finish_paper(c, 0.6)
        subtitle(c, "二氧化碳从叶片背面的气孔进来")
    elif name == "chloro":
        L.draw_chloro(c, t, labels=1.0)
        finish_paper(c, 0.5)
        subtitle(c, "叶绿体里一叠一叠的结构叫基粒")
    elif name == "mem":
        L.draw_mem(c, t, labels=1.0, photon_k=0.7, split=0.55)
        finish_paper(c, 0.5)
        subtitle(c, "叶绿素吸收的光能，先用来做一件事：把水分解")
    return snapshot(surf)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or ["room", "leaf", "epi", "chloro", "mem"]
    thumbs = []
    for n in names:
        img = Image.fromarray(frame(n, 1.3))
        img.save(OUT / f"{n}.png")
        thumbs.append(img.resize((960, 540)))
    sheet = Image.new("RGB", (1920, 540 * ((len(thumbs) + 1) // 2)), (20, 20, 20))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 2) * 960, (i // 2) * 540))
    sheet.save(OUT / "sheet.jpg", quality=90)
    print("ok", names)

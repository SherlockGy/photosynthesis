# 编排表：全片每一段音乐的织体、和弦走向和力度。
# 作曲（music.py）和画面（scenes.py）都读这一张表：画面的层级切换、推镜、光子击中和音乐的重音落在同一格。
#
# 配乐思路（为本片原创）：D 大调，96 BPM。明亮、好奇，跟着镜头一层层往里走
#   L 轻：钢琴分解和弦 + 拨弦 + 竖琴，沙锤八分，底鼓只在 1、3 拍轻轻一下（开场、喘口气）
#   G 律动：加贝斯、马林巴琶音、边击（鼓边，木头声）在 2、4 拍（叶面、表皮、回到式子）
#   M 深：进入暗场。垫子 + 低音弦乐铺底，竖琴慢琶音，钢琴高音零星落下，底鼓像心跳（叶绿体内部）
#   R 推：一小节的蓄力。定音鼓滚奏、弦乐渐强、竖琴上行，最后半拍全体停住
#   F 全开：重音（定音鼓 + 钢琴低八度 + 弦乐短奏）+ 鼓组加密 + 主旋律 + 弦乐；choir 再加合唱（地球）
#   C 循环：卡尔文循环。马林巴和拨弦的十六分循环音型，一圈一圈地转
#   O 收尾：只留钢琴，最后一个和弦停在空中，等观众回答
# 不用金属音色（没有钟、铃、钢片琴、钟琴、三角铁、镲、踩镲），不加音效
from timeline import BAR, BEAT, SECTION

# 和弦走向（D 大调）
PROG = {
    "OPEN": ["D", "A/C#", "Bm", "G"],     # I – V – vi – IV：开场、叶面、回到式子
    "DEEP": ["Bm", "G", "D", "A"],        # vi – IV – I – V：暗场里，安静但不阴郁
    "CYCLE": ["G", "A", "F#m", "Bm"],     # IV – V – iii – vi：循环，没有落到主和弦，一直往下转
    "WIDE": ["G", "D/F#", "Em", "A"],     # 地球：最开阔的一段
    "END": ["G", "A", "Bm", "Bm"],        # 收尾停在 vi 上，像一个问号
    "A": ["A"], "G": ["G"], "Bm": ["Bm"],
}

# (起始小节, 结束小节, 织体, 和弦走向, 力度, 色彩)；色彩 choir = 加合唱
PLAN = [
    (0, 4, "L", "OPEN", 0.8, ""),        # 式子、提问
    (4, 5, "R", "A", 0.75, ""),          # 放大镜推进镜片
    (5, 11, "G", "OPEN", 0.85, ""),      # 叶脉里的水、气孔
    (11, 15, "M", "DEEP", 0.85, ""),     # 钻进气孔：叶肉细胞、叶绿体、基粒
    (15, 19, "G", "DEEP", 0.75, ""),     # 为什么是绿色的
    (19, 20, "R", "A", 0.9, ""),         # 推进类囊体膜
    (20, 24, "F", "OPEN", 1.0, ""),      # 光子打中叶绿素，水被分解
    (24, 29, "G", "OPEN", 0.85, ""),     # 氧气来自水、ATP
    (29, 30, "L", "G", 0.7, ""),         # 平移到基质
    (30, 38, "C", "CYCLE", 0.85, ""),    # 卡尔文循环、糖、纤维素
    (38, 39, "R", "A", 0.85, ""),        # 一口气拉回盆栽
    (39, 42, "G", "OPEN", 0.8, ""),      # 回到式子
    (42, 43, "R", "A", 1.0, ""),         # 拉远
    (43, 45, "F", "WIDE", 1.1, "choir"), # 地球：全片最满
    (45, 48, "O", "END", 0.6, ""),       # 晚上还会放出氧气吗？
]

# 重音（小节, 强度 0~1）：画面在这一格轻推一下镜头；强度 ≥ 0.8 的加一道很淡的亮光
DROPS = {5: 0.5, 11: 0.4, 20: 1.0, 30: 0.4, 39: 0.6, 43: 1.0}


def _verify():
    """编排表里写死的小节号必须和时间线（由配音推出来）一致；配音或时间线变了这里会报错，不会悄悄错位"""
    expect = {"leaf": 5, "chloro": 11, "light": 20, "dark": 30, "sum": 39, "earth": 43, "end": 45}
    bad = {k: round(SECTION[k] / BAR, 3) for k, b in expect.items() if abs(SECTION[k] / BAR - b) > 0.01}
    assert not bad, f"[arrange 编排表] 小节号和时间线对不上：{bad}"
    for a, b in zip(PLAN, PLAN[1:]):
        assert abs(a[1] - b[0]) < 1e-9, f"[arrange 编排表] 段落不连续：{a} → {b}"


_verify()


def bar_time(b):
    return b * BAR


def block_at(b):
    for blk in PLAN:
        if blk[0] <= b < blk[1]:
            return blk
    return None


def texture(t):
    blk = block_at(t / BAR)
    return blk[2] if blk else "O"


def drop_times():
    return {bar_time(b): k for b, k in DROPS.items()}


def hold_windows():
    """蓄力小节最后半拍：乐队停住，画面也停住"""
    return [(bar_time(b1) - BEAT / 2, bar_time(b1)) for b0, b1, k, *_ in PLAN if k == "R"]


if __name__ == "__main__":
    for b0, b1, kind, prog, lv, color in PLAN:
        print(f"小节 {b0:5.1f}-{b1:5.1f}  {b0 * BAR:6.1f}s  {kind}  {prog:6s} 力度 {lv:.2f} {color}")

import csv
import json
import math
import os
from collections import deque
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# Paths and parameters
# ----------------------------------------------------------------------------
MODEL_PATH = "exp-3.pt"
VIDEO_PATH = "checks.mp4"
ZONES_JSON = "checkout_zones.json"
OUTPUT_VIDEO = "output_analytics.mp4"
OUTPUT_CSV = "service_records.csv"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Dashboard panel. Height MUST equal TARGET_HEIGHT so both can be stacked
# side by side into one output video.
DASH_W = 400
DASH_H = TARGET_HEIGHT
SS = 2  # dashboard supersampling factor (smooth edges)

# True  -> live view shows two separate windows (video + dashboard)
# False -> live view shows one combined window
SEPARATE_WINDOWS = True

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

# Any class whose name contains one of these is treated as staff.
# EVERYTHING else (customer, person, client, ...) is treated as a customer.
STAFF_KEYWORDS = ("cashier", "staff", "employee", "worker", "clerk", "teller")

# --- Zone membership ---------------------------------------------------------
MIN_OVERLAP = 0.30

# --- Temporal logic (seconds) ------------------------------------------------
QUEUE_GRACE_SEC = 1.5
MIN_QUEUE_DWELL_SEC = 0.5
SERVICE_GRACE_SEC = 0.6
MIN_SERVICE_SEC = 1.0

# --- Scene behaviour ---------------------------------------------------------
SERVICE_TARGET_SEC = 45     # service ring/bar is "full" at this many seconds
LONG_WAIT_WARN_SEC = 45     # queue ring/longest-wait turns red at this
STAND_SPEED = 30.0          # px/sec below which a person counts as standing
SPEED_WIN_SEC = 0.5         # window used to measure speed
TRAIL_SEC = 2.0             # length of movement trails
HEAT_SCALE = 4              # heatmap resolution divider

LEVEL_LOW_MAX = 2           # queue size thresholds for load level
LEVEL_MED_MAX = 5

# --- Palette (RGB) -----------------------------------------------------------
C_BG_TOP = (17, 20, 33)
C_BG_BOT = (10, 12, 20)
C_CARD = (25, 30, 47)
C_BORDER = (42, 50, 72)
C_TRACK = (40, 47, 70)
C_TEXT = (236, 239, 246)
C_MUTED = (139, 148, 170)
C_ACCENT = (56, 189, 248)
C_ACCENT2 = (167, 139, 250)
C_GREEN = (52, 211, 153)
C_AMBER = (251, 191, 36)
C_RED = (248, 113, 113)
C_ORANGE = (251, 146, 60)
C_SERVICE = (56, 189, 248)
C_STAFF = (167, 139, 250)
C_FREE = (148, 163, 184)
C_STAND = (226, 232, 240)

STATE_COLORS = {
    "staff": C_STAFF,
    "service": C_SERVICE,
    "queue": C_ORANGE,
    "standing": C_STAND,
    "walking": C_FREE,
    "free": C_FREE,
}


def bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


def blend(c1, c2, t):
    return tuple(int(a * t + b * (1 - t)) for a, b in zip(c1, c2))


def lane_color(n):
    """Traffic-light colour for a queue lane."""
    if n <= 0:
        return C_GREEN
    if n <= 2:
        return C_AMBER
    return C_RED


def load_level(ratio, total_q):
    if ratio >= 0.8 or total_q > LEVEL_MED_MAX:
        return "HIGH", C_RED
    if ratio >= 0.4 or total_q > LEVEL_LOW_MAX:
        return "MODERATE", C_AMBER
    return "LOW", C_GREEN


def progress_color(base, p):
    if p is None:
        return base
    if p >= 1.0:
        return C_RED
    if p >= 0.7:
        return C_AMBER
    return base


def mmss(sec):
    sec = int(max(0, sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ----------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------
_REG = ["segoeui.ttf", "SegoeUI.ttf", "arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
_BOLD = ["segoeuib.ttf", "arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]


@lru_cache(maxsize=None)
def get_font(size, bold=False):
    for name in (_BOLD if bold else _REG):
        try:
            return ImageFont.truetype(name, int(size))
        except OSError:
            continue
    try:
        return ImageFont.load_default(int(size))
    except TypeError:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Zones
# ----------------------------------------------------------------------------
def normalize_zone_type(raw, name=""):
    """Returns 'queue', 'service', or None if it can't be determined."""
    for text in (raw, name):
        t = str(text or "").strip().lower()
        if not t:
            continue
        if any(k in t for k in ("queue", "wait", "line")):
            return "queue"
        if any(k in t for k in ("service", "cashier", "counter", "checkout", "till")):
            return "service"
    return None


def load_zones(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    zones = []
    for z in data["zones"]:
        name = z.get("name", "") or "Zone"
        ztype = normalize_zone_type(z.get("type", ""), name)
        if ztype is None:
            print(f"[WARN] Zone '{name}' has no usable type -> ignored.")
            continue

        pts = np.array(z["points"], dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, TARGET_WIDTH - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, TARGET_HEIGHT - 1)

        mask = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        zones.append(
            {
                "name": name,
                "type": ztype,
                "points": pts,
                "mask": mask,
                "rect": cv2.boundingRect(pts),
            }
        )
    return zones


def zone_score(zone, box):
    """Return (is_inside, score). Uses feet point + lower-body overlap."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(TARGET_WIDTH - 1, x1))
    x2 = max(0, min(TARGET_WIDTH, x2))
    y1 = max(0, min(TARGET_HEIGHT - 1, y1))
    y2 = max(0, min(TARGET_HEIGHT, y2))
    if x2 <= x1 or y2 <= y1:
        return False, 0.0

    mask = zone["mask"]
    fx = (x1 + x2) // 2
    fy = min(TARGET_HEIGHT - 1, y2)
    feet_in = bool(mask[fy, fx]) or bool(mask[max(0, fy - 12), fx])

    ly1 = y1 + (y2 - y1) // 2
    roi = mask[ly1:y2, x1:x2]
    overlap = float(roi.mean()) if roi.size else 0.0

    inside = feet_in or overlap >= MIN_OVERLAP
    return inside, overlap + (1.0 if feet_in else 0.0)


def hit_test(x, y, zones, dets):
    """What is under the mouse? Person (smallest box wins) first, else zone."""
    best, best_area = None, None
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            a = (x2 - x1) * (y2 - y1)
            if best_area is None or a < best_area:
                best, best_area = d, a
    if best is not None:
        return ("person", best["id"])
    for z in zones:
        if cv2.pointPolygonTest(z["points"], (float(x), float(y)), False) >= 0:
            return ("zone", z["name"])
    return None


# ----------------------------------------------------------------------------
# Video panel drawing
# ----------------------------------------------------------------------------
def corner_box(frame, box, color, thickness=2, length=14):
    x1, y1, x2, y2 = box
    length = max(6, min(length, (x2 - x1) // 3, (y2 - y1) // 3))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
    ):
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness + 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness + 1, cv2.LINE_AA)


def draw_zones(frame, zones, queue_counts, hover_name=None, sel_name=None):
    """Queue lanes are tinted like a traffic light; counters are cyan."""
    def zcolor(z):
        if z["type"] == "service":
            return C_SERVICE
        return lane_color(queue_counts.get(z["name"], 0))

    overlay = frame.copy()
    for z in zones:
        cv2.fillPoly(overlay, [z["points"]], bgr(zcolor(z)))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    for z in zones:
        if z["name"] in (hover_name, sel_name):
            ov2 = frame.copy()
            cv2.fillPoly(ov2, [z["points"]], bgr(zcolor(z)))
            cv2.addWeighted(ov2, 0.16, frame, 0.84, 0, frame)

    for z in zones:
        col = bgr(zcolor(z))
        thick = 3 if z["name"] == hover_name else 2
        cv2.polylines(frame, [z["points"]], True, col, thick, cv2.LINE_AA)
        for p in z["points"]:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, col, -1, cv2.LINE_AA)
        if z["name"] == sel_name:
            cv2.polylines(frame, [z["points"]], True, (255, 255, 255), 3, cv2.LINE_AA)


def draw_trails(frame, ctx):
    ov = frame.copy()
    drawn = False
    for d in ctx["dets"]:
        p = ctx["people"].get(d["id"])
        if not p:
            continue
        pts = [(x, y) for _, x, y in p["trail"]]
        n = len(pts)
        if n < 2:
            continue
        col = bgr(d["color"])
        for i in range(1, n):
            cv2.line(ov, pts[i - 1], pts[i], col, 1 + int(3 * i / n), cv2.LINE_AA)
        drawn = True
    if drawn:
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)


def apply_heat(frame, heat):
    h = cv2.GaussianBlur(heat, (0, 0), 3)
    mx = float(h.max())
    if mx <= 0:
        return frame
    h = (h / mx * 255).astype(np.uint8)
    h = cv2.resize(h, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    cm = cv2.applyColorMap(h, cv2.COLORMAP_INFERNO)
    alpha = (h.astype(np.float32) / 255.0 * 0.7)[:, :, None]
    return (frame * (1 - alpha) + cm * alpha).astype(np.uint8)


def draw_ring(frame, d, debug=False):
    """Perspective progress ring around a person's feet."""
    x1, y1, x2, y2 = d["box"]
    fx, fy = (x1 + x2) // 2, y2
    rx = max(14, (x2 - x1) // 3)
    ry = max(6, rx // 3)
    base = bgr(d["color"])
    dark = tuple(int(v * 0.45) for v in base)
    cv2.ellipse(frame, (fx, fy), (rx, ry), 0, 0, 360, dark, 2, cv2.LINE_AA)
    p = d.get("progress")
    if p is not None:
        col = bgr(progress_color(d["color"], p))
        sweep = max(6, int(360 * min(1.0, p)))
        cv2.ellipse(frame, (fx, fy), (rx, ry), 0, -90, -90 + sweep, col, 3, cv2.LINE_AA)
    else:
        cv2.ellipse(frame, (fx, fy), (rx, ry), 0, 0, 360, base, 1, cv2.LINE_AA)
    if debug:
        cv2.circle(frame, (fx, fy), 4, (0, 0, 255), -1)


def render_labels(frame, labels):
    """Draws all pill labels in one pass with simple collision avoidance."""
    if not labels:
        return frame
    H, W = frame.shape[:2]
    base = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    placed = []

    def hit(r):
        for p in placed:
            if r[0] < p[2] + 3 and r[2] > p[0] - 3 and r[1] < p[3] + 2 and r[3] > p[1] - 2:
                return True
        return False

    def clamp(cx, cy, w, h):
        cx = int(max(2, min(W - w - 2, cx)))
        cy = int(max(2, min(H - h - 2, cy)))
        return (cx, cy, cx + w, cy + h)

    for lab in sorted(labels, key=lambda l: l["prio"]):
        size = lab.get("size", 12)
        font = get_font(size, True)
        w = int(d.textlength(lab["text"], font=font) + 28)
        h = size + 11

        gx1, gy1, gx2, gy2 = lab["geom"]
        if lab["kind"] == "zone":
            midx = (gx1 + gx2) / 2 - w / 2
            cands = [(midx, gy1 - h - 6), (midx, gy1 + 6), (midx, gy2 + 6), (gx1, gy1 - h - 6)]
        else:
            cands = [
                (gx1, gy1 - h - 4),
                (gx1, gy1 - 2 * h - 8),
                (gx2 - w, gy1 - h - 4),
                (gx1, gy2 + 4),
                (gx1 + 3, gy1 + 3),
            ]

        chosen = None
        for cx, cy in cands:
            r = clamp(cx, cy, w, h)
            if not hit(r):
                chosen = r
                break
        if chosen is None:
            chosen = clamp(cands[0][0], cands[0][1], w, h)
        placed.append(chosen)

        x1, y1, x2, y2 = chosen
        mid = (y1 + y2) / 2
        col = lab["color"]
        fill = (12, 14, 24, 230) if lab.get("emph") else (12, 14, 24, 214)
        d.rounded_rectangle((x1, y1, x2, y2), radius=h // 2, fill=fill,
                            outline=col + (255,), width=2 if lab.get("emph") else 1)
        d.ellipse((x1 + 9, mid - 3, x1 + 15, mid + 3), fill=col + (255,))
        d.text((x1 + 21, mid), lab["text"], font=font,
               fill=lab.get("text_color", C_TEXT) + (255,), anchor="lm")

    out = Image.alpha_composite(base, layer).convert("RGB")
    return np.ascontiguousarray(np.asarray(out)[:, :, ::-1])


# ----------------------------------------------------------------------------
# Dashboard (smooth, supersampled, TrueType)
# ----------------------------------------------------------------------------
_BASE = None


def get_base():
    global _BASE
    if _BASE is None:
        w, h = DASH_W * SS, DASH_H * SS
        t = np.linspace(0, 1, h)[:, None, None]
        arr = (np.array(C_BG_TOP) * (1 - t) + np.array(C_BG_BOT) * t) * np.ones((1, w, 1))
        arr = arr.astype(np.uint8)
        x = np.linspace(0, 1, w)[None, :, None]
        bar = np.array(C_ACCENT) * (1 - x) + np.array(C_ACCENT2) * x
        arr[: 4 * SS] = bar.astype(np.uint8)
        _BASE = Image.fromarray(arr, "RGB")
    return _BASE


class Painter:
    def __init__(self):
        self.s = SS
        self.img = get_base().copy()
        self.d = ImageDraw.Draw(self.img)

    def rect(self, x1, y1, x2, y2, fill=None, outline=None, r=0, width=1):
        s = self.s
        box = [x1 * s, y1 * s, x2 * s, y2 * s]
        if r:
            self.d.rounded_rectangle(box, radius=r * s, fill=fill, outline=outline, width=int(width * s))
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=int(width * s))

    def text(self, x, y, txt, size=12, fill=C_TEXT, bold=False, anchor="lm"):
        s = self.s
        self.d.text((x * s, y * s), str(txt), font=get_font(size * s, bold), fill=fill, anchor=anchor)

    def twidth(self, txt, size, bold=False):
        return self.d.textlength(str(txt), font=get_font(size * self.s, bold)) / self.s

    def circle(self, cx, cy, r, fill):
        s = self.s
        self.d.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], fill=fill)

    def arc(self, cx, cy, r, start, end, fill, width):
        s = self.s
        self.d.arc([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s],
                   start, end, fill=fill, width=int(width * s))

    def line(self, pts, fill, width=1):
        s = self.s
        self.d.line([(x * s, y * s) for x, y in pts], fill=fill, width=int(width * s), joint="curve")

    def alpha_poly(self, pts, rgba):
        s = self.s
        xs = [p[0] * s for p in pts]
        ys = [p[1] * s for p in pts]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1
        layer = Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon([(x * s - x1, y * s - y1) for x, y in pts], fill=rgba)
        self.img.paste(layer, (x1, y1), layer)

    def out(self):
        small = self.img.resize((DASH_W, DASH_H), Image.LANCZOS)
        return np.ascontiguousarray(np.asarray(small)[:, :, ::-1])


def section_title(P, x, y, title, right=None):
    P.text(x, y, title, 10.5, C_MUTED, True)
    if right:
        P.text(DASH_W - 20, y, right, 10, C_MUTED, False, anchor="rm")


def runs(arr):
    """Run-length encode an int array -> [(start, end, value)] for non-zero runs."""
    if len(arr) == 0:
        return []
    ch = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    starts = np.concatenate(([0], ch))
    ends = np.concatenate((ch, [len(arr)]))
    return [(int(s), int(e), int(arr[s])) for s, e in zip(starts, ends) if arr[s] != 0]


def draw_dashboard(ctx, UI, sel):
    P = Painter()
    zones = ctx["zones"]
    zi = ctx["zone_info"]
    st = ctx["stats"]
    fidx = ctx["frame_idx"]
    x0, x1 = 20, DASH_W - 20
    cw = x1 - x0
    limit = DASH_H - 48
    rows_hit = []

    s_zones = [z for z in zones if z["type"] == "service"]
    q_zones = [z for z in zones if z["type"] == "queue"]

    # ---- Header + progress ------------------------------------------------
    P.text(x0, 26, "CHECKOUT ANALYTICS", 17, C_TEXT, True)
    blink = (fidx // 15) % 2 == 0
    P.circle(x1 - 40, 26, 4, C_RED if blink else blend(C_RED, C_BG_TOP, 0.35))
    P.text(x1, 26, "LIVE", 11, C_MUTED, True, anchor="rm")
    total_time = ctx["total_time"]
    tt = f"{mmss(ctx['video_time'])} / {mmss(total_time)}" if total_time > 0 else mmss(ctx["video_time"])
    P.text(x0, 48, tt, 12, C_MUTED)
    P.rect(x0, 60, x1, 64, fill=C_TRACK, r=2)
    if total_time > 0:
        pw = max(5, cw * min(1.0, ctx["video_time"] / total_time))
        P.rect(x0, 60, x0 + pw, 64, fill=C_ACCENT, r=2)

    # ---- Hero: counters-busy gauge ---------------------------------------
    n_srv = len(s_zones)
    busy_n = sum(1 for z in s_zones if zi[z["name"]]["count"] > 0)
    ratio = busy_n / n_srv if n_srv else 0.0
    level, lcol = load_level(ratio, st["total_q"])
    P.rect(x0, 78, x1, 148, fill=C_CARD, outline=C_BORDER, r=14)
    gx, gy, gr = x0 + 42, 113, 24
    P.arc(gx, gy, gr, 0, 360, C_TRACK, 7)
    if ratio > 0:
        if ratio >= 0.999:
            P.arc(gx, gy, gr, 0, 360, lcol, 7)
        else:
            P.arc(gx, gy, gr, -90, -90 + 360 * ratio, lcol, 7)
    P.text(gx, gy, f"{busy_n}/{n_srv}" if n_srv else "--", 14, C_TEXT, True, anchor="mm")
    P.text(x0 + 84, 96, "COUNTERS BUSY", 10, C_MUTED, True)
    P.text(x0 + 84, 118, f"{level} LOAD", 20, lcol, True)
    P.text(x0 + 84, 136, f"{st['in_view']} customers  -  {st['cashiers']} cashiers", 11, C_MUTED)

    # ---- Four live tiles --------------------------------------------------
    y = 156
    gap = 8
    w4 = (cw - 3 * gap) / 4
    tiles = [
        ("IN VIEW", str(st["in_view"]), C_STAND),
        ("AT COUNTER", str(st["at_counter"]), C_SERVICE),
        ("WAITING", str(st["total_q"]) if q_zones else "--", C_AMBER),
        ("CASHIERS", str(st["cashiers"]), C_STAFF),
    ]
    for i, (label, value, col) in enumerate(tiles):
        cx = x0 + i * (w4 + gap)
        P.rect(cx, y, cx + w4, y + 48, fill=C_CARD, outline=C_BORDER, r=10)
        P.rect(cx + 10, y + 8, cx + 24, y + 10, fill=col, r=1)
        P.text(cx + 10, y + 19, label, 9, C_MUTED, True)
        P.text(cx + 10, y + 36, value, 18, C_TEXT, True)
    y += 48

    # ---- Zones list (click to inspect) -----------------------------------
    y += 18
    section_title(P, x0, y, "ZONES", "click a row to inspect")
    y += 10
    n = max(1, len(zones))
    row_h = max(19, min(23, 120 // n))
    max_bar = max(4, st["max_queue"])
    for z in zones:
        if y + row_h > limit - 200:
            break
        nm = z["name"]
        info = zi[nm]
        cy = y + row_h / 2
        is_sel = sel == ("zone", nm)
        if is_sel:
            P.rect(x0 - 6, y, x1 + 6, y + row_h, fill=blend(C_ACCENT, C_BG_TOP, 0.16),
                   outline=C_ACCENT, r=8)
        rows_hit.append((y, y + row_h, nm))
        name = nm if len(nm) <= 10 else nm[:9] + "."
        if z["type"] == "service":
            busy = info["count"] > 0
            col = C_SERVICE if busy else C_MUTED
            frac = min(1.0, info["timer"] / SERVICE_TARGET_SEC) if busy else 0.0
            bcol = progress_color(C_SERVICE, frac if busy else None)
            right = f"#{info['occ'][0][0]}  {mmss(info['timer'])}" if busy else "free"
        else:
            cnt = info["count"]
            col = lane_color(cnt)
            frac = min(1.0, cnt / max_bar)
            bcol = col
            right = f"{cnt} waiting" if cnt else "empty"
        P.circle(x0 + 5, cy, 4.5, col)
        P.text(x0 + 16, cy, name, 12, C_TEXT)
        bx1, bx2 = x0 + 100, x1 - 74
        P.rect(bx1, cy - 3.5, bx2, cy + 3.5, fill=C_TRACK, r=3.5)
        if frac > 0:
            P.rect(bx1, cy - 3.5, bx1 + max(7, (bx2 - bx1) * frac), cy + 3.5, fill=bcol, r=3.5)
        P.text(x1, cy, right, 11.5, C_TEXT if info["count"] else C_MUTED, info["count"] > 0, anchor="rm")
        y += row_h

    # ---- Counter activity timeline (with playhead) -----------------------
    y += 14
    tl_rows = zones[:8]
    if tl_rows and y + 40 < limit:
        section_title(P, x0, y, "ACTIVITY TIMELINE", "now |")
        y += 12
        trh = max(8, min(13, 66 // len(tl_rows)))
        tx0, tx1 = x0 + 62, x1
        tot = max(ctx["total_frames"], fidx, 1)
        top_y = y
        for z in tl_rows:
            nm = z["name"]
            P.text(x0, y + trh / 2, nm if len(nm) <= 9 else nm[:8] + ".", 9.5, C_MUTED)
            P.rect(tx0, y + 1, tx1, y + trh - 1, fill=C_TRACK, r=2)
            arr = np.frombuffer(bytes(ctx["timeline"][nm]), dtype=np.uint8)
            for s_, e_, v in runs(arr):
                xa = tx0 + (s_ / tot) * (tx1 - tx0)
                xb = max(xa + 1.5, tx0 + (e_ / tot) * (tx1 - tx0))
                if z["type"] == "service":
                    c = C_SERVICE
                else:
                    c = C_RED if v >= 2 else C_AMBER
                P.rect(xa, y + 1, xb, y + trh - 1, fill=c, r=1)
            y += trh
        px = tx0 + (fidx / tot) * (tx1 - tx0)
        P.line([(px, top_y - 2), (px, y + 1)], C_TEXT, 1.3)
        P.text(tx0, y + 9, "0:00", 9, C_MUTED)
        P.text(tx1, y + 9, mmss(tot / ctx["fps"]), 9, C_MUTED, anchor="rm")
        y += 14

    # ---- Insights OR selection details -----------------------------------
    y += 12
    if y + 60 < limit:
        rows = []
        title, right = "INSIGHTS", "right-click clears"
        if sel and sel[0] == "zone" and sel[1] in zi:
            nm = sel[1]
            info = zi[nm]
            title = f"SELECTED  -  {nm}"
            if info["type"] == "service":
                rows = [
                    ("Status", f"BUSY  #{info['occ'][0][0]}" if info["count"] else "FREE",
                     C_SERVICE if info["count"] else C_GREEN),
                    ("Current service", mmss(info["timer"]) if info["count"] else "--", C_TEXT),
                    ("Served here", str(info["served"]), C_TEXT),
                    ("Avg service here", f"{info['avg_service']:.0f}s" if info["avg_service"] else "--", C_TEXT),
                    ("Utilization", f"{info['util']:.0f}%", C_TEXT),
                ]
            else:
                ids = ", ".join(f"#{i}" for i, _ in info["members"][:3]) or "--"
                rows = [
                    ("Waiting now", str(info["count"]), lane_color(info["count"])),
                    ("Longest wait", mmss(info["timer"]) if info["count"] else "--", C_TEXT),
                    ("Who", ids, C_TEXT),
                ]
        elif sel and sel[0] == "person":
            tid = sel[1]
            p = ctx["people"].get(tid)
            title = f"SELECTED  -  #{tid}"
            if p is None:
                rows = [("Status", "not tracked", C_MUTED)]
            else:
                fps = ctx["fps"]
                in_view_now = (fidx - p["last"]) <= 2
                cur = next((d for d in ctx["dets"] if d["id"] == tid), None)
                if p["role"] == "staff":
                    state_txt = "Cashier"
                elif cur is None:
                    state_txt = "left view"
                elif cur["state"] in ("service", "queue"):
                    state_txt = f"{'At' if cur['state'] == 'service' else 'Queue'} {cur.get('zone', '')}"
                else:
                    state_txt = cur["state"].capitalize()
                rows = [
                    ("Role", p["role"].capitalize(), C_TEXT),
                    ("State", state_txt, C_SERVICE if in_view_now else C_MUTED),
                    ("In view", mmss((p["last"] - p["first"]) / fps), C_TEXT),
                    ("At counters", mmss(p["frames"]["service"] / fps), C_TEXT),
                    ("Standing / walking",
                     f"{p['frames']['standing'] / fps:.0f}s / {p['frames']['walking'] / fps:.0f}s", C_TEXT),
                ]
        else:
            avg = st["avg_service"]
            if avg > 0:
                avg_txt = f"{avg:.0f}s"
            elif st["live_avg_service"] > 0:
                avg_txt = f"~{st['live_avg_service']:.0f}s live"
            else:
                avg_txt = "--"
            ls_t, ls_z = st["longest_service"]
            free = st["free_counters"]
            free_txt = ("none" if not free else
                        (", ".join(free) if len(free) <= 2 else f"{len(free)} free"))
            cpc = st["in_view"] / st["cashiers"] if st["cashiers"] else None
            rows = [
                ("Served so far", f"{st['served']}  ({st['throughput']:.1f}/min)", C_TEXT),
                ("Avg service time", avg_txt, C_TEXT),
                ("Longest service now", f"{mmss(ls_t)}  {ls_z}" if ls_t > 0 else "--",
                 C_RED if ls_t >= SERVICE_TARGET_SEC else C_TEXT),
                ("Free counters", free_txt, C_GREEN if free else C_MUTED),
                ("Customers / cashier", f"{cpc:.1f}" if cpc is not None else "--", C_TEXT),
            ]
        section_title(P, x0, y, title, right)
        y += 6
        for label, value, col in rows:
            y += 20
            if y > limit:
                break
            P.text(x0, y, label, 12, C_MUTED)
            P.text(x1, y, value, 12.5, col, True, anchor="rm")
        y += 20

    # ---- Live events ------------------------------------------------------
    if y + 36 < limit and ctx["events"]:
        section_title(P, x0, y, "LIVE EVENTS")
        y += 6
        for t, txt, col in list(ctx["events"])[-3:][::-1]:
            y += 18
            if y > limit + 10:
                break
            P.circle(x0 + 4, y, 3, col)
            P.text(x0 + 14, y, mmss(t), 10.5, C_MUTED)
            P.text(x0 + 58, y, txt, 11.5, C_TEXT)

    # ---- Footer -----------------------------------------------------------
    if UI.get("debug") and ctx.get("debug_text"):
        P.text(x0, DASH_H - 44, ctx["debug_text"], 10, C_ACCENT)
    heat_on = "ON" if UI["heat"] else "off"
    trail_on = "ON" if UI["trails"] else "off"
    P.text(x0, DASH_H - 28, f"HEAT {heat_on}   -   TRAILS {trail_on}   -   {'PAUSED' if ctx.get('paused') else 'PLAYING'}",
           10, C_ACCENT, True)
    P.text(x0, DASH_H - 12, "Click select - 1-9 zone - H heat - T trails - . step - Q quit", 9.5, C_MUTED)

    ctx["dash_rows"] = rows_hit
    return P.out()


# ----------------------------------------------------------------------------
# Compose one output frame (video panel + dashboard) from a context dict.
# Called every frame AND while paused so hover/click stay responsive.
# ----------------------------------------------------------------------------
def compose(ctx, UI):
    frame = ctx["raw"].copy()
    zones, dets = ctx["zones"], ctx["dets"]
    fidx, fps = ctx["frame_idx"], ctx["fps"]

    hover = None
    m = UI.get("mouse")
    if m and 0 <= m[0] < TARGET_WIDTH and 0 <= m[1] < TARGET_HEIGHT:
        hover = hit_test(m[0], m[1], zones, dets)
    sel = UI.get("sel")

    if UI["heat"] and ctx.get("heat") is not None:
        frame = apply_heat(frame, ctx["heat"])

    hover_zone = hover[1] if hover and hover[0] == "zone" else None
    sel_zone = sel[1] if sel and sel[0] == "zone" else None
    draw_zones(frame, zones, ctx["queue_counts"], hover_zone, sel_zone)

    if UI["trails"]:
        draw_trails(frame, ctx)

    zi = ctx["zone_info"]
    labels = []

    for z in zones:
        info = zi[z["name"]]
        emph = z["name"] in (hover_zone, sel_zone)
        if z["type"] == "service":
            if info["count"]:
                txt = f"{z['name']}  ·  BUSY  ·  {mmss(info['timer'])}"
                col = C_SERVICE
            else:
                txt = f"{z['name']}  ·  FREE"
                col = C_GREEN
            if emph:
                txt += f"  ·  served {info['served']}"
        else:
            txt = f"{z['name']}  ·  {info['count']} waiting"
            col = lane_color(info["count"])
            if emph and info["count"]:
                txt += f"  ·  longest {mmss(info['timer'])}"
        x, y, w, h = z["rect"]
        labels.append({"kind": "zone", "text": txt, "color": col, "geom": (x, y, x + w, y + h),
                       "prio": -1 if emph else 0, "size": 13, "emph": emph})

    sel_person = sel[1] if sel and sel[0] == "person" else None
    hover_person = hover[1] if hover and hover[0] == "person" else None

    for d in dets:
        tid = d["id"]
        emph = tid in (sel_person, hover_person)
        c = bgr(d["color"])
        corner_box(frame, d["box"], (255, 255, 255) if tid == sel_person else c,
                   3 if emph else 2)
        draw_ring(frame, d, UI.get("debug", False))

        st_ = d["state"]
        t = d["timer"]
        if d["staff"]:
            txt, prio = f"{d['name'].title()} #{tid}", 1
        elif st_ == "service":
            txt, prio = f"#{tid}  ·  serving {int(t)}s", 2
        elif st_ == "queue":
            txt, prio = f"#{tid}  ·  waiting {int(t)}s", 3
        elif st_ == "standing":
            txt, prio = f"#{tid}  ·  standing {int(t)}s", 4
        elif st_ == "walking":
            txt, prio = f"#{tid}  ·  walking", 4
        else:
            txt, prio = f"#{tid}", 4
        if emph:
            p = ctx["people"].get(tid)
            if p:
                txt += f"  ·  in view {mmss((fidx - p['first']) / fps)}"
            prio = -2
        labels.append({"kind": "cust", "text": txt, "color": d["color"], "geom": d["box"],
                       "prio": prio, "size": 12, "emph": emph})

    frame = render_labels(frame, labels)
    dash = draw_dashboard(ctx, UI, sel)
    return frame, dash


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    import time
    from ultralytics import YOLO

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        return

    print("[INFO] Loading model and zones...")
    model = YOLO(MODEL_PATH)
    zones = load_zones(ZONES_JSON)
    print(f"[INFO] Model classes: {model.names}")
    print(f"[INFO] Zones: {[(z['name'], z['type']) for z in zones]}")

    queue_zones = [z for z in zones if z["type"] == "queue"]
    service_zones = [z for z in zones if z["type"] == "service"]
    if not zones:
        print("[ERROR] No usable zones in JSON.")
        return
    if not queue_zones:
        print("[INFO] No 'queue' zones -> WAITING tile is hidden; counters drive the stats.")
    if not service_zones:
        print("[WARN] No 'service' zone found -> counter stats will stay empty.")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_time = total_frames / fps if total_frames > 0 else 0.0

    queue_grace = int(QUEUE_GRACE_SEC * fps)
    queue_min_dwell = int(MIN_QUEUE_DWELL_SEC * fps)
    service_grace = max(3, int(SERVICE_GRACE_SEC * fps))
    trail_len = max(4, int(TRAIL_SEC * fps))
    speed_win = max(3, int(SPEED_WIN_SEC * fps))

    # ---- Interaction state --------------------------------------------------
    UI = {"mouse": None, "click": None, "sel": None, "heat": False, "trails": True, "debug": False}
    ST = {"paused": False, "step": False, "quit": False}
    holder = {"ctx": None}

    def on_mouse(event, x, y, flags, param):
        src = param
        if src == "combined":
            if x >= TARGET_WIDTH:
                src, x = "dash", x - TARGET_WIDTH
            else:
                src = "video"
        if src == "video" and event == cv2.EVENT_MOUSEMOVE:
            UI["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            UI["click"] = (src, x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            UI["click"] = ("clear", x, y)

    def process_click():
        c = UI["click"]
        if not c:
            return
        UI["click"] = None
        src, x, y = c
        ctx = holder["ctx"]
        if src == "clear":
            UI["sel"] = None
        elif ctx is not None and src == "video":
            hit = hit_test(x, y, zones, ctx["dets"])
            UI["sel"] = None if hit == UI["sel"] else hit
        elif ctx is not None and src == "dash":
            for y1, y2, nm in ctx.get("dash_rows", []):
                if y1 <= y <= y2:
                    hit = ("zone", nm)
                    UI["sel"] = None if hit == UI["sel"] else hit
                    break

    def handle_key(key):
        if key == 255:
            return
        ch = chr(key).lower() if 0 < key < 128 else ""
        if ch == "q":
            ST["quit"] = True
        elif ch == " ":
            ST["paused"] = not ST["paused"]
        elif ch == ".":
            if ST["paused"]:
                ST["step"] = True
        elif ch == "h":
            UI["heat"] = not UI["heat"]
        elif ch == "t":
            UI["trails"] = not UI["trails"]
        elif ch == "d":
            UI["debug"] = not UI["debug"]
        elif ch == "c" or key == 27:
            UI["sel"] = None
        elif ch.isdigit() and ch != "0":
            i = int(ch) - 1
            if i < len(zones):
                hit = ("zone", zones[i]["name"])
                UI["sel"] = None if hit == UI["sel"] else hit

    # ---- Windows ----------------------------------------------------------
    window_name = "Video Feed"
    dash_window = "Analytics Dashboard"
    if SEPARATE_WINDOWS:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, TARGET_WIDTH, TARGET_HEIGHT)
        cv2.moveWindow(window_name, 0, 0)
        cv2.namedWindow(dash_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(dash_window, DASH_W, DASH_H)
        cv2.moveWindow(dash_window, TARGET_WIDTH + 10, 0)
        cv2.setMouseCallback(window_name, on_mouse, "video")
        cv2.setMouseCallback(dash_window, on_mouse, "dash")
    else:
        window_name = "Queue & Cashier Analytics"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, TARGET_WIDTH + DASH_W, TARGET_HEIGHT)
        cv2.setMouseCallback(window_name, on_mouse, "combined")

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (TARGET_WIDTH + DASH_W, TARGET_HEIGHT),
    )

    def show(frame_out, dash_out):
        if SEPARATE_WINDOWS:
            cv2.imshow(window_name, frame_out)
            cv2.imshow(dash_window, dash_out)
        else:
            cv2.imshow(window_name, np.hstack([frame_out, dash_out]))

    # ---- Analytics state ----------------------------------------------------
    queue_state = {}     # id -> {"zone", "enter", "last"}
    service_state = {}   # id -> {"zone", "start", "last"}
    people = {}          # id -> track info (trail, role, time accumulators)
    seen_customers = set()

    service_records = []   # (id, zone, duration)
    wait_records = []      # seconds spent in queue before reaching a cashier
    events = deque(maxlen=60)
    history = deque(maxlen=max(60, int(fps * 20)))
    timeline = {z["name"]: bytearray() for z in zones}
    busy_frames = {z["name"]: 0 for z in service_zones}
    heat = np.zeros((TARGET_HEIGHT // HEAT_SCALE, TARGET_WIDTH // HEAT_SCALE), dtype=np.float32)
    max_queue = 0
    peak_counter = 0

    frame_idx = 0
    customers_seen = 0
    t_prev = time.time()
    fps_live = fps

    def finalize_service(t_id, data):
        dur = (data["last"] - data["start"]) / fps
        if dur >= MIN_SERVICE_SEC:
            service_records.append((t_id, data["zone"], round(dur, 2)))
            events.append((data["last"] / fps, f"#{t_id} left {data['zone']}  ·  {dur:.0f}s", C_GREEN))
            print(f"[SERVICE COMPLETE] Customer #{t_id} at {data['zone']}: {dur:.2f}s")

    print("[INFO] Running...  (click video/dashboard rows to inspect, H heat, T trails, SPACE pause)")
    while True:
        if ST["quit"]:
            break

        # ---- Paused: keep the UI responsive, re-render the same frame -------
        if ST["paused"] and not ST["step"]:
            process_click()
            if holder["ctx"] is not None:
                holder["ctx"]["paused"] = True
                f_out, d_out = compose(holder["ctx"], UI)
                show(f_out, d_out)
            handle_key(cv2.waitKey(30) & 0xFF)
            continue
        ST["step"] = False

        if not cap.isOpened():
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        raw = frame.copy()
        frame_idx += 1
        video_time = frame_idx / fps

        results = model.track(
            source=frame,
            persist=True,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        # ---- Collect detections -------------------------------------------
        detections = []
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.int().cpu().numpy()
            clss = results.boxes.cls.int().cpu().numpy()
            for box, tid, cls in zip(boxes, ids, clss):
                name = model.names[int(cls)]
                is_staff = any(k in name.lower() for k in STAFF_KEYWORDS)
                detections.append(
                    {
                        "id": int(tid),
                        "box": tuple(int(v) for v in box),
                        "name": name,
                        "staff": is_staff,
                        "state": "free",
                        "speed": None,
                    }
                )

        # ---- Track history (trails + speed) -------------------------------
        for d in detections:
            tid = d["id"]
            x1, y1, x2, y2 = d["box"]
            fx, fy = (x1 + x2) // 2, y2
            p = people.get(tid)
            if p is None:
                p = people[tid] = {
                    "role": "staff" if d["staff"] else "customer",
                    "first": frame_idx, "last": frame_idx,
                    "trail": deque(), "mstate": None, "mstate_since": frame_idx,
                    "frames": {"service": 0, "queue": 0, "standing": 0, "walking": 0},
                }
                if not d["staff"]:
                    seen_customers.add(tid)
            p["last"] = frame_idx
            p["trail"].append((frame_idx, fx, fy))
            while p["trail"] and frame_idx - p["trail"][0][0] > trail_len:
                p["trail"].popleft()
            ref = None
            for e in p["trail"]:
                if frame_idx - e[0] <= speed_win:
                    ref = e
                    break
            if ref is not None and frame_idx - ref[0] >= speed_win * 0.6:
                span = (frame_idx - ref[0]) / fps
                d["speed"] = math.hypot(fx - ref[1], fy - ref[2]) / span
            if not d["staff"]:
                hy, hx = min(heat.shape[0] - 1, fy // HEAT_SCALE), min(heat.shape[1] - 1, fx // HEAT_SCALE)
                heat[max(0, hy - 1):hy + 2, max(0, hx - 1):hx + 2] += 1.0

        # ---- Assign every customer to ONE zone (service wins over queue) ---
        in_view = 0
        cashiers = 0
        for d in detections:
            if d["staff"]:
                cashiers += 1
                continue
            customers_seen += 1
            in_view += 1

            best_service, best_s = None, 0.0
            for z in service_zones:
                inside, score = zone_score(z, d["box"])
                if inside and score > best_s:
                    best_service, best_s = z, score
            if best_service is not None:
                d["state"], d["zone"] = "service", best_service["name"]
                continue

            best_queue, best_q = None, 0.0
            for z in queue_zones:
                inside, score = zone_score(z, d["box"])
                if inside and score > best_q:
                    best_queue, best_q = z, score
            if best_queue is not None:
                d["state"], d["zone"] = "queue", best_queue["name"]

        # ---- Update state machines ----------------------------------------
        for d in detections:
            tid = d["id"]
            if d["state"] == "service":
                if tid not in service_state:
                    service_state[tid] = {"zone": d["zone"], "start": frame_idx, "last": frame_idx}
                    events.append((video_time, f"#{tid} at {d['zone']}", C_SERVICE))
                    if tid in queue_state:
                        wait_records.append((frame_idx - queue_state[tid]["enter"]) / fps)
                        del queue_state[tid]
                else:
                    service_state[tid]["last"] = frame_idx
                    service_state[tid]["zone"] = d["zone"]
            elif d["state"] == "queue":
                if tid in service_state:
                    continue
                if tid not in queue_state:
                    queue_state[tid] = {"zone": d["zone"], "enter": frame_idx, "last": frame_idx}
                    events.append((video_time, f"#{tid} joined {d['zone']}", C_AMBER))
                else:
                    queue_state[tid]["last"] = frame_idx
                    queue_state[tid]["zone"] = d["zone"]

        for tid in list(queue_state.keys()):
            if frame_idx - queue_state[tid]["last"] > queue_grace:
                del queue_state[tid]

        for tid in list(service_state.keys()):
            if frame_idx - service_state[tid]["last"] > service_grace:
                finalize_service(tid, service_state[tid])
                del service_state[tid]

        # ---- Final per-person state, timer, ring --------------------------
        for d in detections:
            tid = d["id"]
            p = people[tid]
            if d["staff"]:
                d["state"], d["timer"] = "staff", 0.0
            elif d["state"] == "service":
                s = service_state.get(tid)
                d["timer"] = (frame_idx - s["start"]) / fps if s else 0.0
            elif d["state"] == "queue":
                s = queue_state.get(tid)
                d["timer"] = (frame_idx - s["enter"]) / fps if s else 0.0
            else:
                sp, prev = d["speed"], p["mstate"]
                if sp is None:
                    new = None
                elif prev == "standing":
                    new = "walking" if sp > STAND_SPEED * 1.5 else "standing"
                elif prev == "walking":
                    new = "standing" if sp < STAND_SPEED * 0.6 else "walking"
                else:
                    new = "standing" if sp < STAND_SPEED else "walking"
                if new != prev:
                    p["mstate"], p["mstate_since"] = new, frame_idx
                d["state"] = new or "free"
                d["timer"] = (frame_idx - p["mstate_since"]) / fps
            if d["state"] in ("service", "queue"):
                p["mstate"], p["mstate_since"] = None, frame_idx
            if d["state"] in p["frames"]:
                p["frames"][d["state"]] += 1

            d["color"] = STATE_COLORS[d["state"]]
            if d["state"] == "service":
                pr = d["timer"] / SERVICE_TARGET_SEC
            elif d["state"] == "queue":
                pr = d["timer"] / LONG_WAIT_WARN_SEC
            else:
                pr = None
            d["progress"] = min(1.0, pr) if pr is not None else None

        # ---- Stable counts per zone ---------------------------------------
        queue_members = {z["name"]: [] for z in queue_zones}
        for tid, s in queue_state.items():
            if frame_idx - s["last"] <= queue_grace and (s["last"] - s["enter"]) >= queue_min_dwell:
                queue_members.setdefault(s["zone"], []).append((tid, (frame_idx - s["enter"]) / fps))
        queue_counts = {n: len(m) for n, m in queue_members.items()}

        service_active = {z["name"]: 0 for z in service_zones}
        occupants = {z["name"]: [] for z in service_zones}
        for tid, s in service_state.items():
            if frame_idx - s["last"] <= service_grace:
                service_active[s["zone"]] = service_active.get(s["zone"], 0) + 1
                occupants.setdefault(s["zone"], []).append((tid, (frame_idx - s["start"]) / fps))
        for zname, cnt in service_active.items():
            if cnt > 0:
                busy_frames[zname] = busy_frames.get(zname, 0) + 1

        total_q = sum(queue_counts.values())
        max_queue = max(max_queue, total_q)
        history.append(total_q)

        # ---- Per-zone info + timeline -------------------------------------
        zone_info = {}
        for z in zones:
            nm = z["name"]
            if z["type"] == "service":
                occ = sorted(occupants.get(nm, []), key=lambda t: -t[1])
                recs = [r[2] for r in service_records if r[1] == nm]
                zone_info[nm] = {
                    "type": "service", "count": len(occ), "occ": occ,
                    "timer": occ[0][1] if occ else 0.0,
                    "served": len(recs),
                    "avg_service": float(np.mean(recs)) if recs else 0.0,
                    "util": 100.0 * busy_frames.get(nm, 0) / max(1, frame_idx),
                    "members": [],
                }
                timeline[nm].append(1 if occ else 0)
            else:
                mem = sorted(queue_members.get(nm, []), key=lambda t: -t[1])
                zone_info[nm] = {
                    "type": "queue", "count": len(mem), "occ": [], "members": mem,
                    "timer": mem[0][1] if mem else 0.0,
                    "served": 0, "avg_service": 0.0, "util": 0.0,
                }
                timeline[nm].append(0 if not mem else (2 if len(mem) >= 3 else 1))

        at_counter = sum(zone_info[z["name"]]["count"] for z in service_zones)
        peak_counter = max(peak_counter, at_counter)

        # ---- Stats --------------------------------------------------------
        avg_service = float(np.mean([r[2] for r in service_records])) if service_records else 0.0
        inprog = [t for z in service_zones for _, t in zone_info[z["name"]]["occ"]]
        longest = (0.0, "")
        for z in service_zones:
            if zone_info[z["name"]]["timer"] > longest[0]:
                longest = (zone_info[z["name"]]["timer"], z["name"])
        stats = {
            "in_view": in_view,
            "cashiers": cashiers,
            "at_counter": at_counter,
            "total_q": total_q,
            "max_queue": max_queue,
            "peak_counter": peak_counter,
            "served": len(service_records),
            "avg_service": avg_service,
            "live_avg_service": float(np.mean(inprog)) if inprog else 0.0,
            "longest_service": longest,
            "free_counters": [z["name"] for z in service_zones if zone_info[z["name"]]["count"] == 0],
            "throughput": (len(service_records) / (video_time / 60.0)) if video_time >= 5 else 0.0,
            "unique_customers": len(seen_customers),
        }

        now = time.time()
        fps_live = 0.9 * fps_live + 0.1 * (1.0 / max(1e-6, now - t_prev))
        t_prev = now

        ctx = {
            "raw": raw, "zones": zones, "dets": detections,
            "queue_counts": queue_counts, "service_active": service_active,
            "zone_info": zone_info, "stats": stats, "history": history,
            "people": people, "events": events, "timeline": timeline,
            "frame_idx": frame_idx, "fps": fps, "video_time": video_time,
            "total_time": total_time, "total_frames": total_frames,
            "heat": heat, "dash_rows": [], "paused": ST["paused"],
            "debug_text": (f"DEBUG dets={len(detections)} q={len(queue_state)} "
                           f"s={len(service_state)} unique={len(seen_customers)} proc={fps_live:.0f}fps"),
        }
        holder["ctx"] = ctx

        if frame_idx == 90 and customers_seen == 0:
            print("[WARN] No customers detected in first 90 frames. Check model class names above "
                  "and STAFF_KEYWORDS / CONF_THRESHOLD.")

        # ---- Render, record, show -----------------------------------------
        f_out, d_out = compose(ctx, UI)
        out.write(np.hstack([f_out, d_out]))
        show(f_out, d_out)

        handle_key(cv2.waitKey(1) & 0xFF)
        process_click()

    for tid, data in service_state.items():
        finalize_service(tid, data)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "cashier_zone", "service_duration_sec"])
        w.writerows(service_records)

    print("\n" + "=" * 50)
    print("SESSION SUMMARY REPORT")
    print("=" * 50)
    for t_id, c_name, dur in service_records:
        print(f"Customer #{t_id} at {c_name} | Service Duration: {dur}s")
    if wait_records:
        print(f"Average queue wait : {np.mean(wait_records):.1f}s")
    print(f"Unique customers   : {len(seen_customers)}")
    print(f"Peak at counters   : {peak_counter}")
    print(f"Peak queue length  : {max_queue}")
    for n, f in busy_frames.items():
        print(f"Utilization {n:<12}: {100.0 * f / max(1, frame_idx):.0f}%")
    print(f"Annotated video    : {OUTPUT_VIDEO}")
    print(f"Service CSV        : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

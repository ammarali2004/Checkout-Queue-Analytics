# 🛒 Checkout Queue & Cashier Analytics

> **A customer waits 4 minutes at checkout. Your store never knows.**
> This project watches a checkout area from a single overhead camera and turns it into live, actionable analytics.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-111F68)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Tracking](https://img.shields.io/badge/Tracking-ByteTrack-38BDF8)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎬 Demo

[![Watch the demo on YouTube](https://img.youtube.com/vi/5daQEKE2jHU/maxresdefault.jpg)](https://youtu.be/5daQEKE2jHU)

▶️ **[Watch the full demo on YouTube](https://youtu.be/5daQEKE2jHU)**

<!-- Tip: add a dashboard screenshot at assets/preview.png and show it here:
![Dashboard preview](assets/preview.png) -->

---

## ✨ Features

| | Feature | What it does |
|---|---|---|
| 👁️ | **Detection + tracking** | Detects and tracks every customer and cashier (custom-trained YOLO + ByteTrack) |
| ⏱️ | **Live timers** | A running service/wait timer for every customer at every counter |
| 🚦 | **Traffic-light lanes** | Queue lanes tinted green (free), amber (filling up), red (crowded) |
| 🟢 | **Service rings** | A progress ring around each customer's feet that fills over time and turns amber, then red |
| 📊 | **Activity timeline** | Per-counter busy/idle history with a playhead |
| 🔥 | **Heatmap & trails** | Footfall heatmap and fading movement trails (toggle on/off) |
| 🖱️ | **Click to inspect** | Click any person, counter, or dashboard row to see its stats live |
| 🧭 | **Motion states** | Customers outside zones are classified as *standing* or *walking* from their speed |
| 📰 | **Live event feed** | `#6 at Counter2`, `#8 left Counter2 · 9s` |
| 📈 | **Dashboard** | Counters-busy gauge, LOW / MODERATE / HIGH load, in-view / at-counter / waiting / cashier tiles, insights |
| 🎞️ | **One combined output** | The video and the dashboard are saved side by side in a single MP4 |
| 🧾 | **CSV export** | Every completed service visit saved to `service_records.csv` |

---

## 🧠 How it works

1. **Detect & track:** YOLO detects people, ByteTrack gives each one a stable ID.
2. **Classify roles:** classes containing `cashier`, `staff`, `employee`, `worker`, `clerk` or `teller` are staff; everything else is a customer.
3. **Assign zones:** a customer is inside a zone if their feet point is inside the polygon **or** at least 30% of their lower body overlaps it. Counters (`service`) take priority over `queue` lanes so nobody is counted twice.
4. **State machines with grace periods:** short tracking dropouts do not break timers or flicker the counts.
5. **Stats:** wait time, service time, utilization, throughput, longest current service, free counters, customers per cashier, and more.
6. **Render:** zones, boxes, rings, trails and smart non-overlapping labels on the video, plus a smooth anti-aliased dashboard panel.

---

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

1. Put your trained weights, e.g. `exp-3.pt`, and your video, e.g. `checks.mp4`, in the project folder.
2. Copy `checkout_zones.example.json` to `checkout_zones.json` and edit the polygons (see below).
3. Check the paths at the top of `queue_analytics.py`:

```python
MODEL_PATH = "exp-3.pt"
VIDEO_PATH = "checks.mp4"
ZONES_JSON = "checkout_zones.json"
```

4. Run:

```bash
python queue_analytics.py
```

Results are saved to `output_analytics.mp4` and `service_records.csv`.

---

## 🗺️ Defining zones

Zones are polygons drawn on a **1280×720** canvas (the video is resized to this size).

```json
{
  "zones": [
    {
      "name": "Counter1",
      "type": "service",
      "points": [[960, 175], [1075, 115], [1160, 525], [1040, 525]]
    },
    {
      "name": "Queue A",
      "type": "queue",
      "points": [[100, 300], [400, 300], [400, 600], [100, 600]]
    }
  ]
}
```

- `type` is `"service"` (a counter where customers are served) or `"queue"` (a waiting lane).
- If `type` is blank, it is guessed from the **name** (`queue / wait / line` → queue, `counter / cashier / service / checkout / till` → service). Zones that cannot be classified are skipped with a warning.
- Only have counters and no waiting lanes? That works too. The dashboard is driven by counter activity and the **WAITING** tile shows `--`.

---

## 🎮 Controls

| Input | Action |
|---|---|
| **Left click** on a person or counter | Select it and show its details in the dashboard |
| **Left click** on a dashboard zone row | Select that zone |
| **Right click** / `C` / `Esc` | Clear selection |
| `1`–`9` | Select the N-th zone |
| `H` | Toggle footfall heatmap |
| `T` | Toggle movement trails |
| `Space` | Pause / resume (hover and click still work while paused) |
| `.` | Step one frame forward (while paused) |
| `D` | Debug overlay |
| `Q` | Quit |

The saved video records whatever is on screen at that moment (including heatmap, trails and selection).

---

## ⚙️ Configuration

All settings are constants at the top of `queue_analytics.py`.

| Setting | Default | Meaning |
|---|---|---|
| `CONF_THRESHOLD` / `IOU_THRESHOLD` | `0.35` / `0.5` | Detection confidence and NMS IoU |
| `STAFF_KEYWORDS` | cashier, staff, employee, worker, clerk, teller | Class names treated as staff |
| `MIN_OVERLAP` | `0.30` | Lower-body overlap needed to count as inside a zone |
| `QUEUE_GRACE_SEC` | `1.5` | Keeps someone in the queue through tracking dropouts |
| `MIN_QUEUE_DWELL_SEC` | `0.5` | Time in a lane before counting as waiting |
| `SERVICE_GRACE_SEC` / `MIN_SERVICE_SEC` | `0.6` / `1.0` | Service dropout buffer / minimum visit length |
| `SERVICE_TARGET_SEC` | `45` | Service ring or bar is "full" at this many seconds |
| `LONG_WAIT_WARN_SEC` | `45` | Wait ring turns red at this many seconds |
| `STAND_SPEED` | `30` px/s | Below this speed a person counts as standing |
| `SEPARATE_WINDOWS` | `True` | Two live windows (video + dashboard) or one combined |

---

## 📦 Outputs

| File | Description |
|---|---|
| `output_analytics.mp4` | Video + dashboard side by side (1680×720) |
| `service_records.csv` | `customer_id, cashier_zone, service_duration_sec` |
| Terminal summary | Unique customers, peak at counters, peak queue, per-counter utilization, average wait |

---

## 🗂️ Project structure

```
.
├── queue_analytics.py            # main script
├── checkout_zones.example.json   # example zone polygons
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Troubleshooting

- **Queue count is always 0:** make sure at least one zone has `"type": "queue"` (or a name containing `queue` / `wait` / `line`).
- **`SERVED` stays 0:** a service is only recorded when a customer *leaves* a counter zone. Short videos may end before anyone leaves, and the live timers and timeline still work.
- **No customers detected:** check the model class names printed at startup against `STAFF_KEYWORDS`, and lower `CONF_THRESHOLD`.
- **Fonts look basic:** the dashboard uses Segoe UI on Windows and falls back to Arial / DejaVu Sans.

---

## 🧰 Tech stack

Python · Ultralytics YOLO · ByteTrack · OpenCV · NumPy · Pillow

## 💡 Use cases

Smarter staffing · counter allocation · bottleneck detection · cashier workload analysis · customer experience monitoring.

---

## 🤝 Work with me

I build computer vision and edge AI systems for clients: real-time perception, detection, tracking, and analytics.

- 🎥 **YouTube demo:** https://youtu.be/5daQEKE2jHU
- 💼 **LinkedIn:** [linkedin.com/in/ammarali2004](https://www.linkedin.com/in/ammarali2004)
- 🟢 **Fiverr:** https://www.fiverr.com/s/RVVQrrl
- 🔵 **Upwork:** [My Upwork profile](https://www.upwork.com/freelancers/~0187875a9ecf3cddb3?mp_source=share)
- 📅 **Book a session (Topmate):** https://topmate.io/ammarali2004/

If this project helped you, consider giving it a ⭐

## 📄 License

MIT. See `LICENSE`.

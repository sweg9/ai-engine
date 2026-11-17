"""Draw keys (red), lamps (yellow when on, blue when off) and the press on a contact sheet."""
import json, sys
from PIL import Image, ImageDraw
from pathlib import Path
d = Path(sys.argv[1]); out = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
R = [json.loads(l) for l in open(d / "labels.jsonl")][:n]
W = 480; tiles = []
for r in R:
    im = Image.open(d / "images" / f"{r['image_id']}.jpg"); im = im.resize((W, int(im.height * W / im.width))); dr = ImageDraw.Draw(im)
    for dv in r["source"]["devices"]:
        if dv["bbox"]:
            x, y, w, h = dv["bbox"]; col = (255, 220, 0) if dv["state"] == "on" else (80, 160, 255)
            dr.rectangle([x * W, y * im.height, (x + w) * W, (y + h) * im.height], outline=col, width=2)
            dr.text((x * W + 2, (y + h) * im.height + 1), dv["name"][:12], fill=col)
    ev = ""
    for c in r["controls"]:
        x, y, w, h = c["bbox"]; dr.rectangle([x * W, y * im.height, (x + w) * W, (y + h) * im.height], outline=(255, 0, 0), width=2)
        if c["evidence"]:
            ev = f"press key {c['gang_index']} -> {c['evidence']['changed_device']} {c['evidence']['change']}"
    dr.text((4, 4), f"{r['image_id']} {r['room_type']}", fill=(255, 255, 0)); dr.text((4, 18), ev, fill=(255, 120, 120)); tiles.append(im)
cols = 3; H = tiles[0].height
m = Image.new("RGB", (cols * W, H * ((len(tiles) + cols - 1) // cols)), "gray")
for i, t in enumerate(tiles): m.paste(t, ((i % cols) * W, (i // cols) * H))
m.save(out, quality=85)

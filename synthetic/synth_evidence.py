"""Render before and after pairs of switch presses: which lamp turns on or off when a key is pressed.

Usage (Blender 4.2 or later, no addon needed):
    blender -b --factory-startup -P synth_evidence.py -- --out evidence/syn-e21 --scenes 40 --seed 21

A scene is a room corner: the wall with a door and one or two switch plates, side walls,
floor and ceiling, and two to five lamps (ceiling panel, downlights, pendant, floor lamp,
wall sconce, light under a wall cabinet). Every key runs one lamp; about one key in ten
runs an exhaust fan that cannot be seen. The camera stands in the room and sees the plate
and most lamps.

The scene is rendered once, then each key is pressed in turn and the scene is rendered
again, so a scene with k keys gives k + 1 photos and k before and after pairs. Every
record has the state of each key (the state of the lamp it runs), and the record before a
press has `evidence` for the pressed key. Lamp boxes and states are in `source.devices`.

Output: <out>/images/syn-e<seed>-<n>.jpg and <out>/labels.jsonl, in the labeling-spec
record format. Synthetic records are training data only (`split: revise`).
"""

import json
import math
import random
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default):
    return type(default)(ARGS[ARGS.index(name) + 1]) if name in ARGS else default


OUT = Path(arg("--out", "evidence"))
SCENES = arg("--scenes", 3)
SEED = arg("--seed", 21)
RES = (arg("--width", 1280), arg("--height", 960))


# ---------- scene helpers

def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            continue
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 90
    try:
        sc.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    return sc


def material(name, color, rough=0.5, metal=0.0, emit=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit is not None:
        bsdf.inputs["Emission Color"].default_value = (*emit, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
    return m


def noisy_material(name, color, scale, amount, rough=0.6):
    m = material(name, color, rough)
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tex = nt.nodes.new("ShaderNodeTexWave" if random.random() < 0.4 else "ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = scale
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = amount
    mix.inputs[1].default_value = (*color, 1)
    nt.links.new(tex.outputs["Color"], mix.inputs[2])
    nt.links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    return m


def box(name, size, loc, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat)
    return o


def cylinder(name, r, depth, loc, mat, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc, rotation=rot, vertices=32)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    return o


def gray(v, tint=0.0):
    return (v, v * (1 - tint * 0.3), v * (1 - tint))


WARM = [(1.0, 0.85, 0.65), (1.0, 0.93, 0.82), (0.95, 0.97, 1.0)]


# ---------- lamps. Each lamp has visible parts that glow and a light that lights the room.

class Lamp:
    def __init__(self, name, kind, parts, glow, light, power):
        self.name, self.kind, self.parts, self.glow, self.light, self.power = name, kind, parts, glow, light, power
        self.on = False

    def set(self, on):
        self.on = on
        self.light.data.energy = self.power if on else 0
        for m in self.glow:
            bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            bsdf.inputs["Emission Strength"].default_value = m["on_strength"] if on else 0


def glow_mat(name, color, strength):
    m = material(name, gray(0.95), 0.4, emit=color, strength=0)
    m["on_strength"] = strength
    return m


def add_light(kind, loc, color, size=0.3, rot=(0, 0, 0)):
    bpy.ops.object.light_add(type=kind, location=loc, rotation=rot)
    L = bpy.context.object
    L.data.color = color
    if kind == "AREA":
        L.data.size = size
    if kind == "SPOT":
        L.data.spot_size = math.radians(80)
    L.data.energy = 0
    return L


def ceiling_panel(i, x, y, h):
    c = random.choice(WARM)
    g = glow_mat(f"panel{i}", c, 6)
    s = random.uniform(0.4, 0.9)
    part = box(f"panel{i}", (s, s * random.uniform(0.5, 1), 0.03), (x, y, h - 0.02), g)
    return Lamp("ceiling_light", "ceiling", [part], [g], add_light("AREA", (x, y, h - 0.05), c, s), random.uniform(150, 400))


def downlights(i, x, y, h):
    c = random.choice(WARM)
    g = glow_mat(f"down{i}", c, 25)
    parts, n = [], random.randint(2, 4)
    for k in range(n):
        parts.append(cylinder(f"down{i}_{k}", 0.05, 0.02, (x + (k - (n - 1) / 2) * 0.7, y, h - 0.01), g))
    L = add_light("AREA", (x, y, h - 0.03), c, 0.7 * n)
    return Lamp("downlight", "ceiling", parts, [g], L, random.uniform(100, 300))


def pendant(i, x, y, h):
    c = random.choice(WARM)
    g = glow_mat(f"pend{i}", c, 15)
    drop = random.uniform(0.5, 0.9)
    cord = cylinder(f"cord{i}", 0.004, drop, (x, y, h - drop / 2), material("cord", gray(0.05)))
    shade = cylinder(f"shade{i}", random.uniform(0.1, 0.2), 0.15, (x, y, h - drop - 0.07), g)
    return Lamp("pendant_light", "pendant", [cord, shade], [g], add_light("POINT", (x, y, h - drop - 0.2), c), random.uniform(60, 200))


def floor_lamp(i, x, y, h):
    c = random.choice(WARM)
    g = glow_mat(f"floor{i}", c, 8)
    pole = cylinder(f"pole{i}", 0.012, 1.4, (x, y, 0.7), material("metal", gray(0.2), 0.3, 1.0))
    shade = cylinder(f"lampshade{i}", 0.18, 0.3, (x, y, 1.5), g)
    return Lamp("floor_lamp", "floor", [pole, shade], [g], add_light("POINT", (x, y, 1.45), c), random.uniform(50, 150))


def sconce(i, x, y, h):
    c = random.choice(WARM)
    g = glow_mat(f"sconce{i}", c, 12)
    z = random.uniform(1.7, 2.0)
    part = box(f"sconce{i}", (0.12, 0.08, 0.18), (x, -0.04, z), g)
    return Lamp("wall_light", "wall", [part], [g], add_light("POINT", (x, -0.2, z), c), random.uniform(30, 90))


def cabinet_light(i, x, y, h):
    c = random.choice(WARM)
    g = glow_mat(f"strip{i}", c, 20)
    w = random.uniform(1.0, 1.8)
    cab = box(f"cabinet{i}", (w, 0.35, 0.7), (x, -0.18, 1.85), noisy_material("cab", random.choice([gray(0.9), (0.55, 0.4, 0.25)]), 15, 0.2))
    box("counter", (w, 0.6, 0.9), (x, -0.3, 0.45), material("counter", gray(random.uniform(0.3, 0.8)), 0.4))
    strip = box(f"strip{i}", (w * 0.9, 0.03, 0.01), (x, -0.3, 1.495), g)
    L = add_light("AREA", (x, -0.25, 1.48), c, w * 0.8, rot=(0, 0, 0))
    L.rotation_euler = (math.pi, 0, 0)
    return Lamp("cabinet_light", "cabinet", [strip], [g], L, random.uniform(40, 120))


LAMP_TYPES = [ceiling_panel, downlights, pendant, floor_lamp, sconce, cabinet_light]


# ---------- plates (rocker keys or flat dark keys)

PLATE_COLORS = [gray(0.85), gray(0.7, 0.1), gray(0.35), gray(0.06), (0.75, 0.68, 0.55)]


def plate(x, z, n):
    color = random.choice(PLATE_COLORS)
    flat = n == 1 and random.random() < 0.4
    w, h = (0.075, 0.075) if flat else (0.04 * n + 0.04, 0.12)
    p = box("plate", (w, 0.008, h), (x, -0.004, z), material("plate", color, 0.35))
    km = material("key", gray(0.03) if flat else tuple(min(1, c * 1.04) for c in color), 0.1 if flat else 0.3)
    keys = []
    kw = (w * 0.78 - 0.004 * (n - 1)) / n
    for k in range(n):
        kx = x - w * 0.39 + kw / 2 + k * (kw + 0.004)
        keys.append(box("key", (kw, 0.006, h * 0.78), (kx, -0.011, z), km))
    return p, keys


# ---------- one scene

ROOM_OF = {"cabinet": "kitchen", "floor": "living_room", "pendant": "living_room", "wall": "bedroom", "ceiling": None}


def build(sc):
    W = random.uniform(4.5, 7)
    H = random.uniform(2.3, 2.6)
    wall_c = random.choice([gray(random.uniform(0.6, 0.92), random.uniform(0, 0.3)), (0.75, 0.66, 0.52), (0.55, 0.5, 0.45)])
    wm = noisy_material("wall", wall_c, random.uniform(2, 60), random.uniform(0, 0.3))
    box("wall", (W, 0.05, H), (0, 0.025, H / 2), wm)
    for sx in (-W / 2, W / 2):
        box("side", (0.05, 6, H), (sx, -3, H / 2), wm)
    box("back", (W, 0.05, H), (0, -6, H / 2), wm)
    box("floor", (W, 6, 0.02), (0, -3, 0), noisy_material("floor", random.choice([(0.55, 0.42, 0.3), gray(0.7), (0.4, 0.3, 0.22)]), 30, 0.5))
    box("ceiling", (W, 6, 0.02), (0, -3, H), material("ceiling", gray(0.92)))
    box("baseboard", (W, 0.015, 0.08), (0, -0.007, 0.04), material("base", gray(random.uniform(0.2, 0.95))))

    # door and plates beside it
    door_x = random.uniform(-W / 2 + 1.0, W / 2 - 1.0)
    fm = material("frame", gray(random.uniform(0.3, 0.95)), 0.4)
    for dx in (-0.47, 0.47):
        box("door_frame", (0.06, 0.08, 2.1), (door_x + dx, -0.02, 1.05), fm)
    box("door_frame", (1.0, 0.08, 0.06), (door_x, -0.02, 2.1), fm)
    box("door", (0.88, 0.04, 2.06), (door_x, 0.01, 1.03), noisy_material("door", wall_c, 10, 0.2))
    side = random.choice([-1, 1])
    px = door_x + side * random.uniform(0.62, 0.75)
    pz = random.uniform(1.1, 1.35)
    plates = [plate(px, pz, random.choices([1, 2, 3, 4], [0.3, 0.35, 0.25, 0.1])[0])]
    if random.random() < 0.3:
        p0 = plates[0][0]
        plates.append(plate(px + side * (p0.dimensions.x / 2 + 0.06), pz, random.choice([1, 2])))

    # lamps
    lamps = []
    n = random.randint(2, 5)
    kinds = random.sample(LAMP_TYPES, min(n, len(LAMP_TYPES)))
    for i, f in enumerate(kinds):
        x = random.uniform(-W / 2 + 0.6, W / 2 - 0.6)
        y = random.uniform(-4, -0.8) if f not in (sconce, cabinet_light) else 0
        if f is floor_lamp:
            x, y = random.choice([-W / 2 + 0.4, W / 2 - 0.4]), random.uniform(-3, -0.5)
        if f is cabinet_light and abs(x - door_x) < 1.6:
            x = door_x - math.copysign(2.0, x - door_x) if abs(door_x) < W / 2 - 2.5 else -door_x
        lamps.append(f(i, x, y, H))
    names = {}
    for l in lamps:  # unique names when two lamps have the same kind
        names[l.name] = names.get(l.name, 0) + 1
        if names[l.name] > 1:
            l.name = f"{l.name}_{names[l.name]}"

    # daylight from a window side, or evening
    evening = random.random() < 0.55
    bpy.ops.object.light_add(type="SUN", rotation=(random.uniform(0.6, 1.3), 0, random.uniform(-0.8, 0.8)))
    bpy.context.object.data.energy = random.uniform(0.2, 0.6) if evening else random.uniform(1.0, 3.0)
    w = bpy.data.worlds.new("world")
    sc.world = w
    w.use_nodes = True
    next(nd for nd in w.node_tree.nodes if nd.type == "BACKGROUND").inputs["Strength"].default_value = \
        random.uniform(0.1, 0.25) if evening else random.uniform(0.3, 0.8)
    sc.view_settings.exposure = random.uniform(1.0, 1.8) if evening else random.uniform(-0.3, 0.4)

    # camera inside the room, looking at the plate wall and up enough to see the ceiling lamps
    cam_loc = Vector((px + random.uniform(-1.2, 1.2), random.uniform(-3.2, -1.6), random.uniform(1.3, 1.7)))
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    aim = Vector((px * 0.6 + random.uniform(-0.6, 0.6), 0, random.uniform(1.3, 1.8)))
    cam.rotation_euler = (aim - cam_loc).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = random.uniform(16, 24)
    sc.camera = cam

    keys = [(pi, k) for pi, (p, ks) in enumerate(plates) for k in ks]
    wiring = {}
    for i, key in enumerate(keys):
        wiring[i] = "fan" if random.random() < 0.1 else random.randrange(len(lamps))
    for l in lamps:
        l.set(random.random() < 0.4)
    kinds_here = {l.kind for l in lamps}
    room = next((ROOM_OF[k] for k in ("cabinet", "wall", "floor", "pendant") if k in kinds_here),
                random.choice(["living_room", "bedroom", "hallway"]))
    return dict(plates=plates, keys=keys, lamps=lamps, wiring=wiring, cam=cam, room=room, evening=evening,
                door_x=door_x, cam_dist=(cam_loc - Vector((px, 0, pz))).length)


def bbox2d(sc, cam, objs):
    pts = [world_to_camera_view(sc, cam, o.matrix_world @ Vector(v)) for o in objs for v in o.bound_box]
    if any(p.z <= 0 for p in pts):
        return None
    xs = [p.x for p in pts]
    ys = [1 - p.y for p in pts]
    x0, x1, y0, y1 = max(0, min(xs)), min(1, max(xs)), max(0, min(ys)), min(1, max(ys))
    if x1 - x0 < 3 / RES[0] or y1 - y0 < 3 / RES[1]:
        return None
    return [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]


def center(o):
    # box() leaves the object at the origin with the offset baked into the mesh, so use the mesh centre
    return o.matrix_world @ (sum((Vector(v) for v in o.bound_box), Vector()) / 8)


def gang_order(sc, cam, ks):
    xs = [world_to_camera_view(sc, cam, center(k)).x for k in ks]
    order = sorted(range(len(ks)), key=lambda i: xs[i])
    return {k: n for n, k in enumerate(order)}


def main():
    random.seed(SEED)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    f = open(OUT / "labels.jsonl", "w", encoding="utf-8")
    n_img = 0
    for s in range(1, SCENES + 1):
        sc = clear()
        S = build(sc)
        bpy.context.view_layer.update()
        cam = S["cam"]
        fan_on = random.random() < 0.5

        # boxes do not change between presses; compute them once
        boxes = []
        for i, (pi, k) in enumerate(S["keys"]):
            p, ks = S["plates"][pi]
            boxes.append((bbox2d(sc, cam, [k]), bbox2d(sc, cam, [p]), gang_order(sc, cam, ks)[ks.index(k)], len(ks)))
        if not any(b[0] for b in boxes):
            continue
        dev_boxes = [bbox2d(sc, cam, l.parts) for l in S["lamps"]]

        def device(i):
            w = S["wiring"][i]
            return ("exhaust_fan", "fan") if w == "fan" else (S["lamps"][w].name, "light")

        def state(i):
            w = S["wiring"][i]
            return ("on" if fan_on else "off") if w == "fan" else ("on" if S["lamps"][w].on else "off")

        order = [i for i, b in enumerate(boxes) if b[0]]
        random.shuffle(order)
        shots = []
        for step in range(len(order) + 1):
            image_id = f"syn-e{SEED}-{n_img + 1:05d}"
            sc.view_settings.exposure += random.uniform(-0.05, 0.05)   # a phone camera re-exposes a little
            sc.render.filepath = str(OUT / "images" / f"{image_id}.jpg")
            bpy.ops.render.render(write_still=True)
            n_img += 1
            snap = {"id": image_id, "states": [state(i) for i in range(len(boxes))],
                    "devices": [{"name": l.name, "type": "light", "bbox": dev_boxes[j], "state": "on" if l.on else "off"}
                                for j, l in enumerate(S["lamps"])], "pressed": None}
            shots.append(snap)
            if step == len(order):
                break
            i = order[step]
            before = state(i)
            w = S["wiring"][i]
            if w == "fan":
                fan_on = not fan_on
            else:
                S["lamps"][w].set(not S["lamps"][w].on)
            snap["pressed"] = (i, device(i)[0], f"{before}_to_{state(i)}")

        for j, snap in enumerate(shots):
            controls = []
            for i, (kb, pb, gi, gc) in enumerate(boxes):
                if not kb:
                    continue
                name, dtype = device(i)
                near = abs(center(S["plates"][S["keys"][i][0]][0]).x - S["door_x"]) < 0.9
                ev = None
                if snap["pressed"] and snap["pressed"][0] == i:
                    ev = {"before_image": snap["id"], "after_image": shots[j + 1]["id"],
                          "changed_device": snap["pressed"][1], "change": snap["pressed"][2]}
                controls.append({
                    "kind": "switch", "source_label": None, "decoy": False, "plate_bbox": pb, "bbox": kb,
                    "gang_index": gi, "gang_count": gc, "occluded": False,
                    "height_m": round(center(S["keys"][i][1]).z, 2), "press_action": "press",
                    "press_point": [round(kb[0] + kb[2] / 2, 4), round(kb[1] + kb[3] / 2, 4)], "press_side": None,
                    "reachable": True, "state": snap["states"][i],
                    "context": {"label_text": None, "icon": None, "wall_position": "beside_door" if near else None,
                                "near_door": near, "near_bed": False, "nearby_objects": [],
                                "nearby_devices": sorted({d["name"] for d in snap["devices"] if d["bbox"]})},
                    "evidence": ev,
                    "role": {"device_id": f"{S['room']}_{name}", "device_type": dtype,
                             "scope": "main" if name.startswith(("ceiling_light", "downlight")) else "accent",
                             "effect": "toggle", "wireless_alt": False, "confidence": "certain"},
                })
            f.write(json.dumps({
                "image_id": snap["id"], "room_id": f"syn-e{SEED}-room{s}", "room_type": S["room"],
                "session": f"syn-e{SEED}", "split": "revise", "burst": f"syn-e{SEED}-b{s}", "consent": "synthetic",
                "capture": {"distance_m": round(S["cam_dist"], 2), "view_angle_deg": None,
                            "lighting": "dim" if S["evening"] and not any(d["state"] == "on" for d in snap["devices"]) else "bright"},
                "controls": controls, "status": "frozen",
                "source": {"dataset": "synthetic-evidence", "generator": "synth_evidence.py", "seed": SEED, "scene": s,
                           "devices": snap["devices"]},
            }, ensure_ascii=False) + "\n")
        f.flush()
    f.close()
    print(f"synthetic evidence: {n_img} images written to {OUT}")


main()

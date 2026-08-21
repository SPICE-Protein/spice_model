#!/usr/bin/env python
"""Generate publication figures and exchange files for the first-mut survivors.

Reads the 5 surviving 7QF3 (miniSOG) mutants from ``data/first_mut/*.npz``,
computes a CONSENSUS secondary-structure assignment from those coarse
C-alpha models themselves (>=4/5 agreement across the 5 independent MD
survivors), then writes:

  data/first_mut/figures/first_mut_structures.png   combined 5-panel figure (Figure 3,
                                                     episode-1 hotspot survivors 0-4;
                                                     bottom-right watermark, Arial font)
  data/first_mut/figures/first_mut_alkaline.png     3-panel supplementary figure (alkaline
                                                     pH-10 survivors 5-7; N-terminus rotated
                                                     toward camera so mutations stay visible)
  data/first_mut/figures/first_mut_M{i}.png         per-mutant render
  data/first_mut/figures/first_mut_M{i}.cif         mmCIF (Mol*/RCSB viewable,
                                                     SS written into
                                                     struct_conf /
                                                     struct_sheet_range)
  data/first_mut/figures/first_mut_M{i}.pse         PyMOL session

Honesty note: the secondary structure is derived from OUR OWN coarse
C-alpha structures (helix d3<6.3 A; beta-strand = extended geometry +
strand pairing), NOT from the reference crystal structure. The mmCIF
DETAILS field records this provenance explicitly.

Requirements: conda env `spice` (PyMOL 3.1, Pillow, numpy).
Usage:
  python scripts/plot_first_mut_figures.py
"""
import os
import numpy as np

import pymol
pymol.finish_launching(["pymol", "-cq"])
from pymol import cmd
from PIL import Image, ImageDraw, ImageFont

# Repo layout: model/scripts/this_file.py -> repo root = SPICE
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGO = os.path.join(REPO, "logo.png")

# 每批次: muts = (file, mutation positions, label, title, group)
# groups: vote=共识阈值, mut_scale=突变球尺度, face_nt=是否旋转露出 N 端, out=组合图名
BATCHES = {
    "first_mut": {
        "data_dir": "first_mut",
        "muts": [
            ("pseudo_7qf3_0_20.npz", [50, 102],      "M1", "50Q\u2192Y, 102V\u2192Y  Q=0.91", "hotspot"),
            ("pseudo_7qf3_1_20.npz", [52, 104],      "M2", "52T\u2192N, 104L\u2192N  Q=0.90", "hotspot"),
            ("pseudo_7qf3_2_20.npz", [50, 100],      "M3", "50Q\u2192Y, 100I\u2192N  Q=0.93", "hotspot"),
            ("pseudo_7qf3_3_20.npz", [50, 100, 102], "M4", "50Q\u2192W, 100I\u2192M, 102V\u2192E  Q=0.91", "hotspot"),
            ("pseudo_7qf3_4_20.npz", [50, 102, 103], "M5", "50Q\u2192W, 102V\u2192Y, 103Q\u2192S  Q=0.92", "hotspot"),
            ("pseudo_7qf3_5_20.npz", [1, 2, 3],      "M6", "1M\u2192A, 2E\u2192L, 3K\u2192T  (pH10)", "alkaline"),
            ("pseudo_7qf3_6_20.npz", [1, 2],         "M7", "1M\u2192P, 2E\u2192K  (pH10)", "alkaline"),
            ("pseudo_7qf3_7_20.npz", [1, 2],         "M8", "1M\u2192W, 2E\u2192Q  (pH10)", "alkaline"),
        ],
        "groups": {
            "hotspot":  {"vote": 4, "mut_scale": 0.9, "face_nt": False, "out": "first_mut_structures.png"},
            "alkaline": {"vote": 2, "mut_scale": 1.2, "face_nt": True,  "out": "first_mut_alkaline.png"},
        },
    },
    "second_mut": {
        "data_dir": "second_mut",
        "muts": [
            ("pseudo_7qf3_0_20.npz", [50],        "M1", "50Q\u2192K  Q=0.92  (pH2)", "acid"),
            ("pseudo_7qf3_1_20.npz", [4, 49],     "M2", "4S\u2192Y, 49D\u2192K  Q=0.90  (pH2)", "acid"),
            ("pseudo_7qf3_2_20.npz", [4, 36, 49], "M3", "4S\u2192F, 36L\u2192W, 49D\u2192K  Q=0.89  (pH2)", "acid"),
            ("pseudo_7qf3_3_20.npz", [3],         "M4", "3K\u2192Y  Q=0.92  (pH2)", "acid", {"face_nt": True, "scale": 1.2}),
            ("pseudo_7qf3_4_20.npz", [47],        "M5", "47E\u2192K  Q=0.91  (pH2)", "acid"),
        ],
        "groups": {
            "acid": {"vote": 4, "mut_scale": 1.2, "face_nt": False, "out": "second_mut_structures.png"},
        },
    },
}

AA3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
       "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
       "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
       "W": "TRP", "Y": "TYR"}


# ---------------- consensus secondary structure ----------------
def helix_mask(c, thr=6.3):
    """Calpha-only helix signal: i..i+3 distance below ~helix pitch."""
    L = len(c); m = np.zeros(L, bool)
    for i in range(L - 3):
        if np.linalg.norm(c[i + 3] - c[i]) < thr:
            m[i] = True
    return m


def sheet_paired(c, d3_thr=6.8, d2_thr=6.2, min_len=3, contact=6.5, min_pairs=3):
    """Beta-sheet by extended geometry AND strand pairing (isolated
    extended loops do not count)."""
    L = len(c); ext = np.zeros(L, bool)
    for i in range(L - 3):
        d2 = np.linalg.norm(c[i + 2] - c[i]); d3 = np.linalg.norm(c[i + 3] - c[i])
        if d3 > d3_thr and d2 > d2_thr:
            ext[i] = True
    ext[-1] = ext[-2] = False
    strands = []; i = 0
    while i < L:
        if ext[i]:
            j = i
            while j < L and ext[j]: j += 1
            if j - i >= min_len: strands.append((i, j - 1))
            i = j
        else:
            i += 1
    paired = set()
    for a in range(len(strands)):
        for b in range(a + 1, len(strands)):
            s1, s2 = strands[a]; t1, t2 = strands[b]; cnt = 0
            for i in range(s1, s2 + 1):
                for j in range(t1, t2 + 1):
                    if abs(i - j) > 5 and np.linalg.norm(c[i] - c[j]) < contact:
                        cnt += 1
            if cnt >= min_pairs:
                paired.update(range(s1, s2 + 1)); paired.update(range(t1, t2 + 1))
    return paired


def clean_runs(mask, minrun):
    """Keep only contiguous runs >= minrun."""
    out = np.zeros_like(mask); i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            if j - i >= minrun: out[i:j] = True
            i = j
        else:
            i += 1
    return out


def consensus_ss(coords_list, vote=4, h_minrun=4, e_minrun=3):
    """Per-residue SS agreed by >=vote/len(coords_list) structures.
    Returns 'H'/'S'/'L' (S = PyMOL strand code)."""
    n = len(coords_list); L = len(coords_list[0])
    H = np.zeros((n, L), bool); E = np.zeros((n, L), bool)
    for k, c in enumerate(coords_list):
        H[k] = helix_mask(c)
        for r in sheet_paired(c):
            E[k, r] = True
    h = clean_runs(H.sum(0) >= vote, h_minrun)
    e = clean_runs(E.sum(0) >= vote, e_minrun)
    return ["H" if h[i] else ("S" if e[i] else "L") for i in range(L)]


def kabsch_to_ref(mov, ref):
    pc = mov - mov.mean(0); tc = ref - ref.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ tc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    uf = u.copy(); uf[:, -1] *= d
    r = vt.T @ uf.T
    return pc @ r.T + ref.mean(0)


def write_ca_pdb(coords, path):
    lines = []
    for i, (x, y, z) in enumerate(coords):
        resi = i + 1
        lines.append(f"ATOM  {i + 1:5d}  CA  ALA A{resi:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------- PyMOL panel render + session ----------------
def rotmat_to_z(d, sign=1):
    """Rotation matrix mapping unit vector d onto sign*(0,0,1)."""
    import math
    z = np.array([0.0, 0.0, float(sign)])
    ax = np.cross(d, z); axn = ax / np.linalg.norm(ax)
    ang = math.acos(np.clip(np.dot(d, z), -1.0, 1.0))
    ux, uy, uz = axn; th = ang; ct, st = math.cos(th), math.sin(th); oc = 1 - ct
    return np.array([
        ct + ux*ux*oc, ux*uy*oc - uz*st, ux*uz*oc + uy*st,
        uy*ux*oc + uz*st, ct + uy*uy*oc, uy*uz*oc - ux*st,
        uz*ux*oc - uy*st, uz*uy*oc + ux*st, ct + uz*uz*oc]).reshape(3, 3)


def render_panel(coords, mut_pos, ss, png_path, pse_path, size=1400, mut_scale=1.0, face_nt=False):
    pdb = png_path + ".pdb"
    write_ca_pdb(coords, pdb)
    cmd.reinitialize()
    cmd.load(pdb, "m")
    L = len(coords)
    for j in range(L - 1):
        cmd.bond(f"m and id {j + 1}", f"m and id {j + 2}")
    for i, s in enumerate(ss):
        cmd.alter(f"m and resi {i + 1}", f"ss='{s}'")
    cmd.rebuild()
    cmd.set("cartoon_trace", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_discrete_colors", "on")
    cmd.set("cartoon_rect_length", 2.8)
    cmd.set("cartoon_rect_width", 0.7)
    cmd.show("cartoon")
    cmd.color("red", "m and ss h")
    cmd.color("0xe8c020", "m and ss s")
    cmd.color("0x3fa4a8", "m and ss l")
    mut_sel = "m and resi " + "+".join(str(p) for p in mut_pos)
    cmd.show("spheres", mut_sel)
    cmd.color("0x9c27b0", mut_sel)
    cmd.set("sphere_scale", mut_scale, mut_sel)
    # N 端突变: 先旋转对象让 N 端朝前, 球不用做大也能可见
    if face_nt:
        nt = (coords[0] + coords[1] + coords[2]) / 3
        d = nt - coords.mean(0); d = d / np.linalg.norm(d)
        R = rotmat_to_z(d, sign=1)
        mat = np.eye(4); mat[:3, :3] = R
        mat[:3, 3] = coords.mean(0) - R @ coords.mean(0)
        cmd.transform_selection("m", mat.flatten().tolist(), 1)
    cmd.show("spheres", "m and resi 1")
    cmd.color("0x2ca02c", "m and resi 1")
    cmd.set("sphere_scale", 0.4, "m and resi 1")
    cmd.show("spheres", f"m and resi {L}")
    cmd.color("0xff7f0e", f"m and resi {L}")
    cmd.set("sphere_scale", 0.4, f"m and resi {L}")
    cmd.zoom("m", 2.6)
    cmd.orient("m")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("ray_trace_mode", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)
    cmd.set("two_sided_lighting", 0)
    cmd.ray(size, size)
    cmd.png(png_path, dpi=300)
    cmd.save(pse_path)          # interactive PyMOL session


# ---------------- mmCIF export (Mol*/RCSB compatible) ----------------
def find_runs(mask):
    runs = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            runs.append((i, j - 1)); i = j
        else:
            i += 1
    return runs


def write_mmcif(coords, seq, ss, path, title):
    L = len(coords)
    ss_a = np.array(ss)
    lines = [f"data_{title}", "#", f"_entry.id {title}", "#"]
    lines.append("_struct.title  'SPICE coarse Calpha model of miniSOG 7QF3 mutant ({} residue)'".format(L))
    lines.append("_struct.pdbx_DETAILS 'Coarse C-alpha model. Secondary structure = geometric consensus of 5 SPICE MD survivors (>=4/5).'")
    lines.append("#")
    lines.append("_entity.id 1")
    lines.append("_entity.type 'polymer'")
    lines.append("_entity.src_method 'man'")
    lines.append("#")
    lines.append("_entity_poly.entity_id 1")
    lines.append("_entity_poly.type 'polypeptide(L)'")
    lines.append("_entity_poly.pdbx_seq_one_letter_code_can")
    lines.append(";")
    lines.append(seq)
    lines.append(";")
    lines.append("#")
    hx = find_runs(ss_a == 'H')
    if hx:
        lines.append("loop_")
        for c in ["_struct_conf.id", "_struct_conf.conf_type_id",
                  "_struct_conf.beg_label_comp_id", "_struct_conf.beg_label_asym_id",
                  "_struct_conf.beg_label_seq_id", "_struct_conf.end_label_comp_id",
                  "_struct_conf.end_label_asym_id", "_struct_conf.end_label_seq_id"]:
            lines.append(c)
        for k, (a, b) in enumerate(hx, 1):
            lines.append(f"HX_{k} HELIX_P {AA3[seq[a]]} A {a+1} {AA3[seq[b]]} A {b+1}")
        lines.append("#")
    sh = find_runs(ss_a == 'S')
    if sh:
        lines.append("loop_")
        for c in ["_struct_sheet_range.id", "_struct_sheet_range.sheet_id",
                  "_struct_sheet_range.beg_label_comp_id", "_struct_sheet_range.beg_label_asym_id",
                  "_struct_sheet_range.beg_label_seq_id", "_struct_sheet_range.end_label_comp_id",
                  "_struct_sheet_range.end_label_asym_id", "_struct_sheet_range.end_label_seq_id"]:
            lines.append(c)
        for k, (a, b) in enumerate(sh, 1):
            lines.append(f"ST_{k} S{k} {AA3[seq[a]]} A {a+1} {AA3[seq[b]]} A {b+1}")
        lines.append("#")
    lines.append("loop_")
    for c in ["_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
              "_atom_site.label_atom_id", "_atom_site.label_comp_id",
              "_atom_site.label_asym_id", "_atom_site.label_entity_id",
              "_atom_site.label_seq_id", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
              "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv"]:
        lines.append(c)
    for i, (x, y, z) in enumerate(coords):
        resi = i + 1
        lines.append(f"ATOM  {i+1:5d} C CA {AA3[seq[i]]} A 1 {resi:4d} "
                     f"{x:8.3f} {y:8.3f} {z:8.3f} 1.00 0.00")
    lines.append("#")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------- combined figure + watermark ----------------
def font_best(size):
    """Full-coverage fonts (so '->'-style glyphs render, no tofu boxes)."""
    for fp in ("/Library/Fonts/Arial Unicode.ttf",
               "/System/Library/Fonts/Hiragino Sans GB.ttc",
               "/System/Library/Fonts/STHeiti Medium.ttc"):
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def compose(panel_paths, titles, out_path):
    panels = [Image.open(p).convert("RGBA") for p in panel_paths]
    n = len(panels)
    pw, ph = panels[0].size
    gap = int(pw * 0.05)
    title_h = int(ph * 0.09)
    wm_h = int(ph * 0.13)          # bottom watermark strip
    canvas_w = pw * n + gap * (n - 1)
    canvas_h = title_h + ph + wm_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    tf = font_best(int(ph * 0.045))
    for i, p in enumerate(panels):
        x = i * (pw + gap); y = title_h
        canvas.paste(p, (x, y), p)
        draw.text((x + int(pw * 0.02), int(ph * 0.015)), titles[i],
                  font=tf, fill=(30, 30, 30, 255))
    # watermark: bottom-right, [logo] Folded by SPICE
    logo = Image.open(LOGO).convert("RGBA")
    logo_h = int(ph * 0.10)
    logo_w = int(logo_h * logo.size[0] / logo.size[1])
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)   # must actually resize!
    wf = font_best(int(ph * 0.042))
    text = "Folded by SPICE"
    tw = draw.textlength(text, font=wf)
    margin = int(ph * 0.02)
    total_w = logo_w + int(ph * 0.012) + int(tw)
    x0 = canvas_w - margin - total_w
    y0 = title_h + ph + (wm_h - logo_h) // 2
    canvas.paste(logo, (int(x0), int(y0)), logo)
    tx = x0 + logo_w + int(ph * 0.012)
    ty = title_h + ph + (wm_h - int(ph * 0.042)) // 2
    draw.text((int(tx), int(ty)), text, font=wf, fill=(40, 40, 40, 255))
    canvas.convert("RGB").save(out_path)
    print(f"✅ saved: {out_path} ({canvas_w}x{canvas_h})")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Render survivor-structure figures for a batch")
    ap.add_argument("--batch", default="first_mut", choices=list(BATCHES))
    args = ap.parse_args()
    batch = BATCHES[args.batch]
    data_dir = os.path.join(REPO, "data", batch["data_dir"])
    FIG = os.path.join(data_dir, "figures")
    os.makedirs(FIG, exist_ok=True)
    MUTS = batch["muts"]

    coords = []
    seq = None
    for fn, *_ in MUTS:
        d = np.load(os.path.join(data_dir, fn), allow_pickle=True)
        coords.append(d["coords"].astype(np.float64))
        if seq is None:
            seq = d["seq"] if isinstance(d["seq"], str) else d["seq"].item()
    ref = coords[0]
    aligned = [coords[0]] + [kabsch_to_ref(c, ref) for c in coords[1:]]

    for tag, g in batch["groups"].items():
        idxs = [i for i, m in enumerate(MUTS) if m[4] == tag]
        ss = consensus_ss([aligned[i] for i in idxs], vote=g["vote"])
        nh = ss.count("H"); ne = ss.count("S")
        print(f"[{args.batch}/{tag}] consensus SS: H={nh} ({nh/116:.0%})  E={ne} ({ne/116:.0%})  L={116-nh-ne}")

        out = os.path.join(FIG, g["out"])
        panel_paths = []
        for i in idxs:
            c = aligned[i]
            m = MUTS[i]
            _, mut_pos, name, title, _ = m[:5]
            ov = m[5] if len(m) > 5 else {}
            if not isinstance(ov, dict):
                ov = {"scale": ov}   # 兼容旧的位置标量写法
            ms = ov.get("scale", g["mut_scale"])
            fnt = ov.get("face_nt", g["face_nt"])
            png = os.path.join(FIG, f"{args.batch}_{name}.png")
            pse = os.path.join(FIG, f"{args.batch}_{name}.pse")
            cif = os.path.join(FIG, f"{args.batch}_{name}.cif")
            render_panel(c, mut_pos, ss, png, pse,
                         mut_scale=ms, face_nt=fnt)
            write_mmcif(c, seq, ss, cif, f"{args.batch}_{name}")
            panel_paths.append(png)
            for ext in (png + ".pdb",):
                if os.path.exists(ext):
                    os.remove(ext)  # temp CA pdb used only for rendering
            print(f"  ✅ {name}: png / pse / cif")

        compose(panel_paths, [MUTS[i][3] for i in idxs], out)
    print("done")


if __name__ == "__main__":
    main()

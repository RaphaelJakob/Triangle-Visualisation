"""Bent-edge previews and checks a to e. Run next to main.py: python bend_check.py bottlebrush.jpg nets.jpg rocks.jpg
Writes PNGs and bend_results.json into ./out. Does not change main.py."""
import hashlib, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from main import make_triangle_art, triangle_label_array, triangles_to_image

SEED = 42
STRENGTHS = [0, 0.005, 0.012, 0.025]
SMOOTHNESS = [0.04, 0.08]
MIN_DET = 0.05
KIND = os.environ.get("PHOTO_KIND", "STAND-IN")
OUT = "out"
results = []


def fade_weights(rows, cols, width):
    """1 in the middle, easing smoothly to 0 at the picture's edges over `width` pixels."""
    def ramp(n):
        d = np.minimum(np.arange(n), np.arange(n)[::-1]) / max(width, 1)
        d = np.clip(d, 0, 1)
        return d * d * (3 - 2 * d)            # smoothstep, so the fade has no sharp corner
    return ramp(rows)[:, None] * ramp(cols)[None, :]


def unit_noise(rows, cols, sigma):
    """The same white noise for every strength and smoothness, blurred and scaled to spread 1."""
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(4)[3])
    fields = []
    for _ in range(2):
        white = rng.standard_normal((rows, cols), dtype=np.float32)
        blurred = ndimage.gaussian_filter(white, sigma)
        fields.append(blurred / blurred.std())
    return fields   # [x offset, y offset]


def jacobian_parts(dx, dy):
    dx_dy, dx_dx = np.gradient(dx)   # np.gradient returns d/drow then d/dcol
    dy_dy, dy_dx = np.gradient(dy)
    return dx_dx, dx_dy, dy_dx, dy_dy


def min_det(parts, k):
    a, b, c, d = parts
    return float(((1 + k * a) * (1 + k * d) - (k * b) * (k * c)).min())


def warp_labels_1x(labels, dx, dy):
    rows, cols = labels.shape
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float32)
    bent = ndimage.map_coordinates(labels, [r + dy, c + dx], order=0, mode="reflect")
    return np.asarray(bent).astype(np.int64)   # whole-number triangle IDs, so they can index and be counted


def warp_labels_2x(labels2, dx, dy, strip=512):
    """Warps the 2x label grid. Each 2x pixel j sits at 1x position (j - 0.5) / 2; its offset is
    read from the 1x field with bilinear interpolation and doubled."""
    rows2, cols2 = labels2.shape
    out = np.empty_like(labels2)
    cols_2x = np.arange(cols2, dtype=np.float32)
    for top in range(0, rows2, strip):
        rr = np.arange(top, min(top + strip, rows2), dtype=np.float32)
        R, C = np.meshgrid(rr, cols_2x, indexing="ij")
        p_r, p_c = (R - 0.5) / 2, (C - 0.5) / 2
        off_x = ndimage.map_coordinates(dx, [p_r, p_c], order=1, mode="nearest")
        off_y = ndimage.map_coordinates(dy, [p_r, p_c], order=1, mode="nearest")
        out[top:top + len(rr)] = ndimage.map_coordinates(labels2, [R + 2 * off_y, C + 2 * off_x], order=0, mode="reflect")
    return out


def edges(labels):
    e = np.zeros(labels.shape, bool)
    e[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    e[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return e


for path in sys.argv[1:]:
    name = os.path.splitext(os.path.basename(path))[0]
    t0 = time.perf_counter()
    triangles, colours, straight_image = make_triangle_art(path, SEED)
    cols, rows = straight_image.size
    longer, shorter = max(rows, cols), min(rows, cols)
    labels = triangle_label_array(triangles, (rows, cols))
    labels2 = triangle_label_array(triangles * 2 + 0.5, (rows * 2, cols * 2))
    # The outer ring of 2x sub-pixels lies half a pixel outside the triangles (pixel centres run 0 to
    # width - 1, the picture edge does not), so give each of them the triangle of its nearest neighbour.
    ring_zero = int((labels2 == 0).sum())
    nearest = np.asarray(ndimage.distance_transform_edt(labels2 == 0, return_distances=False, return_indices=True))
    labels2 = labels2[nearest[0], nearest[1]]
    del nearest
    lookup = np.vstack([[0, 0, 0], colours]).astype(np.uint8)
    straight_counts = np.bincount(labels.ravel(), minlength=len(triangles) + 1)
    straight_edges = edges(labels)
    dist_to_straight = np.asarray(ndimage.distance_transform_edt(~straight_edges))
    reference = np.asarray(triangles_to_image(triangles, colours, (rows, cols)))
    crop_size = 600
    cr, cc = rows // 2 - crop_size // 2, cols // 2 - crop_size // 2
    views, crops = {}, {}
    for m in SMOOTHNESS:
        sigma = m * longer
        fade_width = min(2 * sigma, 0.10 * shorter)
        fade = fade_weights(rows, cols, fade_width).astype(np.float32)
        ux, uy = unit_noise(rows, cols, sigma)
        ux *= fade; uy *= fade
        interior = fade >= 0.999
        for s in STRENGTHS:
            px = s * longer
            dx, dy = ux * px, uy * px
            parts = jacobian_parts(dx, dy)
            det = min_det(parts, 1.0)
            k = 1.0
            if det < MIN_DET:      # shrink the bends until nothing folds
                lo, hi = 0.0, 1.0
                for _ in range(30):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if min_det(parts, mid) >= MIN_DET else (lo, mid)
                k = lo
                dx, dy = dx * k, dy * k
            det_used = min_det(jacobian_parts(dx, dy), 1.0)
            bent1 = warp_labels_1x(labels, dx, dy)
            bent2 = warp_labels_2x(labels2, dx, dy)
            picture = np.asarray(Image.fromarray(lookup[bent2]).resize((cols, rows), Image.Resampling.BOX))
            counts = np.bincount(bent1.ravel(), minlength=len(triangles) + 1)
            vanished = int(((straight_counts[1:] > 0) & (counts[1:] == 0)).sum())
            row = dict(photo=name, kind=KIND, size=[cols, rows], smoothness=m, smoothness_px=round(sigma, 1),
                       strength=s, strength_px=round(px, 1), capped=k < 1.0, scale_used=round(k, 4),
                       strength_used=round(s * k, 5), strength_used_px=round(px * k, 1),
                       a_min_det_requested=round(det, 4), a_min_det_used=round(det_used, 4),
                       c_zero_pixels_1x=int((bent1 == 0).sum()), c_zero_pixels_2x=int((bent2 == 0).sum()),
                       e_vanished=vanished, triangles=len(triangles),
                       triangles_empty_before=int((straight_counts[1:] == 0).sum()), ring_zero_2x_before_fill=ring_zero)
            if s == 0:
                plain = lookup[bent1]
                row["b_identical_1x"] = bool(np.array_equal(plain, reference))
                row["b_sha_1x"] = hashlib.sha256(plain.tobytes()).hexdigest()[:16]
                row["b_sha_triangles_to_image"] = hashlib.sha256(reference.tobytes()).hexdigest()[:16]
                row["b_pixels_changed_by_smoothing"] = int((picture != reference).any(axis=2).sum())
            else:
                bent_edges = edges(bent1) & interior
                moved = dist_to_straight[bent_edges]
                med = float(np.median(moved))
                expect = 0.674 * px * k
                row.update(d_ceiling_px=round(float(np.median(dist_to_straight[interior])), 2), d_median_move_px=round(med, 2), d_expected_px=round(expect, 2),
                           d_ratio=round(med / expect, 3), d_consistent=bool(0.5 <= med / expect <= 1.2))
            results.append(row)
            print(json.dumps({k2: v for k2, v in row.items() if k2 not in ("kind",)}), flush=True)
            views[(m, s)] = picture[::4, ::4].copy()
            crops[(m, s)] = picture[cr:cr + crop_size, cc:cc + crop_size].copy()
            del bent1, bent2, picture
    # full-picture overview (shrunk for viewing) and the 100% crop mosaic (no resampling)
    fig, axes = plt.subplots(2, 4, figsize=(16, 2 * 4 * rows / cols + 1))
    for i, m in enumerate(SMOOTHNESS):
        for j, s in enumerate(STRENGTHS):
            r = next(x for x in results if x["photo"] == name and x["smoothness"] == m and x["strength"] == s)
            label = f"{KIND} {name}\nsmooth {m}, strength {s}" + (f" capped to {r['strength_used']}" if r["capped"] else "")
            axes[i, j].imshow(views[(m, s)]); axes[i, j].set_title(label, fontsize=8); axes[i, j].axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}_overview.png", dpi=100); plt.close(fig)
    mosaic = np.full((2 * crop_size + 10, 4 * crop_size + 30, 3), 255, np.uint8)
    for i, m in enumerate(SMOOTHNESS):
        for j, s in enumerate(STRENGTHS):
            y, x = i * (crop_size + 10), j * (crop_size + 10)
            mosaic[y:y + crop_size, x:x + crop_size] = crops[(m, s)]
    Image.fromarray(mosaic).save(f"{OUT}/{name}_crops_100pct.png")
    print(f"{name} done in {time.perf_counter() - t0:.0f} s", flush=True)

json.dump(results, open(f"{OUT}/bend_results.json", "w"), indent=1)

"""Clip-level heatmaps of the uncertainty map on the decoded rollout.

Display rule: the channel-mean map over the predicted latent frames is tinted by its clip-wide percentile rank on a
fixed p50 -> p99 ramp (maps.rank_tint) and painted in orange on a grey copy of the rollout. A black outline marks the
cells above the clip-wide 99th percentile of the 3x3 box-smoothed map, the same statistic as the clip score
u_smooth3_q99. Ranks are taken over the whole clip, so frames can be compared with each other but colours are not
comparable across clips.

Time alignment: the Cosmos3 VAE compresses time by 4. Latent frame 0 is the conditioning frame (pixel frame 0) and
latent frame k >= 1 covers pixel frames 4k - 3 .. 4k.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

from .hier import box_mean
from .maps import rank_tint

ORANGE = np.array([0.90, 0.38, 0.05])
TEMPORAL_RATIO = 4
TOP_Q = 0.99


def read_video(path):
    """Video file -> uint8 array [T, H, W, 3] (imageio with the ffmpeg plugin)."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), format="FFMPEG")
    try:
        return np.stack([np.asarray(f)[..., :3] for f in reader])
    finally:
        reader.close()


def write_video(path, frames, fps):
    import imageio.v2 as imageio

    writer = imageio.get_writer(str(path), format="FFMPEG", fps=fps, codec="libx264", macro_block_size=1)
    try:
        for f in frames:
            writer.append_data(np.asarray(f))
    finally:
        writer.close()


def load_maps(record_dir):
    """Channel-mean U maps (F, h, w) of a record written by `run record`, plus the latent error map if the record
    stores a real-future latent. Returns ({name: map}, active latent frame indices)."""
    z = np.load(Path(record_dir) / "u_divergence.npz")
    maps = {"hier": z["map_hier"], "raw": z["map_raw"]}
    if "latents_gt" in z.files:
        maps["error"] = np.abs(z["latents_final"] - z["latents_gt"]).mean(0)
    return maps, [int(i) for i in z["active_frames"]]


def pixel_to_active(n_pixel, active_frames):
    """For each pixel frame, the index into the active latent frames that covers it, or -1 (conditioning frame)."""
    pos = {f: i for i, f in enumerate(active_frames)}
    out = []
    for t in range(n_pixel):
        k = 0 if t == 0 else (t + TEMPORAL_RATIO - 1) // TEMPORAL_RATIO
        out.append(pos.get(k, -1))
    return out


def clip_layers(m):
    """Clip-wide tint in [0, 1] and outline mask, both (F, h, w)."""
    smooth = box_mean(m, 3)
    return rank_tint(m), smooth >= np.quantile(smooth.ravel(), TOP_Q)


def _resize(a, size, resample):
    """(h, w) array -> (H, W) float array at the frame size."""
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size, resample))


def overlay(frame, tint, mask):
    """frame uint8 (H, W, 3); tint (h, w) in [0, 1]; mask (h, w) bool. Returns uint8 (H, W, 3)."""
    H, W = frame.shape[:2]
    gray = np.repeat(frame.astype(np.float32).mean(-1, keepdims=True) / 255.0, 3, axis=-1)
    t = _resize(tint, (W, H), Image.NEAREST)
    msk = _resize(mask.astype(np.float32), (W, H), Image.NEAREST) > 0.5
    alpha = (0.85 * t)[..., None]
    img = gray * (1 - alpha) + ORANGE[None, None, :] * alpha
    width = max(1, round(min(H, W) / 240))
    img[msk & ~binary_erosion(msk, iterations=width)] = 0.0
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_strip(frames, maps, active_frames, path, columns=None, col_width=320):
    """PNG grid: the rollout in the first row, one overlay row per map. Columns are active latent frames (default all),
    shown at the last pixel frame each latent frame covers."""
    columns = list(range(len(active_frames))) if columns is None else list(columns)
    layers = {name: clip_layers(m) for name, m in maps.items()}
    H, W = frames.shape[1:3]
    w = col_width
    h = round(H * w / W)
    label_w, title_h = 90, 18
    rows = ["rollout"] + list(maps)
    canvas = Image.new("RGB", (label_w + w * len(columns), title_h + h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for j, a in enumerate(columns):
        t = min(len(frames) - 1, active_frames[a] * TEMPORAL_RATIO)
        draw.text((label_w + j * w + 4, 3), f"frame {t}", fill="black")
        for i, name in enumerate(rows):
            if name == "rollout":
                img = frames[t]
            else:
                tint, mask = layers[name]
                img = overlay(frames[t], tint[a], mask[a])
            canvas.paste(Image.fromarray(img).resize((w, h), Image.BILINEAR), (label_w + j * w, title_h + i * h))
    for i, name in enumerate(rows):
        draw.text((4, title_h + i * h + h // 2 - 6), name, fill="black")
    canvas.save(path)
    return path


def render_video(frames, m, active_frames, path, fps=10):
    """MP4 with the rollout on the left and the overlay of map m on the right; the conditioning frame has no map."""
    tint, mask = clip_layers(m)
    idx = pixel_to_active(len(frames), active_frames)
    out = []
    for t, frame in enumerate(frames):
        right = frame if idx[t] < 0 else overlay(frame, tint[idx[t]], mask[idx[t]])
        out.append(np.concatenate([frame, right], axis=1))
    write_video(path, out, fps)
    return path


def load_likelihood_maps(record_dir):
    """Channel-mean per-cell negative log-likelihood maps (F, h, w) of a record written by `run likelihood`, the
    scored video's path and the active latent frame indices."""
    import json

    record_dir = Path(record_dir)
    z = np.load(record_dir / "lik_maps.npz")
    maps = {k[len("nll_map_"):]: z[k] for k in z.files if k.startswith("nll_map_")}
    video = json.loads((record_dir / "lik.json").read_text())["video"]
    return maps, video, [int(i) for i in z["active_frames"]]


def render_record(record_dir, estimator="hier", fps=10, video=True):
    """Write heatmap outputs for a record folder.

    For `run record` outputs: heatmap_strip.png (rollout, U map of the chosen estimator, and latent error when
    available) and heatmap.mp4. For `run likelihood` outputs: nll_strip.png and nll.mp4 with the per-cell negative
    log-likelihood map (high = unlikely under the model).
    """
    record_dir = Path(record_dir)
    paths = []
    if (record_dir / "u_divergence.npz").exists():
        frames = read_video(record_dir / "rollout.mp4")
        maps, active = load_maps(record_dir)
        show = {estimator: maps[estimator]}
        if "error" in maps:
            show["error"] = maps["error"]
        paths.append(render_strip(frames, show, active, record_dir / "heatmap_strip.png"))
        if video:
            paths.append(render_video(frames, maps[estimator], active, record_dir / "heatmap.mp4", fps))
    if (record_dir / "lik_maps.npz").exists():
        maps, src, active = load_likelihood_maps(record_dir)
        frames = read_video(src)
        show = {f"nll {k}": maps[k] for k in (estimator, "ref") if k in maps}
        paths.append(render_strip(frames, show, active, record_dir / "nll_strip.png"))
        if video:
            paths.append(render_video(frames, maps[estimator], active, record_dir / "nll.mp4", fps))
    if not paths:
        raise FileNotFoundError(f"{record_dir}: neither u_divergence.npz nor lik_maps.npz")
    return paths

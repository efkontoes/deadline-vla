"""MP4 -> compact GIF for the README.

    python -m deadlinevla.viz.encode in.mp4 out.gif

GIF encoders dither, which makes every frame differ and defeats compression (the Manim explainer came out at
19 MB). Here every frame is snapped to one shared 64-colour palette without dithering, so flat areas stay
identical between frames. The palette sample includes large swatches of the theme colours, so rare accents
(the orange velocity arrow, the red threat) keep their exact colour. Needs PyAV (installed with manim).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

THEME = ("#0b0b0b", "#52514e", "#2a78d6", "#e34948", "#eb6834", "#008300", "#e4e3df", "#c6dbef", "#6baed6",
         "#2171b5", "#fcfcfb")


def mp4_to_gif(src: str | Path, dst: str | Path, colors: int = 64, frame_ms: int = 100) -> Path:
    import av

    frames = [f.to_image().convert("RGB") for f in av.open(str(src)).decode(video=0)]
    W, H = frames[0].size
    picks = [len(frames) * k // 5 for k in (1, 2, 3, 4)]
    sample = Image.new("RGB", (W, H * (len(picks) + 1)))
    for i, k in enumerate(picks):
        sample.paste(frames[k], (0, i * H))
    draw = ImageDraw.Draw(sample)
    sw = W // len(THEME)
    for i, c in enumerate(THEME):
        draw.rectangle([i * sw, len(picks) * H, (i + 1) * sw, (len(picks) + 1) * H], fill=c)
    pal = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(dst, save_all=True, append_images=q[1:], duration=frame_ms, loop=0)
    return dst


if __name__ == "__main__":
    print(mp4_to_gif(sys.argv[1], sys.argv[2]))

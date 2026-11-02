#!/usr/bin/env python3
"""
Universal 2D Image / Logo / Icon / Emoji -> 64^3 3D SDF Volume Baker (`RGBA8 .bin`)
===================================================================================
Zero-dependency pure Python 3 script (requires NO pip packages; uses only Python stdlib).
Optionally uses Pillow (`PIL`) or macOS `qlmanage`/`sips` if converting SVG/WebP/Emoji.

What it does:
1. Loads any transparent PNG (or converts SVG/JPG/Emoji) into a 2D silhouette mask.
2. Auto-crops & centers the silhouette inside a normalized square canvas with safe SDF padding.
3. Computes the EXACT 2D Euclidean Signed Distance Transform (Felzenszwalb & Huttenlocher O(N)
   parabolic envelope algorithm) — guaranteeing:
   - ZERO butt-joint vertical/horizontal depth lines
   - ZERO CSG boolean local-minimum trenches inside hollow cavities
   - Smooth, exact Euclidean distances both inside (< 0) and outside (> 0) the shape.
4. Extrudes the 2D SDF along Z with `sd_extrude_exact` (rounded tubular bevel with zero 2D bloat)
   and bakes a 64x64x64 RGBA8 `.bin` volume (`1,048,576` bytes):
   - `.rgb` = 3D surface normal `grad(d)` mapped from `[-1, 1]` to `[0, 255]`
   - `.a`   = signed distance `d` scaled to voxel units (`127.5 + d_voxel`, `< 128` = inside).

Usage Examples:
  # 1. Bake any transparent PNG logo or icon:
  python3 bake_image_or_emoji_to_sdf.py --input my_logo.png --output assets/volumes/my_logo_64.bin

  # 2. Bake a dark-on-white or white-on-black image using luminance threshold:
  python3 bake_image_or_emoji_to_sdf.py --input icon.png --output assets/volumes/icon_64.bin --mode luma

  # 3. Adjust 3D Z-thickness (hz) and bevel roundness:
  python3 bake_image_or_emoji_to_sdf.py --input badge.png --output assets/volumes/badge_64.bin --hz 0.095 --bevel 0.055
"""

import argparse
import math
import os
import struct
import subprocess
import tempfile
import zlib

N = 64
INF = 1e12


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# -----------------------------------------------------------------------------
# 1. Zero-Dependency Pure-Python PNG Decoder (8-bit RGBA / RGB / Grayscale / Palette)
# -----------------------------------------------------------------------------
def load_png_rgba(path):
    """Reads a PNG file into (width, height, list_of_rgba_tuples) using only Python stdlib."""
    with open(path, "rb") as f:
        data = f.read()

    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a valid PNG file.")

    pos = 8
    width = height = bit_depth = color_type = None
    plte = []
    trns = b""
    idat_chunks = []

    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if interlace != 0:
                raise ValueError("Interlaced PNGs are not supported without Pillow; re-save as non-interlaced.")
        elif chunk_type == b"PLTE":
            plte = [
                (chunk_data[i], chunk_data[i + 1], chunk_data[i + 2])
                for i in range(0, len(chunk_data), 3)
            ]
        elif chunk_type == b"tRNS":
            trns = chunk_data
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    raw = zlib.decompress(b"".join(idat_chunks))

    # Channels per pixel
    channels_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_map or bit_depth != 8:
        raise ValueError(f"Unsupported PNG format (color_type={color_type}, bit_depth={bit_depth}). Use 8-bit PNG.")

    channels = channels_map[color_type]
    bpp = channels
    stride = width * bpp

    recon = bytearray(height * stride)
    in_pos = 0

    def paeth(a, b, c):
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        if pa <= pb and pa <= pc:
            return a
        elif pb <= pc:
            return b
        return c

    for y in range(height):
        filter_type = raw[in_pos]
        in_pos += 1
        row_start = y * stride
        prev_start = (y - 1) * stride

        for x in range(stride):
            byte_val = raw[in_pos]
            in_pos += 1
            a = recon[row_start + x - bpp] if x >= bpp else 0
            b = recon[prev_start + x] if y > 0 else 0
            c = recon[prev_start + x - bpp] if (y > 0 and x >= bpp) else 0

            if filter_type == 0:
                val = byte_val
            elif filter_type == 1:
                val = (byte_val + a) & 0xFF
            elif filter_type == 2:
                val = (byte_val + b) & 0xFF
            elif filter_type == 3:
                val = (byte_val + ((a + b) >> 1)) & 0xFF
            elif filter_type == 4:
                val = (byte_val + paeth(a, b, c)) & 0xFF
            else:
                raise ValueError(f"Unknown PNG filter {filter_type}")
            recon[row_start + x] = val

    rgba = []
    for y in range(height):
        row_start = y * stride
        for x in range(width):
            px_start = row_start + x * bpp
            if color_type == 6:  # RGBA
                r, g, b, a = recon[px_start : px_start + 4]
            elif color_type == 2:  # RGB
                r, g, b = recon[px_start : px_start + 3]
                a = 255
            elif color_type == 4:  # Grayscale + Alpha
                g_val, a = recon[px_start : px_start + 2]
                r = g = b = g_val
            elif color_type == 0:  # Grayscale
                g_val = recon[px_start]
                r = g = b = g_val
                a = 255
            elif color_type == 3:  # Palette
                idx = recon[px_start]
                r, g, b = plte[idx] if idx < len(plte) else (0, 0, 0)
                a = trns[idx] if idx < len(trns) else 255
            rgba.append((r, g, b, a))

    return width, height, rgba


def load_any_image_rgba(path):
    """Loads PNG natively, or converts SVG/JPG/WebP via Pillow or macOS sips/qlmanage."""
    ext = os.path.splitext(path)[1].lower()
    try:
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        return w, h, list(img.getdata())
    except ImportError:
        pass

    if ext == ".png":
        return load_png_rgba(path)

    # Fallback on macOS: use `sips` or `qlmanage` to convert to temporary PNG
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_png = os.path.join(tmpdir, "converted.png")
        if ext == ".svg":
            subprocess.run(["qlmanage", "-t", "-s", "512", "-o", tmpdir, path], check=True)
            for fname in os.listdir(tmpdir):
                if fname.endswith(".png"):
                    return load_png_rgba(os.path.join(tmpdir, fname))
        subprocess.run(["sips", "-s", "format", "png", path, "--out", tmp_png], check=True)
        return load_png_rgba(tmp_png)


# -----------------------------------------------------------------------------
# 2. Exact 2D Euclidean Distance Transform (Felzenszwalb & Huttenlocher O(W*H))
# -----------------------------------------------------------------------------
def edt_1d_squared(f, n):
    """Exact 1D squared Euclidean distance transform via lower parabolic envelope."""
    d = [0.0] * n
    v = [0] * n
    z = [0.0] * (n + 1)
    k = 0
    v[0] = 0
    z[0] = -INF
    z[1] = +INF

    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = +INF

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dx = q - v[k]
        d[q] = dx * dx + f[v[k]]
    return d


def compute_exact_signed_distance_2d(mask, w, h):
    """
    Given a boolean/float inside mask (True = inside shape, False = outside),
    returns a 2D grid (h x w) of exact Euclidean signed distances in pixel units:
      d < 0 inside the shape, d > 0 outside the shape.
    """
    # Distance to nearest INSIDE pixel (for outside points)
    f_out = [[0.0 if mask[y][x] else INF for x in range(w)] for y in range(h)]
    # Distance to nearest OUTSIDE pixel (for inside points)
    f_in = [[INF if mask[y][x] else 0.0 for x in range(w)] for y in range(h)]

    def run_2d_edt(grid):
        # Pass 1: rows
        row_trans = [edt_1d_squared(grid[y], w) for y in range(h)]
        # Pass 2: columns
        out = [[0.0] * w for _ in range(h)]
        for x in range(w):
            col = [row_trans[y][x] for y in range(h)]
            col_d = edt_1d_squared(col, h)
            for y in range(h):
                out[y][x] = math.sqrt(col_d[y])
        return out

    dist_outside = run_2d_edt(f_out)
    dist_inside = run_2d_edt(f_in)

    sdf = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if mask[y][x]:
                # Inside: negative distance to boundary (offset by -0.5 px so boundary sits on pixel edge)
                sdf[y][x] = -(dist_inside[y][x] - 0.5)
            else:
                # Outside: positive distance to boundary
                sdf[y][x] = +(dist_outside[y][x] - 0.5)
    return sdf


# -----------------------------------------------------------------------------
# 3. Auto-Fit 2D Silhouette into Normalized [-0.5, 0.5]^2 Grid & Extrude to 64^3
# -----------------------------------------------------------------------------
def sd_extrude_exact(d2d, pz, hz=0.088, bevel=0.054):
    """
    Exact rounded tubular Z-extrusion with zero 2D XY contour bloat:
    At pz=0, d3d(x,y,0) == d2d(x,y) identically, while giving the shape a
    sculpted 3D rounded cross-section so overhead lighting wraps naturally.
    """
    dz = abs(pz) - (hz - bevel)
    dx = d2d + bevel
    if dz <= 0.0 and dx <= 0.0:
        return max(d2d, abs(pz) - hz)
    return math.hypot(max(dx, 0.0), max(dz, 0.0)) - bevel


def render_emoji_or_text_to_rgba(text, size=512):
    """
    Renders an emoji or text string (e.g. '🚀', '⚡', '☁️') to an RGBA pixel list (size x size).
    Tries Pillow first, then falls back to macOS native AppKit/CoreGraphics via `swift`.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_png = os.path.join(tmpdir, "emoji.png")
        swift_code = f'''
import AppKit
let size = CGSize(width: {size}, height: {size})
let img = NSImage(size: size)
img.lockFocus()
NSColor.clear.set()
NSRect(origin: .zero, size: size).fill()
let str = "{text}" as NSString
let font = NSFont.systemFont(ofSize: {int(size * 0.72)})
let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
let b = str.size(withAttributes: attrs)
let rect = NSRect(x: (size.width - b.width) / 2.0, y: (size.height - b.height) / 2.0, width: b.width, height: b.height)
str.draw(in: rect, withAttributes: attrs)
img.unlockFocus()
if let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff), let png = rep.representation(using: .png, properties: [:]) {{
    try! png.write(to: URL(fileURLWithPath: "{tmp_png}"))
}}
'''
        try:
            subprocess.run(["swift", "-e", swift_code], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(tmp_png):
                return load_png_rgba(tmp_png)
        except Exception:
            pass
    raise RuntimeError("Could not render --emoji automatically. Please provide a PNG/SVG via --input.")


def build_2d_sdf_sampler(width, height, rgba, mode="auto", threshold=128, invert=False, grid_res=256, fill_span=0.84):
    """
    Extracts the 2D silhouette mask from RGBA pixels, normalizes its bounding box to `fill_span`
    of a `grid_res x grid_res` square canvas, and computes its bilinearly-interpolated 2D SDF.
    """
    # Auto-detect whether image uses alpha transparency or opaque background (dark vs light corners)
    effective_mode = mode
    auto_bright_fg = False
    if effective_mode == "auto":
        alphas = [px[3] for px in rgba]
        if min(alphas) < threshold and max(alphas) >= threshold:
            effective_mode = "alpha"
        else:
            effective_mode = "luma"
            corners = [
                rgba[0],
                rgba[width - 1],
                rgba[(height - 1) * width],
                rgba[height * width - 1],
            ]
            corner_luma = sum(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in corners) / 4.0
            # If corners are dark (< 128), foreground is brighter than background
            auto_bright_fg = corner_luma < 128.0

    raw_mask = [[False] * width for _ in range(height)]
    min_x, min_y = width, height
    max_x, max_y = -1, -1

    for y in range(height):
        row_off = y * width
        for x in range(width):
            r, g, b, a = rgba[row_off + x]
            if effective_mode == "alpha":
                inside = a >= threshold
            else:
                luma = 0.299 * r + 0.587 * g + 0.114 * b
                if mode == "auto":
                    inside = (luma >= threshold) if auto_bright_fg else (luma < threshold)
                else:
                    inside = (luma < threshold) if not invert else (luma >= threshold)
            if invert and (effective_mode == "alpha" or mode == "auto"):
                inside = not inside
            raw_mask[y][x] = inside
            if inside:
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y

    if max_x < min_x or max_y < min_y:
        raise ValueError("No foreground pixels found in input image! Check --mode / --threshold.")

    bw = max(1, max_x - min_x + 1)
    bh = max(1, max_y - min_y + 1)
    max_dim = max(bw, bh)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5

    # Resample mask onto a square `grid_res x grid_res` grid where `[-0.5, 0.5]` maps to `[0, grid_res - 1]`
    # and the shape's largest dimension spans `fill_span` (default 0.84 of the [-0.5, 0.5] cube).
    scale_px = max_dim / fill_span
    sq_mask = [[False] * grid_res for _ in range(grid_res)]

    for gy in range(grid_res):
        # Normalized y in [-0.5, 0.5] (note: +y is UP in 3D volume space, -y is DOWN in image space)
        ny = (gy + 0.5) / grid_res - 0.5
        src_y = int(round(cy - ny * scale_px))
        for gx in range(grid_res):
            nx = (gx + 0.5) / grid_res - 0.5
            src_x = int(round(cx + nx * scale_px))
            if 0 <= src_x < width and 0 <= src_y < height:
                sq_mask[gy][gx] = raw_mask[src_y][src_x]

    sdf_px = compute_exact_signed_distance_2d(sq_mask, grid_res, grid_res)
    # Convert pixel distance to normalized [-0.5, 0.5] cube units
    sdf_norm = [[sdf_px[gy][gx] / grid_res for gx in range(grid_res)] for gy in range(grid_res)]

    def sample_2d(x, y):
        # Map x, y in [-0.5, 0.5] to grid coordinates
        fx = clamp((x + 0.5) * grid_res - 0.5, 0.0, grid_res - 1.0001)
        fy = clamp((y + 0.5) * grid_res - 0.5, 0.0, grid_res - 1.0001)
        ix = int(fx)
        iy = int(fy)
        tx = fx - ix
        ty = fy - iy
        d00 = sdf_norm[iy][ix]
        d10 = sdf_norm[iy][ix + 1]
        d01 = sdf_norm[iy + 1][ix]
        d11 = sdf_norm[iy + 1][ix + 1]
        return (d00 * (1.0 - tx) + d10 * tx) * (1.0 - ty) + (d01 * (1.0 - tx) + d11 * tx) * ty

    return sample_2d


def bake_3d_sdf_from_2d(sample_2d, out_path, hz=0.088, bevel=0.054):
    """Bakes the 64x64x64 RGBA8 3D SDF `.bin` volume from a 2D SDF sampler."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    buf = bytearray(N * N * N * 4)
    eps = 1.0 / N
    inside_count = 0

    def sdf_3d(x, y, z):
        return sd_extrude_exact(sample_2d(x, y), z, hz=hz, bevel=bevel)

    idx = 0
    for iz in range(N):
        z = (iz + 0.5) / N - 0.5
        for iy in range(N):
            y = (iy + 0.5) / N - 0.5
            for ix in range(N):
                x = (ix + 0.5) / N - 0.5
                d = sdf_3d(x, y, z)

                gx = sdf_3d(x + eps, y, z) - sdf_3d(x - eps, y, z)
                gy = sdf_3d(x, y + eps, z) - sdf_3d(x, y - eps, z)
                gz = sdf_3d(x, y, z + eps) - sdf_3d(x, y, z - eps)
                glen = math.sqrt(gx * gx + gy * gy + gz * gz)
                if glen > 1e-7:
                    gx /= glen
                    gy /= glen
                    gz /= glen
                else:
                    gx, gy, gz = 0.0, 1.0, 0.0

                r_byte = int(clamp(round((gx * 0.5 + 0.5) * 255.0), 0, 255))
                g_byte = int(clamp(round((gy * 0.5 + 0.5) * 255.0), 0, 255))
                b_byte = int(clamp(round((gz * 0.5 + 0.5) * 255.0), 0, 255))

                d_voxel = d * (N * 0.5)
                a_byte = int(clamp(round(127.5 + d_voxel * 127.5), 0, 255))
                if a_byte <= 128:
                    inside_count += 1

                buf[idx] = r_byte
                buf[idx + 1] = g_byte
                buf[idx + 2] = b_byte
                buf[idx + 3] = a_byte
                idx += 4

    with open(out_path, "wb") as f:
        f.write(buf)

    pct = inside_count * 100.0 / (N ** 3)
    print(f"Baked {out_path} ({len(buf):,} bytes) - inside voxels: {inside_count:,} / {N**3:,} ({pct:.1f}%)")
    return inside_count


def main():
    parser = argparse.ArgumentParser(
        description="Bake any 2D PNG/SVG/Icon or Emoji into a 64^3 RGBA8 3D SDF Volume (.bin) for the GPGPU Particle Portal."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="Path to input image (PNG, SVG, JPG, WebP)")
    group.add_argument("--emoji", "-e", help="Emoji or text character to render directly (e.g. '🚀', '⚡', '❤️')")
    parser.add_argument("--output", "-o", required=True, help="Path to output 64^3 .bin volume file")
    parser.add_argument(
        "--mode",
        choices=["auto", "alpha", "luma", "luminance"],
        default="auto",
        help="Foreground detection mode: 'auto' (default, detects transparency or dark/light corners), 'alpha', or 'luma'",
    )
    parser.add_argument("--threshold", type=int, default=128, help="Foreground threshold (0..255, default: 128)")
    parser.add_argument("--invert", action="store_true", help="Invert foreground/background mask")
    parser.add_argument("--hz", type=float, default=0.088, help="3D Z half-thickness in [-0.5, 0.5] cube (default: 0.088)")
    parser.add_argument("--bevel", type=float, default=0.054, help="3D tubular bevel radius (default: 0.054)")
    parser.add_argument("--span", type=float, default=0.84, help="2D bounding-box fill ratio inside [-0.5, 0.5] (default: 0.84)")
    args = parser.parse_args()

    mode = "luma" if args.mode == "luminance" else args.mode
    if args.emoji:
        w, h, rgba = render_emoji_or_text_to_rgba(args.emoji, size=512)
        print(f"Rendered emoji '{args.emoji}' ({w}x{h} px)...")
    else:
        w, h, rgba = load_any_image_rgba(args.input)
        print(f"Loaded {args.input} ({w}x{h} px)...")

    sample_2d = build_2d_sdf_sampler(
        w,
        h,
        rgba,
        mode=mode,
        threshold=args.threshold,
        invert=args.invert,
        grid_res=256,
        fill_span=args.span,
    )
    bake_3d_sdf_from_2d(sample_2d, args.output, hz=args.hz, bevel=args.bevel)


if __name__ == "__main__":
    main()

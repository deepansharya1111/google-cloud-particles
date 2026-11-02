#!/usr/bin/env python3
"""
Generates 64x64x64 RGBA8 3D Signed Distance Field (SDF) + Surface Gradient Volume files (.bin)
matching the 3D volumetric particle data contract:
  - RGB (channels 0..2): Normalized 3D SDF gradient (grad * 0.5 + 0.5) * 255
  - A   (channel 3):     Signed distance encoded as clamp(127.5 + d_norm * 64.0, 0, 255)
                         (so in shader: dist = (volData.a * 2.0 - 1.0) * 2.0)
"""
import math
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "volumes")
os.makedirs(OUT_DIR, exist_ok=True)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def sd_circle(px, py, cx, cy, r):
    return math.hypot(px - cx, py - cy) - r


def sd_box2d(px, py, cx, cy, bx, by, r=0.0):
    dx = abs(px - cx) - bx + r
    dy = abs(py - cy) - by + r
    return math.hypot(max(dx, 0.0), max(dy, 0.0)) + min(max(dx, dy), 0.0) - r


def smin(a, b, k=0.03):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


def smax(a, b, k=0.02):
    return -smin(-a, -b, k)


def sd_extrude_capsule(d2d, pz, hz, bevel=0.025):
    """Extrudes a 2D SDF along Z with half-thickness hz and rounded bevel."""
    dx = d2d + bevel * 0.3
    dz = abs(pz) - hz + bevel
    return math.hypot(max(dx, 0.0), max(dz, 0.0)) + min(max(dx, dz), 0.0) - bevel


def sd_extrude_exact(d2d, pz, hz=0.088, bevel=0.054):
    """
    Exact rounded tubular Z-extrusion with zero 2D XY contour bloat:
    At pz=0, d3d(x,y,0) == d2d(x,y) identically, while giving the ribbon a
    sculpted 3D rounded cross-section so overhead studio lighting wraps naturally.
    """
    dz = abs(pz) - (hz - bevel)
    dx = d2d + bevel
    if dz <= 0.0 and dx <= 0.0:
        return max(d2d, abs(pz) - hz)
    return math.hypot(max(dx, 0.0), max(dz, 0.0)) - bevel


def sd_segment(px, py, ax, ay, bx, by):
    pax, pay = px - ax, py - ay
    bax, bay = bx - ax, by - ay
    h = clamp((pax * bax + pay * bay) / (bax * bax + bay * bay), 0.0, 1.0)
    return math.hypot(pax - bax * h, pay - bay * h)


def sdf_google_cloud_ring(x, y, z):
    """
    Official Google Cloud Logo 3D SDF:
    Exact Euclidean 2D ribbon SDF (zero internal seam ridges/grooves, zero mid-air distance sinkholes,
    and zero stray cavity islands) with the exact inner C-ring extension, extruded as a rounded 3D tubular sculpture.
    """
    S = 257.0 / 0.86
    x_px = 173.5 + x * S
    y_px = 171.0 - y * S

    rL = math.hypot(x_px - 117.5, y_px - 201.5)
    rT = math.hypot(x_px - 173.5, y_px - 172.25)
    rR = math.hypot(x_px - 229.5, y_px - 201.5)
    d_diag = (x_px + y_px - 333.0) * 0.70710678

    # 1. Lower U-Stadium (bottom half of Left C-ring + horizontal Bottom Bar + bottom half of Right Lobe)
    # Uses a single C1-continuous clamped stadium axis cx = clamp(x_px, 117.5, 229.5) so there are zero
    # vertical butt-joint seams or depth lines at x_px = 117.5 or x_px = 229.5.
    if y_px >= 201.5:
        cx = clamp(x_px, 117.5, 229.5)
        r_low = math.hypot(x_px - cx, y_px - 201.5)
        d_lower = abs(r_low - 50.0) - 22.5
    else:
        d_lower = 1e6

    # 2. Upper Left C-ring + 45-deg inner hook:
    if y_px <= 201.5 and d_diag <= 0.0:
        d_rad = abs(rL - 50.0) - 22.5
        if x_px > 117.5:
            if d_rad <= 0.0 and d_diag <= 0.0:
                d_left = max(d_rad, d_diag)
            else:
                d_left = math.hypot(max(d_rad, 0.0), max(d_diag, 0.0))
        else:
            d_left = d_rad
    else:
        d_left = sd_segment(x_px, y_px, 142.6418, 190.3582, 175.2851, 157.7149)
        if y_px >= 201.5 and x_px <= 117.5:
            d_left = min(d_left, abs(rL - 50.0) - 22.5)

    # 3. Top Arch (y_px <= 172.25)
    if y_px <= 172.25:
        d_top = abs(rT - 81.75) - 22.5
    else:
        d_top_L = sd_segment(x_px, y_px, 173.5 - 104.25, 172.25, 173.5 - 59.25, 172.25)
        d_top_R = sd_segment(x_px, y_px, 173.5 + 59.25, 172.25, 173.5 + 104.25, 172.25)
        d_top = min(d_top_L, d_top_R)

    # 4. Upper Right Lobe: right half of C_R (x_px >= 229.5) and outside Top Arch inner cavity (rT >= 59.25)
    if x_px >= 229.5 and (y_px >= 172.25 or rT >= 59.25):
        d_right = abs(rR - 50.0) - 22.5
    else:
        d_right = sd_segment(x_px, y_px, 232.75, 172.25, 274.2, 144.8)

    d_cloud_2d = min(d_lower, d_left, d_top, d_right) / S

    # Uniform rounded tubular Z-extrusion across the entire ribbon
    return sd_extrude_exact(d_cloud_2d, z, hz=0.088, bevel=0.054)


def sdf_google_cloud_solid(x, y, z):
    """
    Full Volumetric 3D Cumulus Cloud (Puffy 3D Cloud Sculpture):
    Union of 3D spheres and rounded base so particles swirl through a deep 3D cloud volume.
    """
    px = x * 0.94
    py = y * 0.94 - 0.015
    pz = z * 1.15

    def sd_sphere(cx, cy, cz, r):
        return math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2) - r

    s_top = sd_sphere(-0.02, 0.085, 0.0, 0.25)
    s_left = sd_sphere(-0.22, -0.068, 0.01, 0.18)
    s_right = sd_sphere(0.22, -0.048, -0.01, 0.20)
    s_front = sd_sphere(0.02, -0.02, 0.055, 0.175)
    s_back = sd_sphere(-0.03, -0.01, -0.055, 0.175)

    d_base2d = sd_box2d(px, py, 0.0, -0.105, 0.22, 0.135, 0.04)
    d_base = sd_extrude_capsule(d_base2d, pz, hz=0.12, bevel=0.045)

    d = smin(s_top, s_left, 0.04)
    d = smin(d, s_right, 0.04)
    d = smin(d, s_front, 0.04)
    d = smin(d, s_back, 0.04)
    d = smin(d, d_base, 0.035)
    return d


def sdf_gemini_spark(x, y, z):
    """
    Google Gemini / Vertex AI 4-Point Astroid Sparkle in 3D:
    Concave 4-point star with a 3D crystalline bevel profile.
    """
    ax = abs(x) * 0.92
    ay = abs(y) * 0.92
    az = abs(z)
    r_star = 0.44
    d_2d = (ax ** 0.60 + ay ** 0.60) ** (1.0 / 0.60) - r_star
    taper_hz = max(0.045, 0.14 * (1.0 - 1.25 * math.hypot(ax, ay)))
    return sd_extrude_capsule(d_2d * 0.62, az, hz=taper_hz, bevel=0.03)


def bake_volume(name, sdf_fn, dim=64):
    buf = bytearray(dim * dim * dim * 4)
    inside_count = 0

    pad = dim + 2
    field = [0.0] * (pad * pad * pad)
    for kz in range(pad):
        z = ((kz - 1) / (dim - 1.0)) - 0.5
        for ky in range(pad):
            y = ((ky - 1) / (dim - 1.0)) - 0.5
            row_offset = (kz * pad + ky) * pad
            for kx in range(pad):
                x = ((kx - 1) / (dim - 1.0)) - 0.5
                field[row_offset + kx] = sdf_fn(x, y, z)

    idx = 0
    for z in range(dim):
        kz = z + 1
        for y in range(dim):
            ky = y + 1
            for x in range(dim):
                kx = x + 1
                c_idx = (kz * pad + ky) * pad + kx
                d = field[c_idx]

                gx = field[c_idx + 1] - field[c_idx - 1]
                gy = field[c_idx + pad] - field[c_idx - pad]
                gz = field[c_idx + pad * pad] - field[c_idx - pad * pad]
                length = math.sqrt(gx * gx + gy * gy + gz * gz)
                if length > 1e-8:
                    gx /= length
                    gy /= length
                    gz /= length
                else:
                    gx, gy, gz = 0.0, 1.0, 0.0

                r = int(round(clamp((gx * 0.5 + 0.5) * 255.0, 0.0, 255.0)))
                g = int(round(clamp((gy * 0.5 + 0.5) * 255.0, 0.0, 255.0)))
                b = int(round(clamp((gz * 0.5 + 0.5) * 255.0, 0.0, 255.0)))
                a = int(round(clamp(127.5 + d * 64.0, 0.0, 255.0)))
                if a <= 127:
                    inside_count += 1

                buf[idx] = r
                buf[idx + 1] = g
                buf[idx + 2] = b
                buf[idx + 3] = a
                idx += 4

    out_path = os.path.join(OUT_DIR, f"{name}.bin")
    with open(out_path, "wb") as f:
        f.write(buf)
    total = dim * dim * dim
    print(f"Baked {out_path} ({len(buf)} bytes) - inside voxels: {inside_count}/{total} ({inside_count/total*100:.1f}%)")


if __name__ == "__main__":
    bake_volume("google_cloud_64", sdf_google_cloud_ring, 64)
    bake_volume("google_cloud_solid_64", sdf_google_cloud_solid, 64)
    bake_volume("gemini_spark_64", sdf_gemini_spark, 64)

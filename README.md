# WebGL2 3D SDF Volume & Navier-Stokes Fluid Particle Engine

An interactive, 60fps WebGL2 GPGPU particle system (`150,000+` volumetric spheres) driven by a real-time 2D Navier-Stokes fluid solver and 3D Signed Distance Field (`.bin`) volumes.

Built from first principles as a **universal, zero-build-step open-source template** that can morph between **any 3D logo, icon, SVG, PNG silhouette, or emoji**.

---

## ✨ Key Features

- **150,400 GPGPU Volumetric Particles**: Ping-pong `RGBA32F` floating-point position & velocity simulation with zero CPU-to-GPU per-frame overhead.
- **5-Force Volumetric Attractor Physics**:
  1. **2D Navier-Stokes Fluid Velocity Injection** (`fluid-sim.js` — vorticity confinement + 14-iteration Jacobi pressure solver).
  2. **3D Simplex Curl-Noise Turbulence** for organic internal swirling.
  3. **Rotated Home-Voxel Spring Tether (`toOrig`)** with automatic hollow-cavity boost (`outsideBoost`) so hollow rings never trap stray particles.
  4. **3D SDF Gradient Attraction (`-grad`)** with dot-product `cavityGuard` so surface normal forces never fight home-tether forces across hollow openings.
  5. **Soft Shell Boundary Confinement** (`smoothstep(-0.015, 0.025, dist)`).
- **1-Command Image / SVG / Emoji to 3D SDF Baker (`bake_image_or_emoji_to_sdf.py`)**:
  - **100% Zero-Dependency Python 3 CLI** — convert any `.png`, `.svg`, icon, or emoji (`--emoji "🚀"`) into a `64×64×64` RGBA8 3D SDF volume (`1,048,576` bytes) in ~2 seconds using exact Felzenszwalb & Huttenlocher $O(W \times H)$ 2D Euclidean Distance Transform + rounded tubular 3D Z-extrusion (`sd_extrude_exact`).
- **Interactive Physics Deck + 360° Free Inspection Mode**:
  - Tune particle count (`35K`–`150K`), sphere radius, spring return, SDF gradient pull, curl turbulence, fluid splat force, and color themes live.
  - **Save State / Revert Saved / Reset Default** via `localStorage`.
  - **360° Free Orbit Mode** (`R` key or Left HUD toggle) locks out fluid/mouse perturbation and lets you freely drag-orbit $360^\circ$ in yaw & pitch plus scroll-zoom to inspect the 3D particle sculpture from any angle.
- **100% Offline Ready**: Bundled with Three.js (`vendor/three.module.js`), fonts (`fonts/`), and pre-baked `.bin` volumes (`assets/volumes/`).

---

## 🚀 Quick Start (Zero Build Step)

```bash
# Clone or open the directory and serve over any static HTTP server:
python3 -m http.server 8302
```

Then open `http://localhost:8302/` in Chrome, Edge, Safari, or Firefox.

---

## 🛠️ Create Your Own Shape, Logo, Icon, or Emoji in 1 Command

You do **not** need Blender, Houdini, or external Python libraries to create a new 3D particle volume. Use the included zero-dependency baker [`bake_image_or_emoji_to_sdf.py`](./bake_image_or_emoji_to_sdf.py):

### A. Bake from any PNG / SVG Logo or Icon
```bash
python3 bake_image_or_emoji_to_sdf.py \
  --input path/to/your_logo.png \
  --output assets/volumes/my_logo_64.bin
```
*(Note: `--mode auto` automatically detects whether your image uses alpha transparency or an opaque dark/light background.)*

### B. Bake directly from any Emoji or Character (macOS)
```bash
python3 bake_image_or_emoji_to_sdf.py \
  --emoji "🚀" \
  --output assets/volumes/rocket_64.bin
```

### C. Analytical CSG / Mathematical 3D Volumes
For multi-depth 3D sculptures (such as the 6-puff 3D Cumulus Cloud or exact $C^1$-continuous parametric logos), see [`generate_volumes.py`](./generate_volumes.py).

---

## 🤖 Using an AI Coding Assistant? (Read This First)

If you are using an AI assistant (Cursor, Claude, GPT, Gemini, Windsurf, or smaller local models) to add your own logo/icon or embed this particle engine into your existing website, point the AI directly to:

👉 **[`AI_AGENT_INTEGRATION_GUIDE.md`](./AI_AGENT_INTEGRATION_GUIDE.md)**

It contains **copy-paste code snippets**, **exact file line locations**, the **8 Golden Rules of 3D SDF Particle Volumes**, and a **Zero-to-One Blueprint for recreating the entire engine from scratch** so even smaller AI models can integrate or customize shapes on the first try without artifacts.

---

## 🧱 Tech Stack & Architecture

Built with a **Zero-Dependency Static WebGL2 Stack** (no Next.js, no React, no Node.js server, no Rust/WASM, and no Webpack/Vite bundler required):

1. **Browser Runtime (Live on the Website)**:
   - **Vanilla HTML5 + CSS3 + Native ES6 Modules** (`index.html`, `css/styles.css`, `js/main.js`) using browser-native `<script type="importmap">`.
   - **Three.js (Local ES Module)** (`vendor/three.module.js` & `vendor/three.core.js`) — lightweight scene graph, camera, and WebGL2 FBO wrapper.
   - **Raw WebGL2 + GLSL ES 3.00 Shaders** (`js/particle-portal.js` & `js/fluid-sim.js`) — all physics and rendering execute 100% on the GPU:
     - **WebGL2 MRT (Multiple Render Targets) GPGPU Ping-Pong** simulating `150,544` particles (`388×388` `RGBA32F` float textures + `64×64×64` hardware 3D texture `sampler3D`).
     - **2D Navier-Stokes Stable Fluids Solver** (`128×128` FBO ping-pong, 6 shader passes per frame) for cursor fluid interaction.
2. **Offline Volume Baker CLI (Only used locally when generating new `.bin` shapes)**:
   - **Pure Standard-Library Python 3** (`bake_image_or_emoji_to_sdf.py` & `generate_volumes.py`) — requires zero `pip` packages and is **not** needed at runtime or on your web host once `.bin` files are placed in `assets/volumes/`.

---

## 📦 Deployment Size & GitHub Pages Compatibility

### 100% Compatible with GitHub Pages (Zero Build Step)
Every script, stylesheet, font, and `.bin` volume uses **relative paths** (`./vendor/three.module.js`, `./assets/volumes/google_cloud_64.bin`, `./css/styles.css`). You can deploy directly to **GitHub Pages** (`https://<username>.github.io/<repo-name>/`), Vercel, Netlify, Cloudflare Pages, or any static S3/GCS bucket straight from the `main` branch with **zero configuration and zero build commands**.

### Disk & Network Transfer Size Breakdown
Total repository footprint is **~5.3 MB uncompressed on disk**, and **~1.1 MB over the network on initial page load** when served with standard Gzip/Brotli compression:

| Component | Uncompressed (Disk) | Gzipped (Network Transfer) | Notes |
| :--- | :--- | :--- | :--- |
| **HTML + CSS + Custom JS/GLSL Engine** (`index.html`, `css/`, `js/`) | **104 KB** | **~24 KB** | UI, 2D Navier-Stokes solver, and WebGL2 GPGPU shaders |
| **3D SDF Volume Assets** (`3 × .bin` in `assets/volumes/`) | **3.0 MB** (`1.0 MB` each) | **~583 KB – 776 KB** each | Each `64×64×64` RGBA8 volume is `1,048,576` bytes; secondary volumes load on demand |
| **Bundled Three.js Library** (`vendor/`) | **1.9 MB** | **~385 KB** | Self-hosted offline ES modules (`three.module.js` + `three.core.js`) |
| **Local Webfonts** (`fonts/*.woff2`) | **312 KB** | **~312 KB** | Pre-compressed `.woff2` font binaries |
| **Total Footprint** | **~5.3 MB** | **~1.1 MB initial load** | Runs 100% offline or on any static host |

---

## 📁 Repository Structure

```text
google-cloud-particles/
├── index.html                        # Interactive studio HUD + <canvas id="webgl-canvas">
├── css/
│   └── styles.css                    # Atmospheric silver-slate theme & frosted-glass HUD controls
├── js/
│   ├── main.js                       # UI controller, 360° Free Rotate lock & State Snapshot manager
│   ├── particle-portal.js            # WebGL2 3D SDF Volume MRT GPGPU Particle Engine + GLSL3 Shaders
│   └── fluid-sim.js                  # 2D Navier-Stokes WebGL FBO Fluid Solver (128x128)
├── bake_image_or_emoji_to_sdf.py     # Zero-dependency CLI: Any PNG/SVG/Emoji -> 64^3 3D SDF .bin
├── generate_volumes.py               # Analytical 3D SDF generator (Google Cloud Logo, Cumulus 3D, Gemini Spark)
├── AI_AGENT_INTEGRATION_GUIDE.md     # Step-by-step recipe, 8 Golden Rules & Zero-to-One from-scratch blueprint
├── assets/
│   └── volumes/                      # Pre-baked 64x64x64 RGBA8 3D SDF volumes (1,048,576 bytes each)
│       ├── google_cloud_64.bin
│       ├── google_cloud_solid_64.bin
│       └── gemini_spark_64.bin
├── fonts/                            # Self-hosted IBM Plex Sans, IBM Plex Mono & Geist Pixel (.woff2)
└── vendor/                           # Self-hosted Three.js ES modules (three.module.js + three.core.js)
```

---

## 📄 License

MIT License — free for personal and commercial use.

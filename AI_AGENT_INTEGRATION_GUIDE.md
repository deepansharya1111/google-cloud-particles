# AI Agent & Developer Integration Guide (`google-cloud-particles`)

> **INSTRUCTION FOR AI CODING MODELS**:
> Read this entire file before editing any code. This repository is pre-configured so you do **not** need to invent GPGPU shaders, fluid solvers, or 3D SDF math from scratch. Follow the exact recipes below.

---

## Recipe 1: How to Add Any New Logo, Icon, PNG, SVG, or Emoji (3 Steps)

Whenever the user asks to **"add a new shape / logo / icon / emoji"** to this particle system, follow these exact 3 steps:

### Step 1: Bake the `64×64×64` `.bin` Volume Using `bake_image_or_emoji_to_sdf.py`

Do **not** write complex analytical CSG math unless the user asks for a custom multi-depth 3D sculpture. Instead, run the zero-dependency CLI baker:

- **From an Image / Logo / Icon (`.png`, `.svg`, `.jpg`)**:
  ```bash
  python3 bake_image_or_emoji_to_sdf.py \
    --input path/to/logo.png \
    --output assets/volumes/custom_shape_64.bin
  ```
- **From an Emoji or Text Character (`🚀`, `⚡`, `❤️`, `🔥`)**:
  ```bash
  python3 bake_image_or_emoji_to_sdf.py \
    --emoji "🚀" \
    --output assets/volumes/custom_shape_64.bin
  ```

**Verification Check**: The command must print `Baked ... (1,048,576 bytes)` and the inside voxel percentage should be between `3.0%` and `18.0%`.
- If inside voxels are `> 50%` (background was detected instead of foreground), re-run with `--invert`.
- To make the 3D volume thicker or thinner in Z, adjust `--hz 0.088` (default `0.088`) and `--bevel 0.054` (default `0.054`).

---

### Step 2: Register the Volume in `js/particle-portal.js` and `index.html`

1. Open [`js/particle-portal.js`](./js/particle-portal.js) and add an entry to `VOLUME_CATALOG` (around line 14):
   ```javascript
   {
     id: 'custom_shape',
     index: '04',
     name: 'CUSTOM LOGO 3D',
     sub: '64³ Custom SDF Volume · 150K Particles',
     file: './assets/volumes/custom_shape_64.bin',
     shapeType: 3, // Passed to uShapeType in the fragment shader for custom coloring
     volumeScale: 0.90,
     defaultTheme: 0,
     description: 'Custom 3D Signed Distance Field volume with 150,400 interactive particles.',
   }
   ```
2. Open [`index.html`](./index.html) and add a matching `<button class="shape-card" data-shape="3">` inside `#shape-switcher`.

---

### Step 3: Customize the Particle Colors in `FRAGMENT_SHADER` (`js/particle-portal.js`)

In [`js/particle-portal.js`](./js/particle-portal.js), open `FRAGMENT_SHADER` (`getBrandRegionColor(vec3 localPos, int theme, int shapeType)`):

- `localPos` is the particle's **unrotated 3D volume position** in `[-0.5, 0.5]^3` where:
  - `localPos.x`: Left (`-0.5`) to Right (`+0.5`)
  - `localPos.y`: Bottom (`-0.5`) to Top (`+0.5`) *(Note: `+Y` is UP in 3D, whereas `+Y` is DOWN in 2D image pixels!)*
  - `localPos.z`: Back (`-0.5`) to Front (`+0.5`)

Add a color helper function for your `shapeType` (e.g. `shapeType == 3`):

```glsl
vec3 getCustomShapeColor(vec3 p, int theme) {
  if (theme == 2) {
    // Theme 2: Cryo Silver-Slate Monochrome
    return vec3(0.58, 0.63, 0.70);
  }
  // Example: Smooth 2D/3D gradient across the logo
  vec3 colTopLeft     = vec3(0.98, 0.26, 0.21); // Red/Coral
  vec3 colBottomRight = vec3(0.26, 0.52, 0.96); // Blue
  float t = smoothstep(-0.35, 0.35, p.x - p.y);
  return mix(colTopLeft, colBottomRight, t);
}
```
And call it inside `getBrandRegionColor`:
```glsl
if (shapeType == 3) {
  return getCustomShapeColor(localPos, theme);
}
```

---

## Recipe 2: How to Embed the Particle System into Any Existing Website

To drop this particle engine into **any** existing HTML, React, Vue, or Next.js website:

1. Copy these 4 items into your target project's public/static directory:
   - `js/particle-portal.js`
   - `js/fluid-sim.js`
   - `vendor/three.module.js`
   - `assets/volumes/<your_shape>_64.bin`

2. Add a `<canvas id="portal-canvas"></canvas>` inside your hero section or background container:
   ```html
   <div style="position: relative; width: 100%; height: 100vh; overflow: hidden; background: #9aa3b2;">
     <canvas id="portal-canvas" style="width: 100%; height: 100%; display: block;"></canvas>
   </div>

   <script type="module" src="./js/particle-portal.js"></script>
   ```

3. **Headless / Minimal Website Mode (No HUD Sidebar Required)**:
   - `ParticlePortalApp` in [`js/particle-portal.js`](./js/particle-portal.js) uses optional chaining (`?.`) on all HUD DOM elements (`#fps-val`, `#shape-switcher`, `#ctrl-count`, etc.).
   - If you only include `<canvas id="portal-canvas"></canvas>` without any of the HUD HTML elements from `index.html`, the 3D particle simulation and mouse fluid interaction will run **automatically in clean canvas-only mode** with zero console errors!
   - You can also control it programmatically from JavaScript via `window.portalApp`:
     ```javascript
     // Switch shape by index (0, 1, 2...)
     window.portalApp.selectShape(0);

     // Switch color theme (0 = Brand Classic, 1 = Brand Gradient, 2 = Monochrome)
     window.portalApp.setColorTheme(1);

     // Toggle 360-degree drag-orbit mode
     window.portalApp.setFreeRotateMode(true);
     ```

---

## ⚠️ The 7 Golden Rules (Read to Avoid 3D SDF & GPGPU Bugs)

If you write custom analytical SDFs in [`generate_volumes.py`](./generate_volumes.py) or modify the shaders in [`js/particle-portal.js`](./js/particle-portal.js), **never violate these 7 rules**:

### Rule 1: NEVER Use Boolean Subtraction `max(d_outer, -d_inner)` for Hollow Rings
- **Why**: In 3D SDF particle physics, `max(d_outer, -d_inner)` creates a sharp ridge inside the hollow cavity where the gradient $\nabla d = 0$. Particles pushed into the center hole by mouse fluid get trapped in a permanent floating cluster inside the hollow space.
- **Fix**: Either use [`bake_image_or_emoji_to_sdf.py`](./bake_image_or_emoji_to_sdf.py) (which computes the true 2D Euclidean Distance Transform) OR construct hollow rings as the union (`min()`) of exact centerline primitives (`abs(r - R_center) - half_width`).

### Rule 2: NEVER Butt-Joint Two Primitives Edge-to-Edge Without Overlapping
- **Why**: If two segments meet at an internal seam (e.g. a horizontal bar ending at `x = 117.5` meeting an arc starting at `x = 117.5`), both have $d = 0$ on the seam instead of $d = -\text{half\_width}$. When extruded to 3D with `sd_extrude_exact`, $d_{2d} \to 0$ pinches the 3D Z-thickness to zero along that line, creating a visible vertical crease/depth line across the particles.
- **Fix**: Use a single $C^1$-continuous clamped stadium (`cx = clamp(x, x_left, x_right)`) or overlap the inner cores deeply so `min(d1, d2) == -half_width` everywhere along the join.

### Rule 3: NEVER Vary Z-Thickness (`hz`) Across Internal Color Seams
- **Why**: Adding a `smoothstep` Z-bulge to one lobe of a flat logo creates a physical 3D cliff where the thicker lobe meets the thinner lobe. Under directional overhead lighting, that cliff casts a dark shadow ridge right across the color transition.
- **Fix**: Keep `hz` (`0.088`) and `bevel` (`0.054`) uniform across the entire 2D-extruded logo using `sd_extrude_exact(d2d, pz, hz, bevel)`.

### Rule 4: NEVER Put Backticks (`` ` ``) Inside GLSL Shader Comments in JavaScript
- **Why**: `SIM_FRAGMENT_SHADER`, `VERTEX_SHADER`, and `FRAGMENT_SHADER` in [`js/particle-portal.js`](./js/particle-portal.js) are JavaScript template literals (enclosed in `` `...` ``). Writing a backtick inside a GLSL comment (`// see `toOrig` below`) immediately terminates the JavaScript template string and throws `Uncaught SyntaxError: Unexpected identifier`.
- **Fix**: Use single quotes (`'toOrig'`) inside GLSL comments.

### Rule 5: Keep the Hollow-Cavity Guard (`cavityGuard` & `outsideBoost`) Intact in `SIM_FRAGMENT_SHADER`
- **Why**: When a particle is pushed into the hollow hole of a ring logo, its home-voxel spring (`toOrig`) pulls it across the hole toward its true home, while the local SDF surface gradient (`-grad`) pulls it toward the nearest wall. If those two forces oppose each other ($180^\circ$), they cancel out and trap the particle in mid-air.
- **Fix**: Keep `outsideBoost` on Force 3 and `cavityGuard = smoothstep(-0.15, 0.25, dot(-grad, origDir))` on Force 4 in [`js/particle-portal.js`](./js/particle-portal.js).

### Rule 6: Match Color Partition Math to Solid vs. Hollow Topology
- **Why**: A hollow ring has no particles at `(0, 0)`, so an angular gradient `atan(y, x)` looks clean on a ring. On a **solid** 3D shape (like `CLOUD CUMULUS 3D`), placing an `atan(y, x)` pivot inside the volume creates an ugly pinwheel singularity where all colors collide at a single point.
- **Fix**: For solid 3D volumes, use smooth linear/radial field blends or place angular pivots inside a single-color region so the pivot point is masked.

### Rule 7: Remember `+Y` Polarity (`+Y` is DOWN in 2D SVG/Images, `+Y` is UP in 3D Shaders)
- **Why**: Passing `p.y` directly into 2D SVG pixel formulas flips the shape or colors upside-down.
- **Fix**: When converting 3D `localPos` (`p.x, p.y`) to 2D SVG pixel coordinates `(x_px, y_px)`, always negate `p.y`:
  ```glsl
  float x_px = center_x + p.x * scale;
  float y_px = center_y - p.y * scale; // Note the minus sign!
  ```

### Rule 8: Guard Overlapping Color Layers Against Turbulent Particle Bleed in Hollow Cavities
- **Why**: 3D curl-noise turbulence pushes surface particles `3–10 px` outside the exact `d = 0` boundary of a shape. When a foreground segment (like the Left Blue C-Ring + inner hook) overlaps a background arch (like the Red Top Arch `59.25 <= rT <= 104.25`), testing only `d_foreground <= 0` causes turbulent particles floating off the foreground hook tip inside the hollow cavity (`rT < 59.25`) or outside the arch shoulder (`rT > 104.25`) to pick up the background arch's color.
- **Fix**: Subtract the distance outside the background arch's radial band (`d_outside_arch = max(max(r_inner - rT, rT - r_outer), 0.0)`) from `d_foreground` (`d_foreground_guarded = d_foreground - d_outside_arch * 1.35`). On the background arch itself (`r_inner <= rT <= r_outer`), `d_outside_arch == 0` so the overlap seam stays razor-sharp, while inside the hollow cavity or outside the shoulder all turbulent particles stay locked to the foreground color.

---

## 5. Zero-to-One Blueprint: Recreating the Entire Engine From Scratch

If you need to rebuild the entire 3D SDF Volume + 2D Navier-Stokes Fluid Particle Engine **from scratch in a brand-new codebase** (without copying existing files), implement these **5 Subsystems in exact order**:

### Subsystem 1: Offline `64×64×64` 3D SDF Volume Baker (`1,048,576` Bytes `.bin`)
1. Loop `iz, iy, ix` in `0..63` and map each voxel center to normalized cube coordinates `x = (ix + 0.5) / 64.0 - 0.5`, `y = (iy + 0.5) / 64.0 - 0.5`, `z = (iz + 0.5) / 64.0 - 0.5` (`[-0.5, 0.5]^3`).
2. Evaluate the 3D signed distance `d(x, y, z)` (`< 0` inside, `0` on surface, `> 0` outside):
   - **From 2D Image/SVG/Emoji**: Rasterize to a `256×256` mask, compute the exact 2D Euclidean Distance Transform (`d_outside - d_inside`) via the Felzenszwalb & Huttenlocher `O(W * H)` lower parabolic envelope algorithm, normalize by `256 / 0.86`, and extrude along `z` via:
     ```python
     def sd_extrude_exact(d2d, pz, hz=0.088, bevel=0.054):
         dz = abs(pz) - (hz - bevel)
         dx = d2d + bevel
         if dz <= 0.0 and dx <= 0.0:
             return max(d2d, abs(pz) - hz)
         return math.hypot(max(dx, 0.0), max(dz, 0.0)) - bevel
     ```
3. Compute the normalized central-difference gradient `(gx, gy, gz)` and pack 4 bytes per voxel at `idx = ((iz * 64 + iy) * 64 + ix) * 4`:
   - `R, G, B`: `clamp(round((g * 0.5 + 0.5) * 255), 0, 255)`
   - `A`: `clamp(round(127.5 + d * 32.0 * 127.5), 0, 255)` (`<= 128` indicates interior voxels `d <= 0`).

### Subsystem 2: 2D Screen-Space Navier-Stokes Fluid Solver (`128×128` Ping-Pong FBOs)
Run 6 full-screen quad WebGL passes per frame on `128×128` `HalfFloatType`/`FloatType` render targets:
1. **Splat Pass**: Inject a Gaussian cursor velocity impulse `dv * exp(-dist2 / radius)` with aspect-ratio correction.
2. **Curl Pass**: Compute scalar 2D vorticity `w = dvy/dx - dvx/dy`.
3. **Vorticity Confinement Pass**: Add swirling force proportional to `(d|w|/dy, -d|w|/dx) * w` (`curlStrength = 18.0`).
4. **Divergence Pass**: Compute `div = dvx/dx + dvy/dy`.
5. **Jacobi Pressure Pass (`8–12` iterations)**: Relax `p = (pL + pR + pB + pT - div) * 0.25`.
6. **Gradient Subtraction & Semi-Lagrangian Advection Pass**: Subtract `grad(p)` from `v`, back-trace `uv - v * dt`, multiply by `velocityDissipation = 0.96`, and output `tVel`.

### Subsystem 3: WebGL2 MRT GPGPU Particle Ping-Pong (`388×388` = `150,544` Particles)
1. **Dual-Attachment Render Targets (`rt1`, `rt2`)**: Create two `THREE.WebGLRenderTarget(388, 388, { count: 2, type: THREE.FloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter })`:
   - Attachment 0 (`tTexture1`): `vec4(pos.x, pos.y, pos.z, shadow)`
   - Attachment 1 (`tTexture2`): `vec4(vel.x, vel.y, vel.z, smoothSpeed)`
2. **Interior Home-Tether Seeding (`tOrig`)**: Scan the `.bin` buffer for all voxels with `buf[idx + 3] <= 128` (`d <= 0`). Seed each of the `150,544` particles inside a randomly chosen interior voxel with sub-voxel jitter `(ix + rand()) / 64.0 - 0.5` scaled by `uCubeSize / uVolumeScale` (`0.45`). Upload the `.bin` buffer as a `THREE.Data3DTexture(uint8, 64, 64, 64)` with `THREE.LinearFilter`.
3. **GLSL3 MRT Simulation Shader (`SIM_FRAGMENT_SHADER`)**:
   - Sample `tVolume` at `samplePos = rotateY(uRotation) * (currentPos.xyz / uCubeSize) * uVolumeScale + 0.5`.
   - Decode `grad = normalize(volData.rgb * 2.0 - 1.0) * rotateY(uRotation)` and `dist = (volData.a * 2.0 - 1.0) * 2.0`.
   - Sum the **4 Coupled Forces**:
     1. **Screen-Space Fluid Impulse**: Project `currentPos.xyz` to screen UV, sample `tVel`, push along camera `right`/`up`, and compute `invFluidStrength = clamp(1.0 - length(vel) * 0.65 * uInteractForce, 0.05, 1.0)`.
     2. **4D Bitangent Curl Noise**: Cross product of two independent 4D simplex noise gradients (`BitangentNoise4D(vec4(currentPos.xyz * 7.0, uTime))`) for divergence-free 3D turbulence.
     3. **Volume-Locked Home-Tether Spring**: Pull toward `worldOrig = origPos.xyz * rotateY(uRotation)` with `outsideBoost = 1.0 + smoothstep(0.01, 0.12, dist) * 2.2`.
     4. **SDF Surface Containment Pull**: Pull external particles (`dist > 0`) along `-grad`, gated by `cavityGuard = smoothstep(-0.15, 0.25, dot(-grad, normalize(toOrig + 1e-5)))` so `-grad` never cancels `toOrig` across hollow cavities.
   - Apply exponential friction `pow(0.9, uDtRatio)`, integrate `currentPos.xyz += currentVel.xyz * uDtRatio`, and compute wrap-diffuse subsurface lighting + interior depth shadowing (`smoothstep(-0.075, -0.002, dist)`) into `outPos.a`.

### Subsystem 4: Impostor Sphere Point-Sprite Renderer (`THREE.Points`)
- Render `150,544` points reading `tTexture1` and `tTexture2` via a `texuv` buffer attribute.
- In the vertex shader, compute volume-local coordinates `vLocalPos = rotateY(uRotation) * (pos / uCubeSize) * uVolumeScale` and perspective point size `gl_PointSize = uSize / max(length(vPos.xyz), 0.1) * (uResolution.y / 1300.0)`.
- In the fragment shader, discard pixels outside `length(gl_PointCoord - 0.5) > 0.5`, shade each point using `vShadow` (`posData.w`), and evaluate your brand color regions from `vLocalPos`.

### Subsystem 5: Ambient Stage & Volume Morphing
- To morph between shapes at runtime, simply upload the new `.bin` buffer into `tVolume` and re-seed `tOrig` with the new shape's interior voxels—the GPGPU spring + SDF forces automatically fly all `150,544` particles into the new 3D sculpture.

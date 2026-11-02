/**
 * 150,544-Particle WebGL2 MRT 3D OpenVDB/SDF Volume & Fluid Portal Engine
 * Multi-Render-Target GPGPU simulation coupled with a 128x128 2D Navier-Stokes fluid solver,
 * sculpted around 64^3 3D Signed Distance Field volumes (`google_cloud_64.bin`, `google_cloud_solid_64.bin`, `gemini_spark_64.bin`).
 */
import * as THREE from 'three';
import { FluidSim } from './fluid-sim.js';

const SIM_VERTEX_SHADER = /* glsl */ `
  in vec3 position;
  void main() {
    gl_Position = vec4(position, 1.0);
  }
`;

const SIM_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  precision highp int;
  precision highp sampler2D;
  precision highp sampler3D;

  layout(location = 0) out vec4 outPos;
  layout(location = 1) out vec4 outVel;

  uniform sampler2D tTexture1;
  uniform sampler2D tTexture2;
  uniform sampler2D tOrig;
  uniform sampler2D tVel;
  uniform sampler3D tVolume;

  uniform mat4 uProjMatrix;
  uniform mat4 uViewMatrix;
  uniform mat4 uModelMatrix;

  uniform float uCubeSize;
  uniform float uVolumeScale;
  uniform vec3 uLightPos;
  uniform float uRotation;
  uniform float uTime;
  uniform float uDtRatio;
  uniform float uAdditionalNoise;
  uniform float uShowNoise;
  uniform float uInteractForce;
  uniform float uCurlMultiplier;
  uniform float uSurfacePullMultiplier;

  float lerpCoefFPS(float t, float dt) {
    return 1.0 - exp2(log2(1.0 - t) * dt);
  }

  float frictionFPS(float t, float dt) {
    return exp2(log2(t) * dt);
  }

  float lerpFPS(float x, float y, float t, float dt) {
    return mix(x, y, lerpCoefFPS(t, dt));
  }

  mat3 rotateY(float rad) {
    float c = cos(rad);
    float s = sin(rad);
    return mat3(
      c, 0.0, -s,
      0.0, 1.0, 0.0,
      s, 0.0, c
    );
  }

  uvec2 _pcg4d16(uvec4 p) {
    uvec4 v = p * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    return v.xy;
  }

  vec4 _gradient4d(uint hash) {
    vec4 g = vec4(uvec4(hash) & uvec4(0x80000u, 0x40000u, 0x20000u, 0x10000u));
    return g * (1.0 / vec4(0x40000, 0x20000, 0x10000, 0x8000)) - 1.0;
  }

  vec3 BitangentNoise4D(vec4 p) {
    const vec4 F4 = vec4(0.309016994374947451);
    const vec4 C = vec4(
      0.138196601125011,
      0.276393202250021,
      0.414589803375032,
      -0.447213595499958
    );
    vec4 i = floor(p + dot(p, F4));
    vec4 x0 = p - i + dot(i, C.xxxx);
    vec4 i0;
    vec3 isX = step(x0.yzw, x0.xxx);
    vec3 isYZ = step(x0.zww, x0.yyz);
    i0.x = isX.x + isX.y + isX.z;
    i0.yzw = 1.0 - isX;
    i0.y += isYZ.x + isYZ.y;
    i0.zw += 1.0 - isYZ.xy;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;
    vec4 i3 = clamp(i0, 0.0, 1.0);
    vec4 i2 = clamp(i0 - 1.0, 0.0, 1.0);
    vec4 i1 = clamp(i0 - 2.0, 0.0, 1.0);
    vec4 x1 = x0 - i1 + C.xxxx;
    vec4 x2 = x0 - i2 + C.yyyy;
    vec4 x3 = x0 - i3 + C.zzzz;
    vec4 x4 = x0 + C.wwww;
    i = i + 32768.5;
    uvec2 hash0 = _pcg4d16(uvec4(i));
    uvec2 hash1 = _pcg4d16(uvec4(i + i1));
    uvec2 hash2 = _pcg4d16(uvec4(i + i2));
    uvec2 hash3 = _pcg4d16(uvec4(i + i3));
    uvec2 hash4 = _pcg4d16(uvec4(i + 1.0));
    vec4 p00 = _gradient4d(hash0.x);
    vec4 p01 = _gradient4d(hash0.y);
    vec4 p10 = _gradient4d(hash1.x);
    vec4 p11 = _gradient4d(hash1.y);
    vec4 p20 = _gradient4d(hash2.x);
    vec4 p21 = _gradient4d(hash2.y);
    vec4 p30 = _gradient4d(hash3.x);
    vec4 p31 = _gradient4d(hash3.y);
    vec4 p40 = _gradient4d(hash4.x);
    vec4 p41 = _gradient4d(hash4.y);
    vec3 m0 = clamp(0.6 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0, 1.0);
    vec2 m1 = clamp(0.6 - vec2(dot(x3, x3), dot(x4, x4)), 0.0, 1.0);
    vec3 m02 = m0 * m0;
    vec3 m03 = m02 * m0;
    vec2 m12 = m1 * m1;
    vec2 m13 = m12 * m1;
    vec3 temp0 = m02 * vec3(dot(p00, x0), dot(p10, x1), dot(p20, x2));
    vec2 temp1 = m12 * vec2(dot(p30, x3), dot(p40, x4));
    vec4 grad0 = -6.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
    grad0 += m03.x * p00 + m03.y * p10 + m03.z * p20 + m13.x * p30 + m13.y * p40;
    temp0 = m02 * vec3(dot(p01, x0), dot(p11, x1), dot(p21, x2));
    temp1 = m12 * vec2(dot(p31, x3), dot(p41, x4));
    vec4 grad1 = -6.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
    grad1 += m03.x * p01 + m03.y * p11 + m03.z * p21 + m13.x * p31 + m13.y * p41;
    return cross(grad0.xyz, grad1.xyz) * 81.0;
  }

  vec4 hashRand(ivec2 uv) {
    uvec2 h0 = _pcg4d16(uvec4(uint(uv.x), uint(uv.y), 37u, 109u));
    uvec2 h1 = _pcg4d16(uvec4(uint(uv.y), uint(uv.x), 211u, 53u));
    return vec4(
      float(h0.x & 0xffffu),
      float(h0.y & 0xffffu),
      float(h1.x & 0xffffu),
      float(h1.y & 0xffffu)
    ) / 65535.0;
  }

  void main() {
    ivec2 uv = ivec2(gl_FragCoord.xy);
    vec4 currentPos = texelFetch(tTexture1, uv, 0);
    vec4 currentVel = texelFetch(tTexture2, uv, 0);
    vec4 vRand = hashRand(uv);

    // Constant Y rotation of the 3D volume
    mat3 rotMatrix = rotateY(uRotation);

    // Sample 3D SDF Volume (.rgb = surface gradient, .a = signed distance)
    vec3 samplePos = rotMatrix * (currentPos.xyz / uCubeSize) * uVolumeScale + 0.5;
    vec4 volData = texture(tVolume, clamp(samplePos, 0.0, 1.0));
    vec3 rawGrad = volData.rgb * 2.0 - 1.0;
    float gradLen = max(length(rawGrad), 1e-4);
    vec3 grad = (rawGrad / gradLen) * rotMatrix;
    float dist = (volData.a * 2.0 - 1.0) * 2.0;

    // 1. Coupled 2D Screen-Space Navier-Stokes Fluid Interaction
    float pushForce = 0.00055;
    vec4 wPos = uModelMatrix * vec4(currentPos.xyz, 1.0);
    vec4 vPos = uViewMatrix * wPos;
    vec4 posProjected = uProjMatrix * vPos;
    vec2 uvScreen = (posProjected.xy / max(posProjected.w, 0.0001) + 1.0) * 0.5;
    vec3 vel = texture(tVel, clamp(uvScreen, 0.0, 1.0)).xyz;
    vec3 up = vec3(uViewMatrix[0][1], uViewMatrix[1][1], uViewMatrix[2][1]);
    vec3 right = vec3(uViewMatrix[0][0], uViewMatrix[1][0], uViewMatrix[2][0]);
    vec3 disp = right * vel.x + up * vel.y;
    currentVel.xyz += disp * pushForce * uDtRatio * uInteractForce;
    float invFluidStrength = clamp(1.0 - length(vel) * 0.65 * uInteractForce, 0.05, 1.0);

    float additionalNoise = max(uAdditionalNoise, uShowNoise);

    // 2. 4D Bitangent Curl Noise Turbulence
    float force1 = (0.00016 * (0.7 + 0.3 * vRand.z) * uCurlMultiplier) + 0.00055 * additionalNoise;
    currentVel.xyz += BitangentNoise4D(vec4(currentPos.xyz * 7.0, uTime * (1.0 + 0.7 * vRand.y))) * force1 * uDtRatio;

    // 3. Volume-Locked Home-Tether Spring (rotates in lockstep with the 3D SDF volume)
    vec4 origPos = texelFetch(tOrig, uv, 0);
    vec3 worldOrig = origPos.xyz * rotMatrix;
    vec3 toOrig = worldOrig - currentPos.xyz;
    // Boost home-tether spring when a particle is outside the volume (dist > 0) so zero particles can ever linger in hollow cavities
    float outsideBoost = 1.0 + smoothstep(0.01, 0.12, dist) * 2.2;
    currentVel.xyz += toOrig * 0.0022 * outsideBoost * uDtRatio * invFluidStrength * (1.0 - 0.75 * additionalNoise);

    // 4. SDF Surface Containment Pull (pulls external particles dist > 0 inward along -grad)
    // Suppress -grad if it points opposite to toOrig across a hollow cavity so Force 3 and Force 4 can never cancel in mid-air
    float force2 = 0.0022 * (0.7 + 0.3 * vRand.w) * uSurfacePullMultiplier;
    float signForce = mix(0.0, -0.38 - clamp(dist, 0.0, 0.5) * 0.65, sign(dist) + 1.0);
    float cavityGuard = smoothstep(-0.15, 0.25, dot(-grad, normalize(toOrig + vec3(1e-5))));
    currentVel.xyz += grad * force2 * signForce * cavityGuard * uDtRatio * invFluidStrength;

    // Frame-rate independent exponential friction
    currentVel.xyz *= frictionFPS(0.9, uDtRatio);

    // Integrate position
    currentPos.xyz += currentVel.xyz * uDtRatio;

    // Cylindrical portal containment boundary (matches geodesic forcefield cage)
    currentPos.y = clamp(currentPos.y, -0.34, 0.35);
    float xzLen = length(currentPos.xz);
    if (xzLen > 0.275) {
      currentPos.xz = (currentPos.xz / xzLen) * 0.275;
    }

    // NVIDIA GPU Gems Ch.16 Wrap-Diffuse Subsurface Scattering + Interior Depth Shadowing
    vec3 lightPos = normalize(uLightPos);
    float wrap = 0.42;
    float dp = dot(lightPos, grad);
    float wrapDiffuse = max(0.0, (dp + wrap) / (1.0 + wrap));
    wrapDiffuse += max(0.0, -dp) * 0.16;

    float targetShadow = mix(wrapDiffuse * 0.42, wrapDiffuse, smoothstep(-0.075, -0.002, dist));
    targetShadow = clamp(targetShadow * 1.18 + 0.14, 0.0, 1.0);
    currentPos.a = mix(targetShadow, currentPos.a, additionalNoise);

    // Smooth speed for motion-blur alpha & fast-particle highlight tinting
    currentVel.a = lerpFPS(currentVel.a, length(currentVel.xyz), 0.035, uDtRatio);

    outPos = currentPos;
    outVel = currentVel;
  }
`;

const RENDER_VERTEX_SHADER = /* glsl */ `
  attribute vec2 texuv;

  uniform sampler2D tTexture1;
  uniform sampler2D tTexture2;
  uniform float uSize;
  uniform vec2 uResolution;
  uniform float uRotation;
  uniform float uCubeSize;
  uniform float uVolumeScale;

  varying float vShadow;
  varying float vVel;
  varying vec3 vLocalPos;

  mat3 rotateY(float rad) {
    float c = cos(rad);
    float s = sin(rad);
    return mat3(
      c, 0.0, -s,
      0.0, 1.0, 0.0,
      s, 0.0, c
    );
  }

  void main() {
    vec4 posData = texture2D(tTexture1, texuv);
    vec4 velData = texture2D(tTexture2, texuv);

    vec3 pos = posData.xyz;
    vec4 vPos = modelViewMatrix * vec4(pos, 1.0);

    vShadow = posData.w;
    vVel = velData.w;
    // Normalized volume-local coordinates in [-0.5, 0.5]^3 matching generate_volumes.py
    vLocalPos = rotateY(uRotation) * (pos / uCubeSize) * uVolumeScale;

    gl_Position = projectionMatrix * vPos;
    gl_PointSize = uSize / max(length(vPos.xyz), 0.1) * (uResolution.y / 1300.0);
  }
`;

const RENDER_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;

  uniform vec3 uColorLight;
  uniform vec3 uColorDark;
  uniform vec3 uColorInitial;
  uniform vec3 uColorFast;
  uniform vec3 uLightPos;
  uniform float uVisible;
  uniform float uAlpha;
  uniform float uInitialGlow;
  uniform int uPaletteMode;
  uniform int uVolumeIndex;

  varying float vShadow;
  varying float vVel;
  varying vec3 vLocalPos;

  float fit(float value, float min1, float max1, float min2, float max2) {
    return clamp(min2 + (value - min1) * (max2 - min2) / (max1 - min1), min(min2, max2), max(min2, max2));
  }

  float linearstep(float begin, float end, float t) {
    return clamp((t - begin) / (end - begin), 0.0, 1.0);
  }

  vec3 getGoogleCloudQuadrantColor(vec3 p, float ramp) {
    // Official Google Cloud Classic 4-segment brand colors:
    vec3 cRed    = vec3(0.918, 0.263, 0.208); // #EA4335
    vec3 cBlue   = vec3(0.259, 0.522, 0.957); // #4285F4
    vec3 cGreen  = vec3(0.204, 0.659, 0.325); // #34A853
    vec3 cYellow = vec3(0.984, 0.737, 0.016); // #FBBC05

    // Exact 348x348 reference pixel space mapping
    const float S = 257.0 / 0.86;
    float x_px = 173.5 + p.x * S;
    float y_px = 171.0 - p.y * S;

    // Shared 45-degree diagonal axis (x_px + y_px = 333.0) for Y/G seam, Yellow inner tip, and R/B seam
    float d_diag = (x_px + y_px - 333.0) * 0.70710678;

    float rL = length(vec2(x_px - 117.5, y_px - 201.5));
    float d_annL = max(rL - 72.5, 27.5 - rL);

    float rT = length(vec2(x_px - 173.5, y_px - 172.25));
    float d_annT = max(rT - 104.25, 59.25 - rT);

    float rR = length(vec2(x_px - 229.5, y_px - 201.5));
    float d_annR = max(rR - 72.5, 27.5 - rR);

    // 1. Yellow: 180-deg half-annulus at C_L cut by d_diag <= 0 (includes inner extension inside hole)
    float d_yellow = (d_annL <= 0.0 && d_diag <= 0.0) ? max(d_annL, d_diag) : length(max(vec2(d_annL, d_diag), 0.0));

    // 2. Red: Top arch at C_T cut by d_diag <= 0 on right and strictly outside Yellow's disk (rL >= 72.5) and central cavity (rT >= 59.25)
    float d_red = max(max(d_annT, d_diag), max(max((72.5 - rL) * 2.5, (59.25 - rT) * 2.5), y_px - 180.0));

    // 3. Green: Bottom-left arc of C_L (d_diag >= 0) + left half of horizontal bottom bar (x_px <= 173.5)
    float d_bottom_left_band = (x_px < 117.5) ? d_annL : max(229.0 - y_px, y_px - 274.0);
    float d_green = max(max(d_bottom_left_band, -d_diag), max(x_px - 173.5, 201.5 - y_px));

    // 4. Blue: Top-right arc of C_T (d_diag >= 0) + right lobe C_R + right half of horizontal bottom bar (x_px >= 173.5)
    float d_blue_top = max(max(d_annT, -d_diag), max(y_px - 165.0, 173.5 - x_px));
    float d_blue_ring_and_bar = (x_px >= 229.5) ? d_annR : max(max(229.0 - y_px, y_px - 274.0), 173.5 - x_px);
    float d_blue = min(d_blue_top, d_blue_ring_and_bar);

    // Soft-min exponential weights across the exact piece SDFs for crisp seams
    float dMin = min(min(d_yellow, d_red), min(d_green, d_blue));
    float k = 0.65;
    float wYellow = exp(-(d_yellow - dMin) * k);
    float wRed    = exp(-(d_red    - dMin) * k);
    float wGreen  = exp(-(d_green  - dMin) * k);
    float wBlue   = exp(-(d_blue   - dMin) * k);

    float sumW = max(wRed + wBlue + wGreen + wYellow, 0.0001);
    vec3 brandLight = (cRed * wRed + cBlue * wBlue + cGreen * wGreen + cYellow * wYellow) / sumW;

    vec3 brandShadow = mix(vec3(0.08, 0.11, 0.19), brandLight * 0.34, 0.68);
    vec3 brandHighlight = mix(brandLight, vec3(0.92, 0.96, 1.0), 0.18);
    return mix(brandShadow, brandHighlight, clamp(ramp * 1.25, 0.0, 1.0));
  }

  vec3 getGoogleCloudGradientColor(vec3 p, float ramp) {
    // New Updated Google Cloud Gradient Mark (sampled 1:1 from reference PNG):
    // - Left C-ring + inner hook (d_yellow <= 0) is solid Electric Azure Blue (#0087ff) overlapping the Top Arch (#fd3966)
    // - Clockwise perimeter gradient from Top-Left Arch -> Red -> Orange -> Yellow -> Lime -> Kelly Green -> Teal -> Cerulean -> Blue
    const float S = 257.0 / 0.86;
    float x_px = 173.5 + p.x * S;
    float y_px = 171.0 - p.y * S;

    float d_diag = (x_px + y_px - 333.0) * 0.70710678;
    float rL = length(vec2(x_px - 117.5, y_px - 201.5));
    float rT = length(vec2(x_px - 173.5, y_px - 172.25));

    // Distance outside the Left Blue C-Ring + 45-deg inner hook:
    float d_hook_raw = (y_px < 201.5 && x_px > 117.5)
      ? ((rL <= 72.5 && d_diag <= 0.0) ? max(rL - 72.5, d_diag) : length(max(vec2(rL - 72.5, d_diag), 0.0)))
      : max(rL - 72.5, x_px - 175.0);

    // Distance outside the Top Arch's radial band [59.25, 104.25]:
    // On the Top Arch itself (59.25 <= rT <= 104.25), d_outside_arch == 0 so the circular overlap at rL = 72.5 is exact.
    // Inside the central hollow cavity (rT < 59.25) or outside the left shoulder (rT > 104.25), the Top Arch does NOT exist,
    // so subtracting d_outside_arch prevents any particle jittering around the Blue C-Ring or inner hook from picking up Red/Orange!
    float d_outside_arch = (x_px < 215.0 && y_px < 210.0) ? max(max(59.25 - rT, rT - 104.25), 0.0) : 0.0;
    float d_left_blue_disk = d_hook_raw - d_outside_arch * 1.35;

    // Clockwise perimeter angle (0 deg at left-top overlap, sweeping clockwise through top arch, right lobe, and bottom bar)
    float angleDeg = degrees(atan(195.0 - y_px, x_px - 173.5));
    float cw = mod(155.0 - angleDeg, 360.0);

    vec3 cCoralPink = vec3(0.992, 0.224, 0.400); // #fd3966 (cw ~ 18°)
    vec3 cCrimson   = vec3(0.996, 0.129, 0.188); // #fe2130 (cw ~ 41°)
    vec3 cRedOrange = vec3(0.996, 0.220, 0.110); // #fe381c (cw ~ 62°)
    vec3 cOrange    = vec3(1.000, 0.506, 0.004); // #ff8101 (cw ~ 84°)
    vec3 cAmber     = vec3(1.000, 0.780, 0.008); // #ffc702 (cw ~ 109°)
    vec3 cLemon     = vec3(0.949, 0.835, 0.008); // #f2d502 (cw ~ 133°)
    vec3 cLime      = vec3(0.376, 0.824, 0.106); // #60d21b (cw ~ 153°)
    vec3 cGreen     = vec3(0.035, 0.753, 0.290); // #09c04a (cw ~ 171°)
    vec3 cTeal      = vec3(0.000, 0.702, 0.537); // #00b389 (cw ~ 207°)
    vec3 cCerulean  = vec3(0.000, 0.600, 0.906); // #0099e7 (cw ~ 245°)
    vec3 cBlue      = vec3(0.000, 0.529, 1.000); // #0087ff (cw >= 280° & Left C-Ring)

    vec3 loopCol = cCoralPink;
    loopCol = mix(loopCol, cCrimson,   smoothstep(18.0,  41.0,  cw));
    loopCol = mix(loopCol, cRedOrange, smoothstep(41.0,  62.0,  cw));
    loopCol = mix(loopCol, cOrange,    smoothstep(62.0,  84.0,  cw));
    loopCol = mix(loopCol, cAmber,     smoothstep(84.0,  109.0, cw));
    loopCol = mix(loopCol, cLemon,     smoothstep(109.0, 131.0, cw));
    loopCol = mix(loopCol, cLime,      smoothstep(131.0, 152.0, cw));
    loopCol = mix(loopCol, cGreen,     smoothstep(152.0, 174.0, cw));
    loopCol = mix(loopCol, cTeal,      smoothstep(174.0, 210.0, cw));
    loopCol = mix(loopCol, cCerulean,  smoothstep(210.0, 246.0, cw));
    loopCol = mix(loopCol, cBlue,      smoothstep(246.0, 282.0, cw));

    // Crisp circular overlap where the Left Blue C-Ring + inner extension sits over the Coral-Pink Top Arch
    float blueArchMask = 1.0 - smoothstep(-1.4, 1.4, d_left_blue_disk);
    vec3 brandLight = mix(loopCol, cBlue, blueArchMask);

    vec3 brandShadow = mix(vec3(0.06, 0.10, 0.20), brandLight * 0.36, 0.72);
    vec3 brandHighlight = mix(brandLight, vec3(0.95, 0.98, 1.0), 0.16);
    return mix(brandShadow, brandHighlight, clamp(ramp * 1.24, 0.0, 1.0));
  }

  vec3 getCumulusClassicColor(vec3 p, float ramp) {
    // Solid 3D Cumulus Cloud in Classic 4-Color Google Cloud format (no hollow-ring interior gaps)
    vec3 cRed    = vec3(0.918, 0.263, 0.208); // #EA4335
    vec3 cBlue   = vec3(0.259, 0.522, 0.957); // #4285F4
    vec3 cGreen  = vec3(0.204, 0.659, 0.325); // #34A853
    vec3 cYellow = vec3(0.984, 0.737, 0.016); // #FBBC05

    const float S = 257.0 / 0.86;
    float x_px = 173.5 + p.x * S;
    float y_px = 171.0 - p.y * S;

    float d_diag = (x_px + y_px - 333.0) * 0.70710678;
    float rL = length(vec2(x_px - 117.5, y_px - 201.5));

    // Exact 4-region solid partition meeting at the triple point (173.5, 159.5):
    float d_yellow = max(rL - 71.0, d_diag);
    float d_red    = max(71.0 - rL, d_diag);
    float d_green  = max(-d_diag, x_px - 173.5);
    float d_blue   = max(-d_diag, 173.5 - x_px);

    float dMin = min(min(d_yellow, d_red), min(d_green, d_blue));
    float k = 0.55;
    float wYellow = exp(-(d_yellow - dMin) * k);
    float wRed    = exp(-(d_red    - dMin) * k);
    float wGreen  = exp(-(d_green  - dMin) * k);
    float wBlue   = exp(-(d_blue   - dMin) * k);

    float sumW = max(wRed + wBlue + wGreen + wYellow, 0.0001);
    vec3 brandLight = (cRed * wRed + cBlue * wBlue + cGreen * wGreen + cYellow * wYellow) / sumW;

    vec3 brandShadow = mix(vec3(0.08, 0.11, 0.19), brandLight * 0.34, 0.68);
    vec3 brandHighlight = mix(brandLight, vec3(0.92, 0.96, 1.0), 0.18);
    return mix(brandShadow, brandHighlight, clamp(ramp * 1.25, 0.0, 1.0));
  }

  vec3 getCumulusGradientColor(vec3 p, float ramp) {
    // Solid 3D Cumulus Cloud in Updated Google Cloud Gradient format:
    // Pivot (135.0, 195.0) is buried inside the solid Blue Left Lobe (d_left_blue < 0) so the 3D central puffs
    // have zero pinwheel singularity, crisp circular/diagonal overlap at the top-left, and a smooth bottom-bar gradient.
    const float S = 257.0 / 0.86;
    float x_px = 173.5 + p.x * S;
    float y_px = 171.0 - p.y * S;

    float d_diag = (x_px + y_px - 333.0) * 0.70710678;
    float rL = length(vec2(x_px - 117.5, y_px - 201.5));
    float d_left_blue = max(rL - 71.5, d_diag);

    // Pivot inside the solid Blue Left Lobe so angleDeg never has a singularity in the visible rainbow region
    float angleDeg = degrees(atan(195.0 - y_px, x_px - 135.0));
    float cw = mod(130.0 - angleDeg, 360.0);

    vec3 cCoralPink = vec3(0.992, 0.224, 0.400); // #fd3966
    vec3 cCrimson   = vec3(0.996, 0.129, 0.188); // #fe2130
    vec3 cRedOrange = vec3(0.996, 0.220, 0.110); // #fe381c
    vec3 cOrange    = vec3(1.000, 0.506, 0.004); // #ff8101
    vec3 cAmber     = vec3(1.000, 0.780, 0.008); // #ffc702
    vec3 cLemon     = vec3(0.949, 0.835, 0.008); // #f2d502
    vec3 cLime      = vec3(0.376, 0.824, 0.106); // #60d21b
    vec3 cGreen     = vec3(0.035, 0.753, 0.290); // #09c04a
    vec3 cTeal      = vec3(0.000, 0.702, 0.537); // #00b389
    vec3 cCerulean  = vec3(0.000, 0.600, 0.906); // #0099e7
    vec3 cBlue      = vec3(0.000, 0.529, 1.000); // #0087ff

    vec3 loopCol = cCoralPink;
    loopCol = mix(loopCol, cCrimson,   smoothstep(10.0,  32.0,  cw));
    loopCol = mix(loopCol, cRedOrange, smoothstep(32.0,  55.0,  cw));
    loopCol = mix(loopCol, cOrange,    smoothstep(55.0,  78.0,  cw));
    loopCol = mix(loopCol, cAmber,     smoothstep(78.0,  102.0, cw));
    loopCol = mix(loopCol, cLemon,     smoothstep(102.0, 122.0, cw));
    loopCol = mix(loopCol, cLime,      smoothstep(122.0, 138.0, cw));
    loopCol = mix(loopCol, cGreen,     smoothstep(138.0, 155.0, cw));
    loopCol = mix(loopCol, cTeal,      smoothstep(155.0, 174.0, cw));
    loopCol = mix(loopCol, cCerulean,  smoothstep(174.0, 194.0, cw));
    loopCol = mix(loopCol, cBlue,      smoothstep(194.0, 216.0, cw));

    float blueArchMask = 1.0 - smoothstep(-2.0, 2.0, d_left_blue);
    vec3 brandLight = mix(loopCol, cBlue, blueArchMask);

    vec3 brandShadow = mix(vec3(0.06, 0.10, 0.20), brandLight * 0.36, 0.72);
    vec3 brandHighlight = mix(brandLight, vec3(0.95, 0.98, 1.0), 0.16);
    return mix(brandShadow, brandHighlight, clamp(ramp * 1.24, 0.0, 1.0));
  }

  vec3 getGeminiSparkColor(vec3 p, float ramp, bool isClassic) {
    // Official Google Gemini 4-Point AI Spark Color Map (sampled 1:1 from reference PNG):
    // - Top tip (+Y):    Coral Red (#f94543 / #EA4335)
    // - Left tip (-X):   Golden Yellow (#fabc12 / #FBBC05)
    // - Bottom tip (-Y): Emerald Green (#08b962 / #34A853)
    // - Center Core + Entire Right Wing (+X): Gemini Blue (#3186ff / #4285F4)
    vec2 uv = p.xy / 0.42;

    vec3 cRed    = isClassic ? vec3(0.918, 0.263, 0.208) : vec3(0.976, 0.271, 0.263); // #f94543
    vec3 cYellow = isClassic ? vec3(0.984, 0.737, 0.016) : vec3(0.980, 0.737, 0.071); // #fabc12
    vec3 cGreen  = isClassic ? vec3(0.204, 0.659, 0.325) : vec3(0.031, 0.725, 0.384); // #08b962
    vec3 cBlue   = isClassic ? vec3(0.259, 0.522, 0.957) : vec3(0.192, 0.525, 1.000); // #3186ff

    float dRed    = length(vec2((uv.x + 0.10) * 1.15, (uv.y - 0.92) * 0.95));
    float dYellow = length(vec2((uv.x + 0.95) * 0.90, (uv.y - 0.02) * 1.10));
    float dGreen  = length(vec2((uv.x + 0.08) * 1.15, (uv.y + 0.92) * 0.95));

    float innerEdge = isClassic ? 0.36 : 0.16;
    float outerEdge = isClassic ? 0.68 : 0.96;

    float wRed    = 1.0 - smoothstep(innerEdge, outerEdge, dRed);
    float wYellow = 1.0 - smoothstep(innerEdge, outerEdge * 0.94, dYellow);
    float wGreen  = 1.0 - smoothstep(innerEdge, outerEdge, dGreen);

    float totalLobe = wRed + wYellow + wGreen;
    vec3 lobeCol = totalLobe > 0.0001
      ? (cRed * wRed + cYellow * wYellow + cGreen * wGreen) / totalLobe
      : cBlue;
    float lobeAlpha = clamp(max(wRed, max(wYellow, wGreen)) * 1.12, 0.0, 1.0);
    vec3 brandLight = mix(cBlue, lobeCol, lobeAlpha);

    vec3 brandShadow = mix(vec3(0.06, 0.10, 0.20), brandLight * 0.36, 0.72);
    vec3 brandHighlight = mix(brandLight, vec3(0.95, 0.98, 1.0), 0.16);
    return mix(brandShadow, brandHighlight, clamp(ramp * 1.24, 0.0, 1.0));
  }

  void main() {
    float alpha = step(length(gl_PointCoord.xy - 0.5), 0.5) * uVisible;
    if (alpha < 0.001) discard;

    // Sphere impostor normal from gl_PointCoord
    vec2 uv = 2.0 * gl_PointCoord.xy - 1.0;
    vec3 n = vec3(uv.x, -uv.y, sqrt(1.0 - clamp(dot(uv, uv), 0.0, 1.0)));

    vec3 sphereLight = normalize(vec3(-0.42, 0.62, 0.78));
    float lightShadow = max(0.18, dot(sphereLight, normalize(n)));
    float ramp = clamp(lightShadow * vShadow * 1.25, 0.0, 1.0);

    vec3 color;
    if (uPaletteMode == 1) {
      if (uVolumeIndex == 1) {
        color = getCumulusClassicColor(vLocalPos, ramp);
      } else if (uVolumeIndex == 2) {
        color = getGeminiSparkColor(vLocalPos, ramp, true);
      } else {
        color = getGoogleCloudQuadrantColor(vLocalPos, ramp);
      }
    } else if (uPaletteMode == 2) {
      if (uVolumeIndex == 1) {
        color = getCumulusGradientColor(vLocalPos, ramp);
      } else if (uVolumeIndex == 2) {
        color = getGeminiSparkColor(vLocalPos, ramp, false);
      } else {
        color = getGoogleCloudGradientColor(vLocalPos, ramp);
      }
    } else {
      color = mix(uColorDark, uColorLight, ramp);
    }

    // Velocity-based highlight tinting
    float fastFactor = pow(fit(vVel, 0.003, 0.0055, 0.0, 1.0), 2.0);
    color = mix(color, uColorFast, fastFactor);

    // Motion-blur alpha modulation
    alpha *= max(uInitialGlow, pow(fit(vVel, 0.002, 0.007, 1.0, 0.0), 2.0) * 0.5 + 0.5);

    // Opening coalesce glow blend
    vec3 fadeInColor = mix(vec3(1.0), uColorInitial, linearstep(0.0, 1.0, uAlpha));
    color = mix(color, fadeInColor, uInitialGlow);

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), alpha * uAlpha);
  }
`;

const CAGE_VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vPos;
  varying vec3 vNormal;
  varying vec3 vEye;

  void main() {
    vUv = uv;
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vPos = worldPos.xyz;
    vNormal = normalMatrix * normal;
    vEye = -(modelViewMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// Geodesic triangle lattice cylinder shader:
// Razor-thin screen-space anti-aliased triangle wireframe, completely clean in the center behind the particles,
// shimmering softly only along Fresnel grazing edges and pow(noise, 4.0) wave bands.
const CAGE_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  varying vec3 vPos;
  varying vec3 vNormal;
  varying vec3 vEye;

  uniform float uTime;
  uniform float uAlpha;

  float sinenoise1(vec3 p) {
    return sin(p.x * 2.1 + sin(p.y * 2.7 + sin(p.z * 1.9)));
  }

  float geodesicGrid(vec2 uv) {
    // High-frequency fine equilateral triangle lattice
    uv *= vec2(26.0, 9.5);
    vec2 r = vec2(1.0, 1.7320508);
    vec2 h = r * 0.5;
    vec2 a = mod(uv, r) - h;
    vec2 b = mod(uv - h, r) - h;
    vec2 gv = dot(a, a) < dot(b, b) ? a : b;
    float dEdge1 = abs(gv.y);
    float dEdge2 = abs(dot(gv, normalize(vec2(1.7320508, 1.0))));
    float dEdge3 = abs(dot(gv, normalize(vec2(-1.7320508, 1.0))));
    float d = min(dEdge1, min(dEdge2, dEdge3));
    float fw = max(fwidth(d), 0.007);
    float line = 1.0 - smoothstep(0.0, fw * 1.15, d);
    float node = 1.0 - smoothstep(0.0, fw * 2.1, length(gv));
    return clamp(line * 0.55 + node * 0.35, 0.0, 1.0);
  }

  void main() {
    float t = uTime * 0.5;
    float noise = sinenoise1(vPos * 4.0 + vec3(0.0, t * 0.45, -t * 0.13)) * 0.5 + 0.5;
    noise *= sinenoise1(vPos * 2.0 + vec3(t * 0.3, -t * 0.27, t * 0.2)) * 0.5 + 0.5;
    noise = sin(noise * 15.0 - t * 7.0) * 0.5 + 0.5;
    noise = pow(noise, 4.0);

    float triangles = geodesicGrid(vUv + vec2(noise * 0.02));

    float fresnel = 1.0 - max(0.0, dot(normalize(vNormal), normalize(vEye)));
    float softedge = 1.0 - smoothstep(0.65, 0.99, fresnel);
    fresnel = mix(fresnel, 1.0, 0.22);
    float fadetop = 1.0 - smoothstep(0.5, 0.96, vUv.y);
    float fadebottom = smoothstep(0.0, 0.10, vUv.y);

    float alpha = (noise * triangles * 1.9 + noise * 0.025) * fresnel * softedge * fadetop * fadebottom * uAlpha;
    gl_FragColor = vec4(vec3(1.0), clamp(alpha, 0.0, 0.85));
  }
`;

const FLOOR_VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vPos;
  void main() {
    vUv = uv;
    vPos = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// Stepped cryo-silver dais floor with subtle ice-blue recessed step glow
const FLOOR_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  varying vec3 vPos;
  uniform float uTime;
  uniform float uAlpha;

  void main() {
    vec2 p = vPos.xz;
    float r = length(p);
    float angle = atan(p.y, p.x);

    // Subtle matte cryo-silver stepped dais rings
    float innerWell = 1.0 - smoothstep(0.26, 0.31, r);
    float stepRim1 = 1.0 - smoothstep(0.002, 0.010, abs(r - 0.305));
    float stepRim2 = 1.0 - smoothstep(0.002, 0.012, abs(r - 0.425));
    float stepRim3 = 1.0 - smoothstep(0.003, 0.016, abs(r - 0.580));

    // Soft rotating ice-blue recessed groove glow (vec3(0.5, 0.7, 1.0))
    float wave = 0.5 + 0.5 * sin(angle * 2.0 - uTime * 1.2);
    float grooveGlow = (stepRim1 * 0.75 + stepRim2 * 0.45 + stepRim3 * 0.25) * (0.65 + 0.35 * wave);

    // Subtle radial podium shading that blends seamlessly into the light cryo-silver mist
    vec3 baseCol = mix(vec3(0.52, 0.56, 0.65), vec3(0.74, 0.78, 0.86), smoothstep(0.22, 0.65, r));
    baseCol = mix(baseCol, vec3(0.44, 0.48, 0.57), innerWell * 0.45);
    vec3 col = baseCol + vec3(0.55, 0.75, 1.0) * grooveGlow * 0.55;

    float outerFade = 1.0 - smoothstep(0.45, 1.15, r);
    float alpha = clamp((innerWell * 0.32 + grooveGlow * 0.65 + 0.18) * outerFade * uAlpha, 0.0, 0.78);
    gl_FragColor = vec4(col, alpha);
  }
`;

export const VOLUME_CATALOG = [
  {
    id: 'google_cloud',
    index: '01',
    name: 'GOOGLE CLOUD LOGO',
    subtitle: 'Official 3D Hollow Cloud Loop SDF (64³ RGBA8)',
    file: './assets/volumes/google_cloud_64.bin',
    scale: 1.36,
    rotMode: 'sway',
    voxels: '13,488 / 262,144 (5.1%)',
  },
  {
    id: 'google_cloud_solid',
    index: '02',
    name: 'CLOUD CUMULUS 3D',
    subtitle: 'Volumetric Solid 3D Cloud Sculpture (64³ RGBA8)',
    file: './assets/volumes/google_cloud_solid_64.bin',
    scale: 1.36,
    rotMode: 'sway',
    voxels: '28,712 / 262,144 (11.0%)',
  },
  {
    id: 'gemini_spark',
    index: '03',
    name: 'GEMINI AI SPARK',
    subtitle: 'Concave 4-Point Astroid Crystalline Volume (64³ RGBA8)',
    file: './assets/volumes/gemini_spark_64.bin',
    scale: 1.22,
    rotMode: 'spin',
    voxels: '13,736 / 262,144 (5.2%)',
  },
];

export const PALETTE_MODES = {
  cryo: {
    mode: 0,
    light: '#bdc6d4',
    dark: '#222b42',
    initial: '#b5d5ff',
    fast: '#d7ebfa',
  },
  google: {
    mode: 1,
    light: '#e8f0fe',
    dark: '#1a233a',
    initial: '#8ab4f8',
    fast: '#ffffff',
  },
  gradient: {
    mode: 2,
    light: '#eef4ff',
    dark: '#162038',
    initial: '#7fbfff',
    fast: '#ffffff',
  },
};

export class ParticlePortalApp {
  constructor(canvas, onTelemetryUpdate) {
    this.canvas = canvas;
    this.onTelemetryUpdate = onTelemetryUpdate;

    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: 'high-performance',
      alpha: false,
    });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(this.dpr);
    this.renderer.setClearColor(0x9097a8, 1);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, this.width / this.height, 0.1, 30);

    // Camera rig presets for settled portal view
    this.cameraPresets = {
      front: { pos: new THREE.Vector3(0, 0.04, 1.48), target: new THREE.Vector3(0, -0.01, 0) },
      orbit: { pos: new THREE.Vector3(0.85, 0.42, 1.28), target: new THREE.Vector3(0, -0.02, 0) },
      oculus: { pos: new THREE.Vector3(0, 1.42, 0.35), target: new THREE.Vector3(0, -0.05, 0) },
    };
    this.activeCameraPreset = 'front';
    this.activePaletteKey = 'cryo';
    this.cameraBasePos = this.cameraPresets.front.pos.clone();
    this.cameraBaseTarget = this.cameraPresets.front.target.clone();
    this.cameraCurrentPos = new THREE.Vector3(0, 1.35, 2.15); // Starts slightly elevated for intro sweep
    this.cameraCurrentTarget = new THREE.Vector3(0, -0.01, 0);
    this.pointerParallax = new THREE.Vector2(0, 0);

    // 360-Degree Free Rotation Mode state (disables right-side controls & cursor particle disturbance)
    this.freeRotateMode = false;
    this.isDraggingOrbit = false;
    this.dragLastX = 0;
    this.dragLastY = 0;
    this.freeOrbitYaw = 0.0;
    this.freeOrbitPitch = 0.04;
    this.freeOrbitRadius = 1.48;
    this.userInteractForce = 1.0;

    this.camera.position.copy(this.cameraCurrentPos);
    this.camera.lookAt(this.cameraCurrentTarget);

    // 150,544 particles (388 x 388 texture)
    this.texSize = 388;
    this.particleCount = this.texSize * this.texSize;
    this.cubeSize = 0.65;
    this.rotSpeed = 0.75; // rad/s base multiplier

    this.fluidSim = new FluidSim(this.renderer, 128);
    this.volumes = new Map();
    this.currentVolumeIndex = 0;

    // Tween state for morph bursts & intro
    this.uAdditionalNoise = 0.0;
    this.uShowNoise = 1.0;
    this.uAlpha = 0.0;
    this.uInitialGlow = 1.0;
    this.introElapsed = 0;
    this.morphBurstTimer = 0;

    // FPS tracking
    this.lastTime = performance.now();
    this.fpsSmooth = 60;
    this.elapsedTime = 0;

    this._initBackdrop();
    this._initPortalCageAndFloor();
    this._initAmbientSnow();
    this._initGPGPUParticles();
    this._bindEvents();
  }

  _initBackdrop() {
    // Atmospheric lightroom dome with top-right studio glare & dot-matrix twinkle:
    // Lighter cryo-silver studio gradient (#6a6f7d -> #e1e6f1 * 1.1) with subtle dot grid & overhead studio glare
    const geo = new THREE.SphereGeometry(8, 48, 32);
    const mat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uColor1: { value: new THREE.Color('#6a6f7d').multiplyScalar(1.08) },
        uColor2: { value: new THREE.Color('#e1e6f1').multiplyScalar(1.08) },
      },
      vertexShader: /* glsl */ `
        varying vec3 vWorldPos;
        varying vec2 vUv;
        void main() {
          vUv = uv;
          vWorldPos = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        precision highp float;
        varying vec3 vWorldPos;
        varying vec2 vUv;
        uniform float uTime;
        uniform vec3 uColor1;
        uniform vec3 uColor2;

        float sinenoise1(vec3 p) {
          return sin(p.x * 2.1 + sin(p.y * 2.7 + sin(p.z * 1.9)));
        }

        void main() {
          vec3 dir = normalize(vWorldPos);
          // Studio lightroom vertical gradient distribution
          float c = clamp(dir.y * 0.55 + 0.48, 0.0, 1.0);
          float noise = sinenoise1(vWorldPos * 1.8 + uTime * 0.35) * 0.5 + 0.5;
          noise *= sinenoise1(vWorldPos * 0.9 - uTime * 0.22) * 0.5 + 0.5;
          noise = pow(noise, 1.6);

          // Subtle background dot-matrix twinkle
          vec2 dotCell = fract(vec2(atan(dir.z, dir.x) * 28.0, dir.y * 52.0)) - 0.5;
          float dotMask = 1.0 - smoothstep(0.06, 0.14, length(dotCell));
          float dots = dotMask * noise * smoothstep(-0.25, 0.65, dir.y);

          vec3 col = mix(uColor1, uColor2, c);
          col += dots * 0.055;

          // Luminous central studio halo + top-right diagonal glare
          float centerHalo = exp(-length(dir.xy - vec2(0.0, 0.06)) * 2.1) * 0.12;
          float topGlare = exp(-length(dir.xy - vec2(0.28, 0.42)) * 2.4) * 0.14;
          col += vec3(0.88, 0.93, 1.0) * (centerHalo + topGlare);

          gl_FragColor = vec4(col, 1.0);
        }
      `,
    });
    this.backdropMesh = new THREE.Mesh(geo, mat);
    this.scene.add(this.backdropMesh);
  }

  _initPortalCageAndFloor() {
    this.portalGroup = new THREE.Group();
    this.scene.add(this.portalGroup);

    // 1. Geodesic Forcefield Wireframe Cylinder
    const cageGeo = new THREE.CylinderGeometry(0.305, 0.305, 0.92, 64, 6, true);
    cageGeo.translate(0, 0.12, 0);
    this.cageMat = new THREE.ShaderMaterial({
      vertexShader: CAGE_VERTEX_SHADER,
      fragmentShader: CAGE_FRAGMENT_SHADER,
      uniforms: {
        uTime: { value: 0 },
        uAlpha: { value: 0.78 },
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    this.cageMesh = new THREE.Mesh(cageGeo, this.cageMat);
    this.portalGroup.add(this.cageMesh);

    // 2. Overhead Oculus Portal Light Ring
    const oculusGeo = new THREE.PlaneGeometry(1.1, 1.1, 1, 1);
    oculusGeo.rotateX(Math.PI * 0.5);
    oculusGeo.translate(0, 0.56, 0);
    this.oculusMat = new THREE.ShaderMaterial({
      vertexShader: FLOOR_VERTEX_SHADER,
      fragmentShader: /* glsl */ `
        precision highp float;
        varying vec3 vPos;
        uniform float uAlpha;
        void main() {
          float r = length(vPos.xz);
          float ring = 1.0 - smoothstep(0.008, 0.045, abs(r - 0.305));
          float softHalo = exp(-abs(r - 0.305) * 14.0) * 0.55;
          float outerFade = 1.0 - smoothstep(0.32, 0.52, r);
          float a = clamp((ring * 0.85 + softHalo) * outerFade * uAlpha, 0.0, 0.85);
          gl_FragColor = vec4(vec3(0.95, 0.98, 1.0), a);
        }
      `,
      uniforms: {
        uAlpha: { value: 0.88 },
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    this.oculusMesh = new THREE.Mesh(oculusGeo, this.oculusMat);
    this.portalGroup.add(this.oculusMesh);

    // 3. Stepped Cryo-Silver Dais Floor
    const floorGeo = new THREE.PlaneGeometry(2.8, 2.8, 1, 1);
    floorGeo.rotateX(-Math.PI * 0.5);
    floorGeo.translate(0, -0.34, 0);
    this.floorMat = new THREE.ShaderMaterial({
      vertexShader: FLOOR_VERTEX_SHADER,
      fragmentShader: FLOOR_FRAGMENT_SHADER,
      uniforms: {
        uTime: { value: 0 },
        uAlpha: { value: 1.0 },
      },
      transparent: true,
      depthWrite: false,
    });
    this.floorMesh = new THREE.Mesh(floorGeo, this.floorMat);
    this.portalGroup.add(this.floorMesh);

    // 4. Flanking 3D Architectural Reticle Arrows (`<───` and `───>`)
    const buildBracketGeometry = (dirSign) => {
      const positions = new Float32Array([
        // Horizontal stem
        dirSign * 0.35, 0.0, 0.0,
        dirSign * 0.45, 0.0, 0.0,
        // Chevron upper wing
        dirSign * 0.45, 0.0, 0.0,
        dirSign * 0.425, 0.018, 0.0,
        // Chevron lower wing
        dirSign * 0.45, 0.0, 0.0,
        dirSign * 0.425, -0.018, 0.0,
      ]);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      return geo;
    };
    const bracketMat = new THREE.LineBasicMaterial({
      color: 0xf4f8ff,
      transparent: true,
      opacity: 0.52,
      depthWrite: false,
    });
    this.leftBracket = new THREE.LineSegments(buildBracketGeometry(-1), bracketMat);
    this.rightBracket = new THREE.LineSegments(buildBracketGeometry(1), bracketMat);
    this.portalGroup.add(this.leftBracket);
    this.portalGroup.add(this.rightBracket);
  }

  _initAmbientSnow() {
    const count = 260;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3 + 0] = (Math.random() - 0.5) * 3.2;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 2.2;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 2.2;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
      color: 0xf1f6ff,
      size: 0.009,
      transparent: true,
      opacity: 0.26,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.ambientPoints = new THREE.Points(geo, mat);
    this.scene.add(this.ambientPoints);
  }

  _createFallbackVolumeTexture() {
    // 1x1x1 neutral fallback texture until binary volumes finish loading
    const data = new Uint8Array([128, 255, 128, 128]);
    const tex = new THREE.Data3DTexture(data, 1, 1, 1);
    tex.format = THREE.RGBAFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return tex;
  }

  _initGPGPUParticles() {
    const total = this.particleCount;
    const initialPos = new Float32Array(total * 4);
    const initialVel = new Float32Array(total * 4);

    for (let i = 0; i < total; i++) {
      // Spawn inside cylinder radius 0.275, height [-0.34, 0.35]
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.sqrt(Math.random()) * 0.265;
      initialPos[i * 4 + 0] = Math.cos(angle) * radius;
      initialPos[i * 4 + 1] = (Math.random() - 0.5) * 0.64;
      initialPos[i * 4 + 2] = Math.sin(angle) * radius;
      initialPos[i * 4 + 3] = Math.random();
    }

    const makeDataTex = (arr) => {
      const tex = new THREE.DataTexture(arr, this.texSize, this.texSize, THREE.RGBAFormat, THREE.FloatType);
      tex.minFilter = THREE.NearestFilter;
      tex.magFilter = THREE.NearestFilter;
      tex.needsUpdate = true;
      return tex;
    };

    this.tOrig = makeDataTex(initialPos.slice());
    this.initPosTex = makeDataTex(initialPos);
    this.initVelTex = makeDataTex(initialVel);

    // WebGL2 Multiple Render Targets (MRT) ping-pong pair (count: 2)
    const createMRT = () => {
      const mrt = new THREE.WebGLRenderTarget(this.texSize, this.texSize, {
        count: 2,
        type: THREE.FloatType,
        format: THREE.RGBAFormat,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        depthBuffer: false,
        stencilBuffer: false,
      });
      return mrt;
    };

    this.mrtA = createMRT();
    this.mrtB = createMRT();
    this.simFirstStep = true;

    const lightPos = new THREE.Vector3(-0.55, 0.85, 0.75);
    const fallbackVol = this._createFallbackVolumeTexture();

    this.simScene = new THREE.Scene();
    this.simCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    this.simMaterial = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: SIM_VERTEX_SHADER,
      fragmentShader: SIM_FRAGMENT_SHADER,
      uniforms: {
        tTexture1: { value: this.initPosTex },
        tTexture2: { value: this.initVelTex },
        tOrig: { value: this.tOrig },
        tVel: { value: this.fluidSim.texture },
        tVolume: { value: fallbackVol },
        uProjMatrix: { value: new THREE.Matrix4() },
        uViewMatrix: { value: new THREE.Matrix4() },
        uModelMatrix: { value: new THREE.Matrix4() },
        uCubeSize: { value: this.cubeSize },
        uVolumeScale: { value: VOLUME_CATALOG[0].scale },
        uLightPos: { value: lightPos },
        uRotation: { value: 0 },
        uTime: { value: 0 },
        uDtRatio: { value: 1.0 },
        uAdditionalNoise: { value: 0.0 },
        uShowNoise: { value: 1.0 },
        uInteractForce: { value: 1.0 },
        uCurlMultiplier: { value: 1.0 },
        uSurfacePullMultiplier: { value: 1.0 },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.simQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.simMaterial);
    this.simScene.add(this.simQuad);

    // Particle Point Cloud Geometry (150,544 vertices)
    const texuv = new Float32Array(total * 2);
    const dummyPos = new Float32Array(total * 3);
    for (let i = 0; i < total; i++) {
      const u = ((i % this.texSize) + 0.5) / this.texSize;
      const v = (Math.floor(i / this.texSize) + 0.5) / this.texSize;
      texuv[i * 2 + 0] = u;
      texuv[i * 2 + 1] = v;
    }

    const pointsGeo = new THREE.BufferGeometry();
    pointsGeo.setAttribute('position', new THREE.BufferAttribute(dummyPos, 3));
    pointsGeo.setAttribute('texuv', new THREE.BufferAttribute(texuv, 2));

    this.renderMaterial = new THREE.ShaderMaterial({
      vertexShader: RENDER_VERTEX_SHADER,
      fragmentShader: RENDER_FRAGMENT_SHADER,
      uniforms: {
        tTexture1: { value: this.initPosTex },
        tTexture2: { value: this.initVelTex },
        uColorLight: { value: new THREE.Color(PALETTE_MODES.cryo.light) },
        uColorDark: { value: new THREE.Color(PALETTE_MODES.cryo.dark) },
        uColorInitial: { value: new THREE.Color(PALETTE_MODES.cryo.initial) },
        uColorFast: { value: new THREE.Color(PALETTE_MODES.cryo.fast) },
        uLightPos: { value: lightPos },
        uSize: { value: 10.5 },
        uResolution: { value: new THREE.Vector2(this.width * this.dpr, this.height * this.dpr) },
        uVisible: { value: 1.0 },
        uAlpha: { value: 0.0 },
        uInitialGlow: { value: 1.0 },
        uPaletteMode: { value: 0 },
        uVolumeIndex: { value: 0 },
        uRotation: { value: 0 },
        uCubeSize: { value: this.cubeSize },
        uVolumeScale: { value: VOLUME_CATALOG[0].scale },
      },
      transparent: true,
      depthTest: true,
      depthWrite: true,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.SrcAlphaFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
    });

    this.pointsMesh = new THREE.Points(pointsGeo, this.renderMaterial);
    this.pointsMesh.frustumCulled = false;
    this.scene.add(this.pointsMesh);
  }

  _buildVolumeOrigTexture(uint8, scale) {
    const insideVoxels = [];
    const N = 64;
    for (let iz = 0; iz < N; iz++) {
      for (let iy = 0; iy < N; iy++) {
        for (let ix = 0; ix < N; ix++) {
          const idx = (iz * N * N + iy * N + ix) * 4;
          if (uint8[idx + 3] <= 128) {
            insideVoxels.push(ix, iy, iz);
          }
        }
      }
    }

    const total = this.particleCount;
    const origArr = new Float32Array(total * 4);
    const numInside = Math.max(1, insideVoxels.length / 3);
    const factor = this.cubeSize / scale;

    for (let i = 0; i < total; i++) {
      const vIdx = Math.floor(Math.random() * numInside) * 3;
      const sx = (insideVoxels[vIdx + 0] + Math.random()) / N - 0.5;
      const sy = (insideVoxels[vIdx + 1] + Math.random()) / N - 0.5;
      const sz = (insideVoxels[vIdx + 2] + Math.random()) / N - 0.5;
      origArr[i * 4 + 0] = sx * factor;
      origArr[i * 4 + 1] = sy * factor;
      origArr[i * 4 + 2] = sz * factor;
      origArr[i * 4 + 3] = Math.random();
    }

    const tex = new THREE.DataTexture(origArr, this.texSize, this.texSize, THREE.RGBAFormat, THREE.FloatType);
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    return { tex, origArr };
  }

  async loadAllVolumes() {
    const loadOne = async (entry, isFirst = false) => {
      const res = await fetch(entry.file);
      if (!res.ok) throw new Error(`Failed to load volume ${entry.file}`);
      const buf = await res.arrayBuffer();
      const uint8 = new Uint8Array(buf);
      const tex3d = new THREE.Data3DTexture(uint8, 64, 64, 64);
      tex3d.format = THREE.RGBAFormat;
      tex3d.type = THREE.UnsignedByteType;
      tex3d.minFilter = THREE.LinearFilter;
      tex3d.magFilter = THREE.LinearFilter;
      tex3d.wrapS = THREE.ClampToEdgeWrapping;
      tex3d.wrapT = THREE.ClampToEdgeWrapping;
      tex3d.wrapR = THREE.ClampToEdgeWrapping;
      tex3d.needsUpdate = true;

      const { tex: origTex, origArr } = this._buildVolumeOrigTexture(uint8, entry.scale);
      if (isFirst) {
        this.initPosTex.image.data.set(origArr);
        this.initPosTex.needsUpdate = true;
      }
      this.volumes.set(entry.id, { texture: tex3d, origTex, entry });
    };

    // Load default Google Cloud 3D SDF volume first and run 65-step GPU warmup (`initializeShape`)
    await loadOne(VOLUME_CATALOG[0], true);
    const first = this.volumes.get(VOLUME_CATALOG[0].id);
    this.simMaterial.uniforms.tVolume.value = first.texture;
    this.simMaterial.uniforms.tOrig.value = first.origTex;
    this.simMaterial.uniforms.uVolumeScale.value = first.entry.scale;
    this.renderMaterial.uniforms.uVolumeScale.value = first.entry.scale;
    this.simFirstStep = true;

    // Pre-settle particles into the initial 3D volume (`initializeShape`)
    this.simMaterial.uniforms.uShowNoise.value = 0.0;
    for (let i = 0; i < 65; i++) {
      this._stepSimulation(1.0, i * 0.016);
    }
    // Prepare for dramatic t=0 opening coalesce animation
    this.uShowNoise = 0.75;
    this.uAlpha = 0.15;
    this.uInitialGlow = 0.9;

    // Load remaining volumes in parallel
    await Promise.all(VOLUME_CATALOG.slice(1).map((entry) => loadOne(entry, false)));
  }

  setVolumeByIndex(index, triggerBurst = true) {
    const wrapped = (index + VOLUME_CATALOG.length) % VOLUME_CATALOG.length;
    const item = VOLUME_CATALOG[wrapped];
    const loaded = this.volumes.get(item.id);
    if (!loaded) return item;

    this.currentVolumeIndex = wrapped;
    this.simMaterial.uniforms.tVolume.value = loaded.texture;
    this.simMaterial.uniforms.tOrig.value = loaded.origTex;
    this.simMaterial.uniforms.uVolumeScale.value = item.scale;
    this.renderMaterial.uniforms.uVolumeScale.value = item.scale;
    this.renderMaterial.uniforms.uVolumeIndex.value = wrapped;

    if (triggerBurst) {
      this.triggerShockwave(1.0);
    }
    return item;
  }

  triggerShockwave(strength = 1.0) {
    this.uAdditionalNoise = strength;
    this.morphBurstTimer = 0.55;
    if (!this.freeRotateMode) {
      const angle = Math.random() * Math.PI * 2;
      this.fluidSim.injectImpulse(0.5, 0.5, Math.cos(angle) * 0.85, Math.sin(angle) * 0.85);
    }
  }

  setPalette(paletteKey) {
    const key = PALETTE_MODES[paletteKey] ? paletteKey : 'cryo';
    const p = PALETTE_MODES[key];
    this.activePaletteKey = key;
    this.renderMaterial.uniforms.uPaletteMode.value = p.mode;
    this.renderMaterial.uniforms.uColorLight.value.set(p.light);
    this.renderMaterial.uniforms.uColorDark.value.set(p.dark);
    this.renderMaterial.uniforms.uColorInitial.value.set(p.initial);
    this.renderMaterial.uniforms.uColorFast.value.set(p.fast);
  }

  setCameraPreset(presetKey) {
    const preset = this.cameraPresets[presetKey] || this.cameraPresets.front;
    this.activeCameraPreset = presetKey;
    this.cameraBasePos.copy(preset.pos);
    this.cameraBaseTarget.copy(preset.target);
    if (this.freeRotateMode) {
      this.setFreeRotateMode(false);
    }
  }

  setFreeRotateMode(enabled) {
    this.freeRotateMode = Boolean(enabled);
    this.isDraggingOrbit = false;
    if (this.freeRotateMode) {
      // Initialize spherical orbit coordinates from current camera offset
      const offset = this.cameraCurrentPos.clone().sub(this.cameraBaseTarget);
      this.freeOrbitRadius = THREE.MathUtils.clamp(offset.length(), 0.85, 2.35);
      this.freeOrbitPitch = THREE.MathUtils.clamp(
        Math.asin(THREE.MathUtils.clamp(offset.y / Math.max(this.freeOrbitRadius, 0.001), -0.98, 0.98)),
        -1.35,
        1.35
      );
      this.freeOrbitYaw = Math.atan2(offset.x, offset.z);
      // Disable mouse fluid force from disturbing particles
      this.simMaterial.uniforms.uInteractForce.value = 0.0;
      this.pointerParallax.set(0, 0);
      this.canvas.style.cursor = 'grab';
    } else {
      // Restore user fluid interaction force
      this.simMaterial.uniforms.uInteractForce.value = this.userInteractForce;
      this.canvas.style.cursor = 'crosshair';
    }
    return this.freeRotateMode;
  }

  setInteractForce(val) {
    this.userInteractForce = val;
    if (!this.freeRotateMode) {
      this.simMaterial.uniforms.uInteractForce.value = val;
    }
  }

  toggleCageAndRings() {
    this.portalGroup.visible = !this.portalGroup.visible;
    return this.portalGroup.visible;
  }

  getStateSnapshot() {
    return {
      paletteKey: this.activePaletteKey,
      volumeIndex: this.currentVolumeIndex,
      cameraPreset: this.activeCameraPreset,
      curl: this.simMaterial.uniforms.uCurlMultiplier.value,
      pull: this.simMaterial.uniforms.uSurfacePullMultiplier.value,
      fluid: this.userInteractForce,
      size: this.renderMaterial.uniforms.uSize.value,
      rotSpeed: this.rotSpeed,
      cageVisible: this.portalGroup.visible,
    };
  }

  applyStateSnapshot(snapshot) {
    if (!snapshot) return;
    if (snapshot.paletteKey) {
      this.setPalette(snapshot.paletteKey);
    }
    if (typeof snapshot.volumeIndex === 'number') {
      this.setVolumeByIndex(snapshot.volumeIndex, false);
    }
    if (snapshot.cameraPreset) {
      this.setCameraPreset(snapshot.cameraPreset);
    }
    if (typeof snapshot.curl === 'number') {
      this.simMaterial.uniforms.uCurlMultiplier.value = snapshot.curl;
    }
    if (typeof snapshot.pull === 'number') {
      this.simMaterial.uniforms.uSurfacePullMultiplier.value = snapshot.pull;
    }
    if (typeof snapshot.fluid === 'number') {
      this.setInteractForce(snapshot.fluid);
    }
    if (typeof snapshot.size === 'number') {
      this.renderMaterial.uniforms.uSize.value = snapshot.size;
    }
    if (typeof snapshot.rotSpeed === 'number') {
      this.rotSpeed = snapshot.rotSpeed;
    }
    if (typeof snapshot.cageVisible === 'boolean') {
      this.portalGroup.visible = snapshot.cageVisible;
    }
  }

  _stepSimulation(dtRatio, timeSec) {
    const prevTarget = this.renderer.getRenderTarget();

    if (this.simFirstStep) {
      this.simMaterial.uniforms.tTexture1.value = this.initPosTex;
      this.simMaterial.uniforms.tTexture2.value = this.initVelTex;
      this.simFirstStep = false;
    } else {
      this.simMaterial.uniforms.tTexture1.value = this.mrtA.textures[0];
      this.simMaterial.uniforms.tTexture2.value = this.mrtA.textures[1];
    }

    this.pointsMesh.updateMatrixWorld();
    this.simMaterial.uniforms.tVel.value = this.fluidSim.texture;
    this.simMaterial.uniforms.uModelMatrix.value.copy(this.pointsMesh.matrixWorld);
    this.simMaterial.uniforms.uViewMatrix.value.copy(this.camera.matrixWorldInverse);
    this.simMaterial.uniforms.uProjMatrix.value.copy(this.camera.projectionMatrix);
    this.simMaterial.uniforms.uTime.value = timeSec;
    this.simMaterial.uniforms.uDtRatio.value = dtRatio;

    this.renderer.setRenderTarget(this.mrtB);
    this.renderer.render(this.simScene, this.simCamera);
    this.renderer.setRenderTarget(prevTarget);

    // Swap MRT ping-pong buffers
    const temp = this.mrtA;
    this.mrtA = this.mrtB;
    this.mrtB = temp;

    // Bind updated position + velocity textures to point-cloud render shader
    this.renderMaterial.uniforms.tTexture1.value = this.mrtA.textures[0];
    this.renderMaterial.uniforms.tTexture2.value = this.mrtA.textures[1];
  }

  _bindEvents() {
    window.addEventListener('resize', () => {
      this.width = window.innerWidth;
      this.height = window.innerHeight;
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.renderer.setSize(this.width, this.height);
      this.renderer.setPixelRatio(this.dpr);
      this.camera.aspect = this.width / Math.max(this.height, 1);
      this.camera.updateProjectionMatrix();
      this.renderMaterial.uniforms.uResolution.value.set(this.width * this.dpr, this.height * this.dpr);
    });

    this.canvas.addEventListener('pointerdown', (e) => {
      if (!this.freeRotateMode) return;
      this.isDraggingOrbit = true;
      this.dragLastX = e.clientX;
      this.dragLastY = e.clientY;
      this.canvas.style.cursor = 'grabbing';
    });

    window.addEventListener('pointerup', () => {
      if (!this.freeRotateMode) return;
      this.isDraggingOrbit = false;
      this.canvas.style.cursor = 'grab';
    });

    window.addEventListener('pointermove', (e) => {
      if (this.freeRotateMode) {
        if (this.isDraggingOrbit) {
          const dx = e.clientX - this.dragLastX;
          const dy = e.clientY - this.dragLastY;
          this.dragLastX = e.clientX;
          this.dragLastY = e.clientY;
          this.freeOrbitYaw -= dx * 0.0075;
          this.freeOrbitPitch = THREE.MathUtils.clamp(this.freeOrbitPitch + dy * 0.0065, -1.35, 1.35);
        }
        return;
      }

      const nx = e.clientX / Math.max(this.width, 1);
      const ny = 1.0 - e.clientY / Math.max(this.height, 1);
      this.fluidSim.onPointerMove(nx, ny);
      this.pointerParallax.set((nx - 0.5) * 2.0, (ny - 0.5) * 2.0);
    });

    this.canvas.addEventListener(
      'wheel',
      (e) => {
        if (!this.freeRotateMode) return;
        e.preventDefault();
        this.freeOrbitRadius = THREE.MathUtils.clamp(this.freeOrbitRadius + e.deltaY * 0.0012, 0.85, 2.35);
      },
      { passive: false }
    );
  }

  start() {
    const tick = (now) => {
      requestAnimationFrame(tick);
      const dtMs = Math.min(now - this.lastTime, 50);
      this.lastTime = now;
      const dtSec = dtMs / 1000;
      const dtRatio = THREE.MathUtils.clamp(dtMs / 16.6667, 0.25, 2.5);
      this.elapsedTime += dtSec;

      const instFps = 1000 / Math.max(dtMs, 1);
      this.fpsSmooth = THREE.MathUtils.lerp(this.fpsSmooth, instFps, 0.08);

      // Opening coalesce choreography (t = 0 -> 2.5s)
      this.introElapsed += dtSec;
      this.uAlpha = THREE.MathUtils.lerp(this.uAlpha, 1.0, 0.045 * dtRatio);
      this.uInitialGlow = THREE.MathUtils.lerp(this.uInitialGlow, 0.0, 0.04 * dtRatio);
      this.uShowNoise = THREE.MathUtils.lerp(this.uShowNoise, 0.0, 0.05 * dtRatio);

      // Decay shape-morph burst noise (`uAdditionalNoise`)
      if (this.morphBurstTimer > 0) {
        this.morphBurstTimer = Math.max(0, this.morphBurstTimer - dtSec);
        const t = this.morphBurstTimer / 0.55;
        this.uAdditionalNoise = t * t;
      } else {
        this.uAdditionalNoise = 0.0;
      }

      this.simMaterial.uniforms.uAdditionalNoise.value = this.uAdditionalNoise;
      this.simMaterial.uniforms.uShowNoise.value = this.uShowNoise;
      this.renderMaterial.uniforms.uAlpha.value = this.uAlpha;
      this.renderMaterial.uniforms.uInitialGlow.value = this.uInitialGlow;

      // Rotate or 3D-sway SDF volume (paused in 360° Free Rotate mode so user can inspect any angle)
      const activeSpec = VOLUME_CATALOG[this.currentVolumeIndex] || VOLUME_CATALOG[0];
      let nextRot = this.simMaterial.uniforms.uRotation.value;
      if (!this.freeRotateMode) {
        if (activeSpec.rotMode === 'sway') {
          nextRot = Math.sin(this.elapsedTime * 0.85 * this.rotSpeed) * 0.42;
        } else {
          nextRot = this.simMaterial.uniforms.uRotation.value - dtMs * 0.00075 * this.rotSpeed;
        }
      }
      this.simMaterial.uniforms.uRotation.value = nextRot;
      this.renderMaterial.uniforms.uRotation.value = nextRot;

      // Camera positioning: 360° spherical orbit in freeRotateMode, or preset + parallax otherwise
      if (this.freeRotateMode) {
        const cosP = Math.cos(this.freeOrbitPitch);
        const sinP = Math.sin(this.freeOrbitPitch);
        const desiredPos = new THREE.Vector3(
          this.cameraBaseTarget.x + this.freeOrbitRadius * cosP * Math.sin(this.freeOrbitYaw),
          this.cameraBaseTarget.y + this.freeOrbitRadius * sinP,
          this.cameraBaseTarget.z + this.freeOrbitRadius * cosP * Math.cos(this.freeOrbitYaw)
        );
        this.cameraCurrentPos.lerp(desiredPos, 0.14 * dtRatio);
        this.cameraCurrentTarget.lerp(this.cameraBaseTarget, 0.14 * dtRatio);
      } else {
        const shakeX = Math.sin(this.elapsedTime * 0.9) * 0.008;
        const shakeY = Math.cos(this.elapsedTime * 0.7) * 0.006;
        const desiredPos = new THREE.Vector3(
          this.cameraBasePos.x + this.pointerParallax.x * 0.08 + shakeX,
          this.cameraBasePos.y + this.pointerParallax.y * 0.04 + shakeY,
          this.cameraBasePos.z
        );
        this.cameraCurrentPos.lerp(desiredPos, 0.045 * dtRatio);
        this.cameraCurrentTarget.lerp(this.cameraBaseTarget, 0.05 * dtRatio);
      }
      this.camera.position.copy(this.cameraCurrentPos);
      this.camera.lookAt(this.cameraCurrentTarget);

      // 1. Update 128x128 2D Navier-Stokes Fluid Velocity Buffer
      this.fluidSim.update(dtRatio, this.width / Math.max(this.height, 1));

      // 2. Step 150,544-Particle WebGL2 MRT Simulation
      this._stepSimulation(dtRatio, this.elapsedTime);

      // 3. Update portal cage, floor rings, and backdrop
      this.backdropMesh.material.uniforms.uTime.value = this.elapsedTime;
      this.cageMat.uniforms.uTime.value = this.elapsedTime;
      this.floorMat.uniforms.uTime.value = this.elapsedTime;
      this.ambientPoints.rotation.y = this.elapsedTime * 0.04;

      // 4. Render Scene
      this.renderer.render(this.scene, this.camera);

      if (this.onTelemetryUpdate) {
        const rotDeg = this.freeRotateMode
          ? Math.round((((this.freeOrbitYaw * 180) / Math.PI) % 360 + 360) % 360)
          : Math.round(((Math.abs(nextRot) * 180) / Math.PI) % 360);
        this.onTelemetryUpdate({
          fps: Math.round(this.fpsSmooth),
          fluidVel: this.freeRotateMode ? 0 : this.fluidSim.velocityMagnitude,
          rotationDeg: rotDeg,
        });
      }
    };

    requestAnimationFrame(tick);
  }
}

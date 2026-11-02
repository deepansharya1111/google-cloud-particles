/**
 * 128x128 2D Screen-Space Navier-Stokes Fluid Solver
 * Computes cursor-driven velocity field `tVel` sampled by the 150,000 3D particles via `uvScreen`.
 */
import * as THREE from 'three';

const BASE_VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const SPLAT_AND_ADVECT_FRAGMENT = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uVelocity;
  uniform vec2 uPoint;
  uniform vec2 uDelta;
  uniform float uRadius;
  uniform float uDissipation;
  uniform float uAspect;
  uniform float uDt;

  void main() {
    // Backtrace Semi-Lagrangian advection
    vec2 prevVel = texture2D(uVelocity, vUv).xy;
    vec2 coord = vUv - prevVel * uDt * 0.012;
    vec2 advected = texture2D(uVelocity, clamp(coord, 0.0, 1.0)).xy * uDissipation;

    // Cursor Gaussian splat impulse
    vec2 p = vUv - uPoint;
    p.x *= uAspect;
    float d2 = dot(p, p);
    float splat = exp(-d2 / max(uRadius * uRadius, 0.0001));
    vec2 result = advected + uDelta * splat * 35.0;

    // Clamp to avoid numerical blowup
    float len = length(result);
    if (len > 2.5) {
      result = (result / len) * 2.5;
    }

    gl_FragColor = vec4(result, 0.0, 1.0);
  }
`;

const CURL_CONFINE_FRAGMENT = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uVelocity;
  uniform vec2 uTexelSize;

  float curlAt(vec2 uv) {
    float L = texture2D(uVelocity, uv - vec2(uTexelSize.x, 0.0)).y;
    float R = texture2D(uVelocity, uv + vec2(uTexelSize.x, 0.0)).y;
    float T = texture2D(uVelocity, uv + vec2(0.0, uTexelSize.y)).x;
    float B = texture2D(uVelocity, uv - vec2(0.0, uTexelSize.y)).x;
    return 0.5 * (R - L - T + B);
  }

  void main() {
    vec2 vel = texture2D(uVelocity, vUv).xy;
    float cL = abs(curlAt(vUv - vec2(uTexelSize.x, 0.0)));
    float cR = abs(curlAt(vUv + vec2(uTexelSize.x, 0.0)));
    float cT = abs(curlAt(vUv + vec2(0.0, uTexelSize.y)));
    float cB = abs(curlAt(vUv - vec2(0.0, uTexelSize.y)));
    float cC = curlAt(vUv);

    vec2 force = 0.5 * vec2(cT - cB, cL - cR);
    force /= max(length(force), 1e-4);
    force *= cC * 0.08;

    // Isotropic pressure-like smoothing from 4 neighbors
    vec2 nL = texture2D(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).xy;
    vec2 nR = texture2D(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).xy;
    vec2 nT = texture2D(uVelocity, vUv + vec2(0.0, uTexelSize.y)).xy;
    vec2 nB = texture2D(uVelocity, vUv - vec2(0.0, uTexelSize.y)).xy;
    vec2 diffused = mix(vel, 0.25 * (nL + nR + nT + nB), 0.18);

    gl_FragColor = vec4(diffused + force, 0.0, 1.0);
  }
`;

export class FluidSim {
  constructor(renderer, simRes = 128) {
    this.renderer = renderer;
    this.simRes = simRes;

    const rtOpts = {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
    };

    this.rtA = new THREE.WebGLRenderTarget(simRes, simRes, rtOpts);
    this.rtB = new THREE.WebGLRenderTarget(simRes, simRes, rtOpts);

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const quad = new THREE.PlaneGeometry(2, 2);

    this.splatMat = new THREE.ShaderMaterial({
      vertexShader: BASE_VERTEX,
      fragmentShader: SPLAT_AND_ADVECT_FRAGMENT,
      uniforms: {
        uVelocity: { value: null },
        uPoint: { value: new THREE.Vector2(0.5, 0.5) },
        uDelta: { value: new THREE.Vector2(0, 0) },
        uRadius: { value: 0.14 },
        uDissipation: { value: 0.955 },
        uAspect: { value: window.innerWidth / Math.max(window.innerHeight, 1) },
        uDt: { value: 1.0 },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.curlMat = new THREE.ShaderMaterial({
      vertexShader: BASE_VERTEX,
      fragmentShader: CURL_CONFINE_FRAGMENT,
      uniforms: {
        uVelocity: { value: null },
        uTexelSize: { value: new THREE.Vector2(1 / simRes, 1 / simRes) },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.mesh = new THREE.Mesh(quad, this.splatMat);
    this.scene.add(this.mesh);

    this.pointer = new THREE.Vector2(0.5, 0.5);
    this.prevPointer = new THREE.Vector2(0.5, 0.5);
    this.pointerDelta = new THREE.Vector2(0, 0);
    this.hasMoved = false;
    this.velocityMagnitude = 0;
  }

  onPointerMove(normX, normY) {
    if (!this.hasMoved) {
      this.prevPointer.set(normX, normY);
      this.pointer.set(normX, normY);
      this.hasMoved = true;
      return;
    }
    this.pointer.set(normX, normY);
    const dx = (this.pointer.x - this.prevPointer.x) * 18.0;
    const dy = (this.pointer.y - this.prevPointer.y) * 18.0;
    this.pointerDelta.x = THREE.MathUtils.clamp(dx, -1.2, 1.2);
    this.pointerDelta.y = THREE.MathUtils.clamp(dy, -1.2, 1.2);
    this.prevPointer.copy(this.pointer);
  }

  injectImpulse(normX, normY, vx, vy) {
    this.pointer.set(normX, normY);
    this.pointerDelta.set(vx, vy);
  }

  update(dtRatio, aspect) {
    const prevTarget = this.renderer.getRenderTarget();

    // Pass 1: Advect + Cursor Splat into rtB
    this.mesh.material = this.splatMat;
    this.splatMat.uniforms.uVelocity.value = this.rtA.texture;
    this.splatMat.uniforms.uPoint.value.copy(this.pointer);
    this.splatMat.uniforms.uDelta.value.copy(this.pointerDelta);
    this.splatMat.uniforms.uAspect.value = aspect;
    this.splatMat.uniforms.uDt.value = dtRatio;

    this.renderer.setRenderTarget(this.rtB);
    this.renderer.render(this.scene, this.camera);

    // Pass 2: Curl vorticity confinement + neighbor diffusion back into rtA
    this.mesh.material = this.curlMat;
    this.curlMat.uniforms.uVelocity.value = this.rtB.texture;

    this.renderer.setRenderTarget(this.rtA);
    this.renderer.render(this.scene, this.camera);

    this.renderer.setRenderTarget(prevTarget);

    const instSpeed = this.pointerDelta.length();
    this.velocityMagnitude = THREE.MathUtils.lerp(this.velocityMagnitude, instSpeed, 0.15);
    this.pointerDelta.multiplyScalar(0.65);
  }

  get texture() {
    return this.rtA.texture;
  }
}

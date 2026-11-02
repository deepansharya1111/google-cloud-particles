import { ParticlePortalApp, VOLUME_CATALOG } from './particle-portal.js';

const STORAGE_KEY = 'google_cloud_cryo_saved_state_v1';

document.addEventListener('DOMContentLoaded', async () => {
  const canvas = document.getElementById('portal-canvas');
  const elFps = document.getElementById('hud-fps');
  const elFluid = document.getElementById('hud-fluid');
  const elRot = document.getElementById('hud-rot');
  const elTemp = document.getElementById('hud-temp');
  const elSpecId = document.getElementById('spec-id');
  const elSpecName = document.getElementById('spec-name');
  const elSpecSub = document.getElementById('spec-sub');
  const elSpecVoxels = document.getElementById('spec-voxels');
  const elBottomTitle = document.getElementById('bottom-active-title');
  const elBottomIndex = document.getElementById('bottom-active-index');
  const elBottomHint = document.getElementById('hud-bottom-hint');

  const rightDeck = document.getElementById('right-physics-deck');
  const freeRotateBtn = document.getElementById('btn-free-rotate');
  const freeRotateNote = document.getElementById('free-rotate-note');
  const cageBtn = document.getElementById('btn-toggle-cage');
  const stateStatusEl = document.getElementById('state-status');

  let frameCounter = 0;
  const app = new ParticlePortalApp(canvas, ({ fps, fluidVel, rotationDeg }) => {
    frameCounter++;
    if (frameCounter % 4 !== 0) return;
    if (elFps) elFps.textContent = `${fps} FPS`;
    if (elFluid) elFluid.textContent = fluidVel.toFixed(3);
    if (elRot) elRot.textContent = `${String(rotationDeg).padStart(3, '0')}°`;
    if (elTemp) {
      const baseTemp = -0.72 - fluidVel * 2.4;
      elTemp.textContent = `${baseTemp.toFixed(2)}°C`;
    }
  });

  window.__portalApp = app;

  const updateActiveVolumeUI = (item) => {
    if (!item) return;
    if (elSpecId) elSpecId.textContent = `////// SPECIMEN_${item.index}`;
    if (elSpecName) elSpecName.textContent = item.name;
    if (elSpecSub) elSpecSub.textContent = item.subtitle;
    if (elSpecVoxels) elSpecVoxels.textContent = item.voxels;
    if (elBottomTitle) elBottomTitle.textContent = item.name;
    if (elBottomIndex) elBottomIndex.textContent = `VOLUME_${item.index} // 64³ SDF`;

    document.querySelectorAll('[data-volume-index]').forEach((btn) => {
      const idx = Number(btn.getAttribute('data-volume-index'));
      btn.classList.toggle('is-active', idx === app.currentVolumeIndex);
    });
  };

  const syncDeckUIFromSnapshot = (snap) => {
    if (!snap) return;

    // Palette buttons
    document.querySelectorAll('[data-palette]').forEach((btn) => {
      btn.classList.toggle('is-active', btn.getAttribute('data-palette') === snap.paletteKey);
    });

    // Camera preset buttons
    document.querySelectorAll('[data-camera]').forEach((btn) => {
      btn.classList.toggle('is-active', !app.freeRotateMode && btn.getAttribute('data-camera') === snap.cameraPreset);
    });

    // Volume UI
    const volItem = VOLUME_CATALOG[snap.volumeIndex] || VOLUME_CATALOG[0];
    updateActiveVolumeUI(volItem);

    // Sliders & readouts
    const setSliderVal = (id, valId, val) => {
      const input = document.getElementById(id);
      const valEl = document.getElementById(valId);
      if (input && typeof val === 'number') {
        input.value = String(val);
      }
      if (valEl && typeof val === 'number') {
        valEl.textContent = val.toFixed(2);
      }
    };

    setSliderVal('slider-curl', 'val-curl', snap.curl);
    setSliderVal('slider-pull', 'val-pull', snap.pull);
    setSliderVal('slider-fluid', 'val-fluid', snap.fluid);
    setSliderVal('slider-size', 'val-size', snap.size);
    setSliderVal('slider-rot', 'val-rot', snap.rotSpeed);

    if (cageBtn && typeof snap.cageVisible === 'boolean') {
      cageBtn.textContent = snap.cageVisible ? '[ CAGE & RINGS: ON ]' : '[ CAGE & RINGS: OFF ]';
    }
  };

  const setRightDeckDisabled = (disabled) => {
    if (!rightDeck) return;
    rightDeck.classList.toggle('is-disabled-deck', disabled);
    rightDeck.querySelectorAll('button, input').forEach((ctrl) => {
      ctrl.disabled = disabled;
    });
  };

  const syncFreeRotateUI = (enabled) => {
    if (freeRotateBtn) {
      freeRotateBtn.classList.toggle('is-active', enabled);
      const labelSpan = freeRotateBtn.querySelector('span:first-child');
      if (labelSpan) {
        labelSpan.textContent = enabled ? '[ 360° FREE ROTATE: ON ]' : '[ 360° FREE ROTATE: OFF ]';
      }
    }
    if (freeRotateNote) {
      freeRotateNote.textContent = enabled
        ? 'ACTIVE: Drag anywhere on canvas to orbit 360° • Scroll to zoom • Physics & fluid locked.'
        : 'Locks physics deck & cursor fluid so you can drag 360° around the sculpture.';
    }
    if (elBottomHint) {
      elBottomHint.textContent = enabled
        ? '360° FREE ROTATION ACTIVE • DRAG CANVAS TO ORBIT 360° • SCROLL WHEEL TO ZOOM • CURSOR FLUID DISABLED'
        : 'SWIPE CURSOR ACROSS THE PORTAL TO DISTURB THE 128² NAVIER-STOKES FLUID FIELD • CLICK VOLUMES TO MORPH';
    }
    document.querySelectorAll('[data-camera]').forEach((btn) => {
      btn.classList.toggle('is-active', !enabled && btn.getAttribute('data-camera') === app.activeCameraPreset);
    });
    setRightDeckDisabled(enabled);
  };

  // Load 3D SDF volume textures and start 60fps loop
  await app.loadAllVolumes();
  updateActiveVolumeUI(VOLUME_CATALOG[0]);

  // Capture initial default state snapshot (and load any previously saved snapshot from localStorage)
  const defaultSnapshot = app.getStateSnapshot();
  let savedSnapshot = { ...defaultSnapshot };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        savedSnapshot = { ...defaultSnapshot, ...parsed };
        if (stateStatusEl) stateStatusEl.textContent = 'SAVED READY';
      }
    }
  } catch {
    // Ignore storage errors in private browsing
  }

  app.start();

  // Bind bottom volume buttons
  document.querySelectorAll('[data-volume-index]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.getAttribute('data-volume-index'));
      const item = app.setVolumeByIndex(idx);
      updateActiveVolumeUI(item);
    });
  });

  // Bind Prev / Next carousel arrows
  document.getElementById('btn-prev-vol')?.addEventListener('click', () => {
    const item = app.setVolumeByIndex(app.currentVolumeIndex - 1);
    updateActiveVolumeUI(item);
  });
  document.getElementById('btn-next-vol')?.addEventListener('click', () => {
    const item = app.setVolumeByIndex(app.currentVolumeIndex + 1);
    updateActiveVolumeUI(item);
  });

  // Bind Palette Mode buttons
  document.querySelectorAll('[data-palette]').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (app.freeRotateMode) return;
      const key = btn.getAttribute('data-palette');
      app.setPalette(key);
      document.querySelectorAll('[data-palette]').forEach((b) => {
        b.classList.toggle('is-active', b === btn);
      });
    });
  });

  // Bind Camera Preset buttons (clicking a preset also exits 360° Free Rotate mode)
  document.querySelectorAll('[data-camera]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const key = btn.getAttribute('data-camera');
      app.setCameraPreset(key);
      syncFreeRotateUI(false);
    });
  });

  // Bind 360° Free Rotation Mode toggle button
  freeRotateBtn?.addEventListener('click', () => {
    const nextState = app.setFreeRotateMode(!app.freeRotateMode);
    syncFreeRotateUI(nextState);
  });

  // Bind Physics Sliders
  const bindSlider = (id, valId, onInput) => {
    const input = document.getElementById(id);
    const valEl = document.getElementById(valId);
    if (!input) return;
    input.addEventListener('input', () => {
      if (app.freeRotateMode) return;
      const v = parseFloat(input.value);
      if (valEl) valEl.textContent = v.toFixed(2);
      onInput(v);
    });
  };

  bindSlider('slider-curl', 'val-curl', (v) => {
    app.simMaterial.uniforms.uCurlMultiplier.value = v;
  });
  bindSlider('slider-pull', 'val-pull', (v) => {
    app.simMaterial.uniforms.uSurfacePullMultiplier.value = v;
  });
  bindSlider('slider-fluid', 'val-fluid', (v) => {
    app.setInteractForce(v);
  });
  bindSlider('slider-size', 'val-size', (v) => {
    app.renderMaterial.uniforms.uSize.value = v;
  });
  bindSlider('slider-rot', 'val-rot', (v) => {
    app.rotSpeed = v;
  });

  // Bind Action Buttons
  document.getElementById('btn-shockwave')?.addEventListener('click', () => {
    if (app.freeRotateMode) return;
    app.triggerShockwave(1.15);
  });

  cageBtn?.addEventListener('click', () => {
    if (app.freeRotateMode) return;
    const visible = app.toggleCageAndRings();
    cageBtn.textContent = visible ? '[ CAGE & RINGS: ON ]' : '[ CAGE & RINGS: OFF ]';
  });

  // Bind Save State & Revert Saved Buttons
  document.getElementById('btn-save-state')?.addEventListener('click', () => {
    if (app.freeRotateMode) return;
    savedSnapshot = app.getStateSnapshot();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(savedSnapshot));
    } catch {
      // Ignore storage errors
    }
    if (stateStatusEl) {
      const now = new Date();
      const ts = now.toTimeString().slice(0, 8);
      stateStatusEl.textContent = `SAVED @ ${ts}`;
      stateStatusEl.classList.add('is-flash');
      setTimeout(() => stateStatusEl.classList.remove('is-flash'), 900);
    }
  });

  document.getElementById('btn-revert-state')?.addEventListener('click', () => {
    if (app.freeRotateMode) return;
    const targetSnap = savedSnapshot || defaultSnapshot;
    app.applyStateSnapshot(targetSnap);
    syncDeckUIFromSnapshot(targetSnap);
    if (stateStatusEl) {
      stateStatusEl.textContent = 'STATE REVERTED';
      stateStatusEl.classList.add('is-flash');
      setTimeout(() => stateStatusEl.classList.remove('is-flash'), 900);
    }
  });
});

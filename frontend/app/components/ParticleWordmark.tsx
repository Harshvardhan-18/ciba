"use client";

import { useEffect, useRef, useCallback } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Particle {
  x: number;
  y: number;
  /** Final letterform target x */
  tx: number;
  /** Final letterform target y */
  ty: number;
  /** Origin x (phone position with jitter) */
  ox: number;
  /** Origin y (phone position with jitter) */
  oy: number;
  /** Convergence waypoint x (with small jitter) */
  wx: number;
  /** Convergence waypoint y (with small jitter) */
  wy: number;
  /** Bezier control point x for phase A (phone → convergence) */
  cpx: number;
  /** Bezier control point y for phase A (phone → convergence) */
  cpy: number;
  /** Fraction of total duration spent in phase A (phone → convergence) */
  phaseASplit: number;
  delay: number;
  duration: number;
  alpha: number;
  phase: number;
  freq: number;
  r: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORD = "BRAND";

/**
 * Convergence point as a fraction of the hero image's native dimensions.
 * Measured from bg.png (1456×816): the bright glow orb is at ~50% x, ~29.5% y.
 */
const CONV_X_FRAC = 0.50;
const CONV_Y_FRAC = 0.295;

/**
 * Three phone origin points, measured from bg.png (1456×816).
 * Each value is the bright phone-screen glow where that figure's light stream begins.
 *
 * Left phone:   pixel ~(360, 560)  → fractions (0.247, 0.686)
 * Center phone: pixel ~(700, 590)  → fractions (0.481, 0.723)
 * Right phone:  pixel ~(1040, 560) → fractions (0.714, 0.686)
 */
const ORIGIN_LEFT   = { xFrac: 0.300, yFrac: 0.850 };
const ORIGIN_CENTER = { xFrac: 0.500, yFrac: 0.850 };
const ORIGIN_RIGHT  = { xFrac: 0.690, yFrac: 0.830 };

const ORIGINS = [ORIGIN_LEFT, ORIGIN_CENTER, ORIGIN_RIGHT] as const;

const IMG_W = 1456;
const IMG_H = 816;

/** Per-particle jitter radius around each phone origin (px on canvas) */
const SCATTER_RADIUS = 8;
/** Jitter radius around convergence waypoint so particles don't all hit a single pixel */
const CONV_JITTER = 8;

/** Animation timing — longer to accommodate the two-leg journey */
const ANIMATION_TOTAL_MS = 3800;
const MIN_DELAY_MS = 0;
const MAX_DELAY_MS = 1000;
const MIN_DUR_MS = 2200;
const MAX_DUR_MS = 3200;

const IDLE_START_MS = ANIMATION_TOTAL_MS + 400;

/** Extra pixel shift to push the particle convergence point lower down (increase to push convergence point further down) */
const CONVERGENCE_OFFSET_Y_PX = 100;

/** Vertical offset in pixels to shift only the rendered glow effect downwards */
const GLOW_OFFSET_Y = 50;

const SAMPLE_STEP = 4;

// ---------------------------------------------------------------------------
// Easing & math helpers
// ---------------------------------------------------------------------------

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function rand(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

/** Evaluate a quadratic bezier at parameter t. */
function quadBezier(p0: number, cp: number, p1: number, t: number): number {
  const mt = 1 - t;
  return mt * mt * p0 + 2 * mt * t * cp + t * t * p1;
}

// ---------------------------------------------------------------------------
// Image point mapping
// Replicates CSS object-fit:cover + object-position:center 20%
// ---------------------------------------------------------------------------

function mapImagePoint(
  containerW: number,
  containerH: number,
  xFrac: number,
  yFrac: number
): { x: number; y: number } {
  const scaleX = containerW / IMG_W;
  const scaleY = containerH / IMG_H;
  const scale = Math.max(scaleX, scaleY);

  const renderedW = IMG_W * scale;
  const renderedH = IMG_H * scale;

  const offsetX = (containerW - renderedW) * 0.5;
  // 0.20 matches object-position: center 20% in CSS
  const offsetY = (containerH - renderedH) * 0.20;

  return {
    x: offsetX + xFrac * renderedW,
    y: offsetY + yFrac * renderedH,
  };
}

// ---------------------------------------------------------------------------
// Offscreen text sampler
// ---------------------------------------------------------------------------

function sampleTextPoints(
  word: string,
  targetFontPx: number,
  fontFamily: string
): Array<{ x: number; y: number; w: number; h: number }> {
  const offscreen = document.createElement("canvas");
  const ctx = offscreen.getContext("2d", { willReadFrequently: true });
  if (!ctx) return [];

  const fontStr = `700 ${targetFontPx}px ${fontFamily}`;
  ctx.font = fontStr;
  const metrics = ctx.measureText(word);
  const textW = Math.ceil(metrics.width) + 20;
  const textH = Math.ceil(targetFontPx * 1.4);

  offscreen.width = textW;
  offscreen.height = textH;

  ctx.clearRect(0, 0, textW, textH);
  ctx.fillStyle = "#ffffff";
  ctx.font = fontStr;
  ctx.textBaseline = "alphabetic";
  ctx.fillText(word, 10, targetFontPx * 0.88);

  const imageData = ctx.getImageData(0, 0, textW, textH);
  const { data } = imageData;
  const points: Array<{ x: number; y: number; w: number; h: number }> = [];

  for (let y = 0; y < textH; y += SAMPLE_STEP) {
    for (let x = 0; x < textW; x += SAMPLE_STEP) {
      const alpha = data[(y * textW + x) * 4 + 3];
      if (alpha > 128) {
        points.push({ x, y, w: textW, h: textH });
      }
    }
  }

  return points;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface ParticleWordmarkProps {
  word?: string;
  className?: string;
}

export default function ParticleWordmark({
  word = WORD,
  className = "",
}: ParticleWordmarkProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const taglineRef = useRef<HTMLSpanElement>(null);
  const rafRef = useRef<number | null>(null);
  const animStartRef = useRef<number | null>(null);
  const hasStartedRef = useRef(false);

  const buildAndAnimate = useCallback(() => {
    if (hasStartedRef.current) return;
    hasStartedRef.current = true;

    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return;

    const hero = wrapper.closest<HTMLElement>(".lp-hero") ?? wrapper;
    const heroRect = hero.getBoundingClientRect();

    const dpr = Math.min(window.devicePixelRatio || 1, 2); // cap at 2x for perf
    const W = heroRect.width;
    const H = heroRect.height;

    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    // Map convergence point to canvas coords (with optional pixel offset)
    const conv = mapImagePoint(W, H, CONV_X_FRAC, CONV_Y_FRAC);
    const cx = conv.x;
    const cy = conv.y + CONVERGENCE_OFFSET_Y_PX;

    // Map the three phone origins to canvas coords
    const phoneOrigins = ORIGINS.map((o) => mapImagePoint(W, H, o.xFrac, o.yFrac));

    // Text assembles BELOW the convergence glow so there is headroom
    // above "Elevate Your".
    const textTargetX = cx;
    const textTargetY = cy + H * 0.08;

    // Scale font size relative to hero width; clamped 32–80px
    const targetFontPx = Math.max(32, Math.min(80, Math.round(W * 0.072)));
    const fontFamily = `"Inter", "Segoe UI", system-ui, sans-serif`;

    const rawPoints = sampleTextPoints(word, targetFontPx, fontFamily);
    if (rawPoints.length === 0) return;

    const textW = rawPoints[0].w;
    const textH = rawPoints[0].h;

    // Centre the BRAND block on the text target point
    const blockOffsetX = textTargetX - textW / 2;
    const blockOffsetY = textTargetY - textH * 0.55;

    // --- Position "Elevate Your" tagline above the BRAND block ---
    if (taglineRef.current) {
      const tl = taglineRef.current;
      tl.style.fontSize = `${targetFontPx}px`;
      // Sit just above BRAND block top with 1.35× line-height gap
      tl.style.top = `${blockOffsetY - targetFontPx * 1.35}px`;
      tl.style.left = `${textTargetX}px`;
      tl.style.transform = "translateX(-50%)";
      // Trigger CSS fade-in
      requestAnimationFrame(() => tl.classList.add("is-visible"));
    }

    // ---------------------------------------------------------------
    // Build particles with three-origin assignment + bezier control points
    // ---------------------------------------------------------------
    const particles: Particle[] = rawPoints.map((pt, i) => {
      const tx = blockOffsetX + pt.x;
      const ty = blockOffsetY + pt.y;

      // Round-robin assignment to the three phone origins
      const originIdx = i % 3;
      const phone = phoneOrigins[originIdx];

      // Small jitter around the phone origin
      const phoneAngle = Math.random() * Math.PI * 2;
      const phoneDist = rand(0, SCATTER_RADIUS);
      const ox = phone.x + Math.cos(phoneAngle) * phoneDist;
      const oy = phone.y + Math.sin(phoneAngle) * phoneDist;

      // Small jitter around the convergence waypoint
      const convAngle = Math.random() * Math.PI * 2;
      const convDist = rand(0, CONV_JITTER);
      const wx = cx + Math.cos(convAngle) * convDist;
      const wy = cy + Math.sin(convAngle) * convDist;

      // Compute bezier control point for phase A (phone → convergence).
      // The control point is offset perpendicular to the straight line
      // from origin to convergence. Outer phones curve more than center.
      const midX = (ox + wx) / 2;
      const midY = (oy + wy) / 2;
      // Direction vector from origin to convergence
      const dx = wx - ox;
      const dy = wy - oy;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      // Perpendicular unit vector (rotate 90° CW)
      const perpX = -dy / len;
      const perpY = dx / len;
      // How much to bow the curve: proportional to horizontal offset from
      // center. Left phone bows right (+), right phone bows left (-),
      // center phone is nearly straight.
      let bowSign: number;
      if (originIdx === 0) bowSign = 1;        // left → bow right
      else if (originIdx === 2) bowSign = -1;   // right → bow left
      else bowSign = (Math.random() - 0.5) * 0.3; // center → tiny random
      const bowMagnitude = len * 0.22 * Math.abs(bowSign === 0 ? 0.1 : bowSign);
      const cpx = midX + perpX * bowMagnitude * Math.sign(bowSign || 1);
      const cpy = midY + perpY * bowMagnitude * Math.sign(bowSign || 1);

      // Phase A takes 55–65% of total duration per particle
      const phaseASplit = rand(0.55, 0.65);

      return {
        x: ox,
        y: oy,
        tx,
        ty,
        ox,
        oy,
        wx,
        wy,
        cpx,
        cpy,
        phaseASplit,
        delay: rand(MIN_DELAY_MS, MAX_DELAY_MS),
        duration: rand(MIN_DUR_MS, MAX_DUR_MS),
        alpha: rand(0.78, 1.0),
        phase: Math.random() * Math.PI * 2,
        freq: rand(0.3, 0.9),
        r: rand(1.0, 1.85),
      };
    });

    // ---------------------------------------------------------------
    // Animation loop
    // ---------------------------------------------------------------
    const draw = (now: number) => {
      if (!animStartRef.current) animStartRef.current = now;
      const elapsed = now - animStartRef.current;
      const isIdle = elapsed > IDLE_START_MS;

      ctx.clearRect(0, 0, W, H);

      // Convergence glow — stays bright while particles pass through,
      // then fades after most particles have entered phase B.
      if (!isIdle) {
        // The glow peaks around the time most particles reach the
        // convergence (roughly 55–65% of their travel which starts
        // at delay 0–1000ms with duration 2200–3200ms). We keep it
        // near-full until ~2800ms, then fade over the remainder.
        const glowHoldEnd = 2800;
        const glowFadeEnd = ANIMATION_TOTAL_MS;
        let glowAlpha: number;
        if (elapsed < glowHoldEnd) {
          // Ramp up quickly, hold
          glowAlpha = Math.min(1, elapsed / 600);
        } else {
          // Fade out
          const fadeProg = Math.min(1, (elapsed - glowHoldEnd) / (glowFadeEnd - glowHoldEnd));
          glowAlpha = 1 - easeOutCubic(fadeProg);
        }
        if (glowAlpha > 0.01) {
          const glowY = cy + GLOW_OFFSET_Y;
          const grd = ctx.createRadialGradient(cx, glowY, 0, cx, glowY, SCATTER_RADIUS * 3);
          grd.addColorStop(0, `rgba(255, 245, 210, ${glowAlpha * 0.6})`);
          grd.addColorStop(0.35, `rgba(220, 200, 140, ${glowAlpha * 0.25})`);
          grd.addColorStop(1, "rgba(0,0,0,0)");
          ctx.fillStyle = grd;
          ctx.beginPath();
          ctx.arc(cx, glowY, SCATTER_RADIUS * 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Draw particles
      for (const p of particles) {
        const rawT = Math.max(0, (elapsed - p.delay) / p.duration);
        const progress = Math.min(1, rawT);

        if (progress <= 0) {
          // Not started yet — stay invisible at origin
          p.x = p.ox;
          p.y = p.oy;
          ctx.globalAlpha = 0;
          continue;
        }

        let alpha: number;

        if (progress < p.phaseASplit) {
          // --- PHASE A: phone origin → convergence (via bezier) ---
          const phaseT = progress / p.phaseASplit;
          const eased = easeInOutCubic(phaseT);
          p.x = quadBezier(p.ox, p.cpx, p.wx, eased);
          p.y = quadBezier(p.oy, p.cpy, p.wy, eased);
          // Fade in during first 30% of phase A
          alpha = p.alpha * Math.min(1, phaseT / 0.3);
        } else {
          // --- PHASE B: convergence → final text target (straight lerp) ---
          const phaseT = (progress - p.phaseASplit) / (1 - p.phaseASplit);
          const eased = easeOutCubic(phaseT);
          p.x = lerp(p.wx, p.tx, eased);
          p.y = lerp(p.wy, p.ty, eased);
          alpha = p.alpha;
        }

        if (isIdle) {
          // Subtle shimmer once settled
          const shimmerDepth = 0.1;
          alpha = p.alpha * (1 - shimmerDepth + shimmerDepth * Math.sin(now * 0.001 * p.freq + p.phase));
        }

        ctx.globalAlpha = alpha;
        ctx.fillStyle = "#f6f3ea";
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.globalAlpha = 1;

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
  }, [word]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (mq.matches) {
      // Motion suppressed — accessible text is already present; skip canvas
      return;
    }

    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return;

    const hero = wrapper.closest<HTMLElement>(".lp-hero") ?? document.body;
    const img = hero.querySelector<HTMLImageElement>("img.lp-hero-img");

    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let started = false;

    const start = () => {
      if (started) return;
      started = true;
      if (timeoutId) clearTimeout(timeoutId);
      requestAnimationFrame(() => buildAndAnimate());
    };

    if (img) {
      if (img.complete && img.naturalWidth > 0) {
        // Image already loaded (cached / fast)
        timeoutId = setTimeout(start, 80);
      } else {
        img.addEventListener("load", start, { once: true });
        img.addEventListener("error", start, { once: true });
        // Fallback: start 2 s after mount regardless
        timeoutId = setTimeout(start, 2000);
      }
    } else {
      timeoutId = setTimeout(start, 300);
    }

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [buildAndAnimate]);

  return (
    <div
      ref={wrapperRef}
      className={`particle-wordmark-wrapper ${className}`.trim()}
      aria-hidden="true"
    >
      {/* Real text — always in the DOM for a11y and crawlers */}
      <span className="particle-wordmark-a11y">Elevate Your {word}</span>

      {/* "Elevate Your" — positioned by JS above the BRAND particle block */}
      <span ref={taglineRef} className="particle-wordmark-tagline">
        Elevate Your
      </span>

      {/* Canvas — decorative overlay, pointer-events:none keeps hero interactive */}
      <canvas
        ref={canvasRef}
        className="particle-wordmark-canvas"
        role="img"
        aria-label={`${word} wordmark assembled from particles`}
      />
    </div>
  );
}

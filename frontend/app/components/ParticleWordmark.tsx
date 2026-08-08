"use client";

import { useEffect, useRef, useCallback } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Particle {
  x: number;
  y: number;
  tx: number;
  ty: number;
  ox: number;
  oy: number;
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
 * The image (1456×816) has the glow at ~50% x, ~30% y.
 */
const CONV_X_FRAC = 0.50;
const CONV_Y_FRAC = 0.295;

const IMG_W = 1456;
const IMG_H = 816;

const SCATTER_RADIUS = 44;

const ANIMATION_TOTAL_MS = 2200;
const MIN_DELAY_MS = 0;
const MAX_DELAY_MS = 900;
const MIN_DUR_MS = 900;
const MAX_DUR_MS = 1600;

const IDLE_START_MS = ANIMATION_TOTAL_MS + 300;

const SAMPLE_STEP = 4;

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function rand(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

// ---------------------------------------------------------------------------
// Convergence point calculation
// Replicates CSS object-fit:cover + object-position:center 38%
// ---------------------------------------------------------------------------
function calcConvergence(containerW: number, containerH: number): { cx: number; cy: number } {
  const scaleX = containerW / IMG_W;
  const scaleY = containerH / IMG_H;
  const scale = Math.max(scaleX, scaleY);

  const renderedW = IMG_W * scale;
  const renderedH = IMG_H * scale;

  const offsetX = (containerW - renderedW) * 0.5;
  // 0.20 matches object-position: center 20% in CSS
  const offsetY = (containerH - renderedH) * 0.20;

  const cx = offsetX + CONV_X_FRAC * renderedW;
  const cy = offsetY + CONV_Y_FRAC * renderedH;

  return { cx, cy };
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

    const { cx, cy } = calcConvergence(W, H);

    // Text assembles BELOW the actual convergence glow so there is clear
    // headroom above "Elevate Your". Particles still scatter FROM the glow
    // (cx, cy), then travel down to the text target — reads as light
    // streaming down and crystallising into words.
    const textTargetX = cx;
    const textTargetY = cy + H * 0.24;

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

    const particles: Particle[] = rawPoints.map((pt) => {
      const tx = blockOffsetX + pt.x;
      const ty = blockOffsetY + pt.y;

      // Scatter origin: the actual glow convergence point
      const angle = Math.random() * Math.PI * 2;
      const dist = rand(0, SCATTER_RADIUS);
      const ox = cx + Math.cos(angle) * dist;
      const oy = cy + Math.sin(angle) * dist;

      return {
        x: ox,
        y: oy,
        tx,
        ty,
        ox,
        oy,
        delay: rand(MIN_DELAY_MS, MAX_DELAY_MS),
        duration: rand(MIN_DUR_MS, MAX_DUR_MS),
        alpha: rand(0.78, 1.0),
        phase: Math.random() * Math.PI * 2,
        freq: rand(0.3, 0.9),
        r: rand(1.0, 1.85),
      };
    });

    const draw = (now: number) => {
      if (!animStartRef.current) animStartRef.current = now;
      const elapsed = now - animStartRef.current;
      const isIdle = elapsed > IDLE_START_MS;

      ctx.clearRect(0, 0, W, H);

      // Convergence glow — fades out as particles travel away
      if (!isIdle) {
        const glowFade = 1 - easeOutCubic(Math.min(1, elapsed / (ANIMATION_TOTAL_MS * 0.75)));
        if (glowFade > 0.01) {
          const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, SCATTER_RADIUS * 2.2);
          grd.addColorStop(0, `rgba(255, 245, 210, ${glowFade * 0.55})`);
          grd.addColorStop(0.45, `rgba(220, 190, 120, ${glowFade * 0.2})`);
          grd.addColorStop(1, "rgba(0,0,0,0)");
          ctx.fillStyle = grd;
          ctx.beginPath();
          ctx.arc(cx, cy, SCATTER_RADIUS * 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Draw particles
      for (const p of particles) {
        const t = Math.max(0, (elapsed - p.delay) / p.duration);
        const progress = Math.min(1, t);
        const ease = easeOutCubic(progress);

        p.x = lerp(p.ox, p.tx, ease);
        p.y = lerp(p.oy, p.ty, ease);

        let alpha: number;
        if (isIdle) {
          const shimmerDepth = 0.1;
          alpha = p.alpha * (1 - shimmerDepth + shimmerDepth * Math.sin(now * 0.001 * p.freq + p.phase));
        } else {
          alpha = p.alpha * Math.max(0, ease);
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

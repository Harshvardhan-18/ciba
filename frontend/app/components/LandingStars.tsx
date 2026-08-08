"use client";

import { useEffect, useRef } from "react";

// ---------------------------------------------------------------------------
// Deterministic PRNG — same star layout on every render/mount
// (Linear Congruential Generator, 32-bit)
// ---------------------------------------------------------------------------
function makeLCG(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = Math.imul(s, 1664525) + 1013904223;
    return (s >>> 0) / 0xffffffff;
  };
}

const rng = makeLCG(0xc0ffee42);

interface Star {
  x: number; // 0–1 fractional position
  y: number; // 0–1 fractional position
  r: number; // radius in CSS px
  a: number; // base opacity
}

// Pre-generate all stars once at module level — stable across re-renders
const STARS: Star[] = Array.from({ length: 260 }, () => ({
  x: rng(),
  y: rng(),
  r: 0.25 + rng() * 1.35,
  a: 0.06 + rng() * 0.58,
}));

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function LandingStars() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    const draw = () => {
      const W = parent.offsetWidth;
      const H = parent.offsetHeight;
      if (W === 0 || H === 0) return;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      canvas.style.width = `${W}px`;
      canvas.style.height = `${H}px`;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, W, H);

      for (const star of STARS) {
        const x = star.x * W;
        const y = star.y * H;

        ctx.globalAlpha = star.a;
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(x, y, star.r, 0, Math.PI * 2);
        ctx.fill();

        // A handful of brighter stars get a soft halo
        if (star.r > 1.2 && star.a > 0.45) {
          const grd = ctx.createRadialGradient(x, y, 0, x, y, star.r * 3.5);
          grd.addColorStop(0, `rgba(255,255,255,${star.a * 0.35})`);
          grd.addColorStop(1, "rgba(255,255,255,0)");
          ctx.fillStyle = grd;
          ctx.beginPath();
          ctx.arc(x, y, star.r * 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      ctx.globalAlpha = 1;
    };

    // Initial draw (wait one frame so parent has correct dimensions)
    const rafId = requestAnimationFrame(draw);

    // Keep in sync with any layout changes (e.g. content loads, font swap)
    const ro = new ResizeObserver(draw);
    ro.observe(parent);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="landing-stars-canvas"
    />
  );
}

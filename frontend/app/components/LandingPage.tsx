"use client";

import Image from "next/image";
import { signIn } from "next-auth/react";
import ParticleWordmark from "./ParticleWordmark";
import LandingStars from "./LandingStars";

const PIPELINE = [
  { t: "Brief", d: "Product, brand and goal in" },
  { t: "Concepts", d: "Director proposes 2–3 distinct ideas" },
  { t: "Planner", d: "Per-channel asset specs" },
  { t: "Generation", d: "Reference-conditioned render" },
  { t: "Evaluation", d: "VLM + SigLIP + OCR scores it" },
  { t: "Approved", d: "A campaign you can ship" },
];

const FEATURES = [
  {
    kicker: "Preserved, not reinvented",
    title: "Reference-conditioned generation",
    body: "Your real product image is the conditioning input, not just a line in a prompt — so product identity survives into every generated frame.",
  },
  {
    kicker: "Recomposed, not cropped",
    title: "Cross-platform recomposition",
    body: "One concept becomes three genuinely different compositions — Instagram Feed, Story, Website Hero — each re-framed and re-placed for its channel.",
  },
  {
    kicker: "Self-correcting",
    title: "Hybrid evaluation",
    body: "A VLM + SigLIP + OCR engine scores product fidelity and composition, and automatically regenerates anything below threshold before it reaches you.",
  },
];

const SIGN_IN = { callbackUrl: "/studio" } as const;

export default function LandingPage() {
  return (
    <div className="landing">
      {/* Star field — absolute canvas at z-index 0; hero image covers it
          in the hero section, sections show stars through their dark bg */}
      <LandingStars />
      <header className="lp-nav">
        <span className="lp-wordmark">CIBA</span>
        <button className="lp-btn" onClick={() => signIn("google", SIGN_IN)}>
          Sign in with Google
        </button>
      </header>

      <section className="lp-hero">
        <div className="lp-hero-media">
          <Image
            src="/bg.png"
            alt="Campaign creative built by CIBA"
            fill
            priority
            sizes="100vw"
            className="lp-hero-img"
          />
        </div>
        {/* Particle animation — absolute overlay, z-index 5, pointer-events none */}
        <ParticleWordmark word="BRAND" />
      </section>

      <section className="lp-section">
        <h2 className="lp-section-title">
          Not another <em>prompt-to-image</em> tool.
        </h2>
        <p className="lp-section-sub">
          A prompt-to-image tool makes one picture from one prompt. CIBA plans the whole
          campaign: the Director turns your brief into concepts, the Planner recomposes them
          per channel, generation renders them from your real product imagery, and the
          evaluator scores and re-runs anything below standard.
        </p>
        <div className="lp-pipeline">
          {PIPELINE.map((step, i) => (
            <div className="lp-step" key={step.t}>
              <span className="n">0{i + 1}</span>
              <span className="t">{step.t}</span>
              <span className="d">{step.d}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="lp-section">
        <h2 className="lp-section-title">
          The three things it gets right.
        </h2>
        <div className="lp-cards">
          {FEATURES.map((f) => (
            <div className="lp-card" key={f.title}>
              <span className="kicker">{f.kicker}</span>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="lp-footer">
        <div className="lp-footer-inner">
          <span className="lp-wordmark">CIBA</span>
          <span>
            © {new Date().getFullYear()} CIBA — agentic creative campaign engine. Built as a
            portfolio project.
          </span>
        </div>
      </footer>
    </div>
  );
}

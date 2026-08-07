"use client";

import { useState } from "react";
import { useSession, signIn, signOut } from "next-auth/react";
import {
  type Asset,
  type Brand,
  type Campaign,
  type Concept,
  type Product,
  TERMINAL_ASSET_STATUSES,
  createBrand,
  createProduct,
  setProductImages,
  createCampaign,
  getCampaign,
  getConcepts,
  selectConcept,
  getAssets,
  regenerateAsset,
  poll,
  mediaUrl,
} from "../lib/api";

type Step = "setup" | "brief" | "concepts" | "assets";

const STEP_LABELS: { id: Step; label: string }[] = [
  { id: "setup", label: "Brand & product" },
  { id: "brief", label: "Brief" },
  { id: "concepts", label: "Concepts" },
  { id: "assets", label: "Assets" },
];

const stepIdx = (s: Step) => STEP_LABELS.findIndex((x) => x.id === s);

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

/* ------------------------------------------------------------------ */

export function CampaignStudio() {
  const { data: session, status } = useSession();

  if (status === "loading") {
    return (
      <div className="center">
        <span className="spinner" /> <span className="muted">Loading…</span>
      </div>
    );
  }
  if (!session) return <SignInScreen />;
  return <Studio />;
}

function SignInScreen() {
  return (
    <div className="studio">
      <div className="hero">
        <div className="eyebrow">CIBA · Agentic Creative Campaign Engine</div>
        <h1>From product to campaign, without a prompt.</h1>
        <p>
          Turn a product, brand and brief into a directed multi-channel campaign —
          concept, plan, generate, evaluate, and improve.
        </p>
        <button className="btn btn-primary" onClick={() => signIn("google")}>
          Continue with Google
        </button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Studio() {
  const { data: session } = useSession();
  const [step, setStep] = useState<Step>("setup");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [brand, setBrand] = useState<Brand | null>(null);
  const [product, setProduct] = useState<Product | null>(null);
  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);

  const go = (s: Step) => {
    setStep(s);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleSetup = (b: Brand, p: Product) => {
    setBrand(b);
    setProduct(p);
    go("brief");
  };

  const handleBrief = async (c: Campaign) => {
    setCampaign(c);
    setError(null);
    setBusy(true);
    try {
      await poll(
        () => getCampaign(c.id),
        (c2) => c2.status !== "generating_concepts"
      );
      const c2 = await getCampaign(c.id);
      if (c2.status === "failed") throw new Error("The Director failed to produce concepts.");
      const list = await getConcepts(c.id);
      setConcepts(list);
      go("concepts");
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const handleSelectConcept = async (conceptId: string) => {
    if (!campaign) return;
    setError(null);
    setBusy(true);
    try {
      await selectConcept(campaign.id, conceptId);
      const result = await poll(
        () => getAssets(campaign.id),
        (list) => list.length > 0 && list.every((a) => TERMINAL_ASSET_STATUSES.includes(a.status))
      );
      setAssets(result);
      go("assets");
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRegenerate = async (assetId: string) => {
    if (!campaign) return;
    setError(null);
    setBusy(true);
    try {
      await regenerateAsset(campaign.id, assetId);
      const result = await poll(
        () => getAssets(campaign.id),
        (list) => list.length > 0 && list.every((a) => TERMINAL_ASSET_STATUSES.includes(a.status))
      );
      setAssets(result);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const restart = () => {
    setCampaign(null);
    setConcepts([]);
    setAssets([]);
    go("setup");
  };

  return (
    <div className="studio">
      <header className="studio-header">
        <div className="studio-wordmark">
          <h1>CIBA</h1>
          <span className="tag">Creative Campaign Engine</span>
        </div>
        <div className="gap">
          <span className="small muted">{session?.user?.name ?? "…"}</span>
          <button className="btn btn-ghost" onClick={() => signOut()}>
            Sign out
          </button>
        </div>
      </header>

      <div className="steps">
        {STEP_LABELS.map((s) => (
          <span
            key={s.id}
            className={`step ${
              stepIdx(s.id) === stepIdx(step) ? "active" : ""
            } ${stepIdx(s.id) < stepIdx(step) ? "done" : ""}`}
          >
            {s.label}
          </span>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {step === "setup" && <SetupPanel onComplete={handleSetup} />}
      {step === "brief" && (
        <BriefPanel
          brand={brand}
          product={product}
          onSubmitted={handleBrief}
          onBack={() => go("setup")}
        />
      )}
      {step === "concepts" && <ConceptsPanel concepts={concepts} busy={busy} onSelect={handleSelectConcept} />}
      {step === "assets" && (
        <AssetsPanel assets={assets} busy={busy} onRegenerate={handleRegenerate} onRestart={restart} />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */

function SetupPanel({ onComplete }: { onComplete: (b: Brand, p: Product) => void }) {
  const [brand, setBrand] = useState<Brand | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [bName, setBName] = useState("");
  const [bTone, setBTone] = useState("");
  const [pName, setPName] = useState("");
  const [pDesc, setPDesc] = useState("");
  const [paths, setPaths] = useState("");

  const createBrandHandler = async () => {
    setError(null);
    setBusy(true);
    try {
      const b = await createBrand({ name: bName, tone: bTone || undefined });
      setBrand(b);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const createProductHandler = async () => {
    if (!brand) return;
    setError(null);
    setBusy(true);
    try {
      let p = await createProduct({
        brand_id: brand.id,
        name: pName,
        description: pDesc || undefined,
      });
      const list = paths
        .split(/[\n,]/)
        .map((s) => s.trim())
        .filter(Boolean);
      if (list.length) p = await setProductImages(p.id, list);
      onComplete(brand, p);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <div className="section-title">Brand &amp; product setup</div>
      <div className="section-sub">
        One-time setup the engine works from. Product image paths can be Kaggle dataset paths
        or URLs, one per line.
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="row">
        <div className="field">
          <label>Brand name</label>
          <input
            className="input"
            value={bName}
            onChange={(e) => setBName(e.target.value)}
            placeholder="e.g. Pulse Athletic"
          />
        </div>
        <div className="field">
          <label>Tone / voice</label>
          <input
            className="input"
            value={bTone}
            onChange={(e) => setBTone(e.target.value)}
            placeholder="e.g. Bold and aspirational"
          />
        </div>
      </div>
      <button className="btn" disabled={busy || !bName.trim()} onClick={createBrandHandler}>
        {brand ? `✓ Brand created — ${brand.name}` : "Create brand"}
      </button>

      {brand && (
        <>
          <div style={{ height: 18 }} />
          <div className="row">
            <div className="field">
              <label>Product name</label>
              <input
                className="input"
                value={pName}
                onChange={(e) => setPName(e.target.value)}
                placeholder="e.g. Signature Carbon Sneaker"
              />
            </div>
            <div className="field">
              <label>Product description</label>
              <input
                className="input"
                value={pDesc}
                onChange={(e) => setPDesc(e.target.value)}
                placeholder="e.g. Limited edition sneaker"
              />
            </div>
          </div>
          <div className="field">
            <label>Product image paths (Kaggle dataset or URL, one per line)</label>
            <textarea
              className="input"
              rows={2}
              value={paths}
              onChange={(e) => setPaths(e.target.value)}
              placeholder={"/kaggle/input/my-product/p1.webp"}
            />
          </div>
          <button className="btn btn-primary" disabled={busy || !pName.trim()} onClick={createProductHandler}>
            Create product &amp; continue
          </button>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */

function BriefPanel({
  brand,
  product,
  onSubmitted,
  onBack,
}: {
  brand: Brand | null;
  product: Product | null;
  onSubmitted: (c: Campaign) => void;
  onBack: () => void;
}) {
  const [brief, setBrief] = useState("");
  const [audience, setAudience] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!brand || !product) return;
    setError(null);
    setBusy(true);
    try {
      const c = await createCampaign({
        brand_id: brand.id,
        product_id: product.id,
        brief_text: brief,
        target_audience: audience || undefined,
      });
      onSubmitted(c);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <div className="section-title">Campaign brief</div>
      <div className="section-sub">The Director turns this into 2–3 distinct creative concepts.</div>
      <div className="small muted" style={{ marginBottom: 14 }}>
        Brand: <strong>{brand?.name}</strong> · Product: <strong>{product?.name}</strong>
      </div>
      {error && <div className="error-banner">{error}</div>}
      <div className="field">
        <label>Brief</label>
        <textarea
          className="input"
          rows={4}
          value={brief}
          onChange={(e) => setBrief(e.target.value)}
          placeholder="Launch campaign for our new sneaker, targeting urban millennials…"
        />
      </div>
      <div className="field">
        <label>Target audience (optional)</label>
        <input
          className="input"
          value={audience}
          onChange={(e) => setAudience(e.target.value)}
          placeholder="Urban millennials 25–35"
        />
      </div>
      <div className="gap">
        <button className="btn" onClick={onBack} disabled={busy}>
          Back
        </button>
        <button className="btn btn-primary" disabled={busy || !brief.trim()} onClick={submit}>
          {busy ? "Director is thinking…" : "Generate concepts"}
        </button>
      </div>
      {busy && (
        <div className="small muted" style={{ marginTop: 12 }}>
          <span className="spinner" />
          Running the Director…
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */

function ConceptsPanel({
  concepts,
  busy,
  onSelect,
}: {
  concepts: Concept[];
  busy: boolean;
  onSelect: (id: string) => void;
}) {
  if (!concepts.length) {
    return (
      <div className="panel center">
        <span className="spinner" /> Waiting for the Director…
      </div>
    );
  }
  return (
    <div className="panel">
      <div className="section-title">Choose a concept</div>
      <div className="section-sub">
        Only the selected concept gets generated — Kaggle GPU is scarce.
      </div>
      <div className="concept-grid">
        {concepts.map((c) => (
          <div className="concept-card" key={c.id}>
            <h3>{c.name}</h3>
            <div className="headline">“{c.ad_copy.headline}”</div>
            <div className="swatch-row">
              {(c.visual_dna.palette ?? []).map((hex) => (
                <span key={hex} className="swatch" style={{ background: hex }} />
              ))}
            </div>
            <div className="desc">{c.description}</div>
            <div>
              {(c.visual_dna.mood ?? []).map((m) => (
                <span key={m} className="chip accent">
                  {m}
                </span>
              ))}
              <span className="chip">{c.visual_dna.photography_style}</span>
              <span className="chip">{c.visual_dna.lighting}</span>
            </div>
            {c.ad_copy.subcopy && <div className="small muted">“{c.ad_copy.subcopy}”</div>}
            <button className="btn btn-primary btn-block" disabled={busy} onClick={() => onSelect(c.id)}>
              {busy ? "Planning & generating…" : "Select & generate"}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

const PLACEMENT_LABELS: Record<string, string> = {
  ig_feed: "Instagram Feed · 4:5",
  ig_story: "Instagram Story · 9:16",
  website_hero: "Website Hero · 16:9",
};

function statusBadge(status: string) {
  switch (status) {
    case "approved":
      return <span className="badge badge-ok">Approved</span>;
    case "manual_review":
      return <span className="badge badge-warn">Manual review</span>;
    case "infra_failed":
      return <span className="badge badge-bad">Infra failed</span>;
    default:
      return <span className="badge">{status}</span>;
  }
}

function AssetImage({ url, label }: { url: string | null; label: string }) {
  const [failed, setFailed] = useState(false);
  if (!url || failed) {
    return <div className="placeholder">{label} · no image yet</div>;
  }
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={url} alt={label} onError={() => setFailed(true)} />;
}

function AssetsPanel({
  assets,
  busy,
  onRegenerate,
  onRestart,
}: {
  assets: Asset[];
  busy: boolean;
  onRegenerate: (assetId: string) => void;
  onRestart: () => void;
}) {
  if (!assets.length) {
    return (
      <div className="panel center">
        <span className="spinner" /> Waiting for generation…
      </div>
    );
  }
  return (
    <div className="panel">
      <div className="section-title">Generated assets</div>
      <div className="section-sub">
        Every attempt is stored — this is the “system improved its output” screen.
      </div>
      <div className="asset-grid">
        {assets.map((asset) => {
          const latest = asset.attempts[asset.attempts.length - 1];
          return (
            <div className="asset-card" key={asset.id}>
              <div className="asset-head">
                <h4>{PLACEMENT_LABELS[asset.placement] ?? asset.placement}</h4>
                {statusBadge(asset.status)}
              </div>
              <div className="asset-media">
                <AssetImage url={mediaUrl(latest?.image_url ?? null)} label={asset.placement} />
              </div>
              {asset.attempts.map((a) => {
                const ev = a.evaluation;
                const url = mediaUrl(a.image_url);
                return (
                  <div className="attempt" key={a.attempt_number}>
                    <div className="attempt-head">
                      <strong>Attempt {a.attempt_number}</strong>
                      <span>
                        {a.infra_failed
                          ? "infra failure"
                          : ev
                            ? ev.passed
                              ? "passed ✓"
                              : "failed"
                            : "evaluating…"}
                      </span>
                    </div>
                    {ev && (
                      <>
                        <div className="small" style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span>Product fidelity</span>
                          <span>{Math.round(ev.product_fidelity * 100)}%</span>
                        </div>
                        <div className="bar">
                          <span style={{ width: `${Math.round(ev.product_fidelity * 100)}%` }} />
                        </div>
                        <div className="small" style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span>Overall</span>
                          <span>{Math.round(ev.overall_score * 100)}%</span>
                        </div>
                        <div className="bar">
                          <span style={{ width: `${Math.round(ev.overall_score * 100)}%` }} />
                        </div>
                        {ev.failure_reason && (
                          <div className="attempt-meta attempt-reason">{ev.failure_reason}</div>
                        )}
                      </>
                    )}
                    {url && <div className="attempt-meta">image: {url.split("/").pop()}</div>}
                  </div>
                );
              })}
              {asset.status === "manual_review" && (
                <div className="asset-actions">
                  <button className="btn btn-block" disabled={busy} onClick={() => onRegenerate(asset.id)}>
                    {busy ? "Regenerating…" : "Regenerate"}
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="gap" style={{ marginTop: 20 }}>
        <button className="btn btn-ghost" onClick={onRestart}>
          New campaign
        </button>
      </div>
    </div>
  );
}

// Client-side API client for the FastAPI backend.
// Every call mints a fresh Bearer token via /api/token (server-side HS256 JWS)
// and hits NEXT_PUBLIC_API_URL (default: http://localhost:8000/api/v1).
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

// Origin (without /api/v1) — used to build /media image URLs for the gallery.
export const API_ORIGIN = API_BASE.replace(/\/api\/v1\/?$/, "");

export type Brand = {
  id: string;
  name: string;
  description: string | null;
  primary_colors: string[];
  secondary_colors: string[];
  fonts: string[];
  tone: string | null;
  logo_url: string | null;
};

export type Product = {
  id: string;
  brand_id: string;
  name: string;
  description: string | null;
  product_images: string[];
};

export type Campaign = {
  id: string;
  status: string;
  brief_text: string;
  selected_concept_id: string | null;
};

export type Concept = {
  id: string;
  name: string;
  description: string;
  visual_dna: {
    palette: string[];
    lighting: string;
    environment: string;
    materials: string[];
    mood: string[];
    photography_style: string;
  };
  ad_copy: { headline: string; subcopy: string | null; cta: string | null };
  rationale: string;
  status: string;
};

export type Attempt = {
  attempt_number: number;
  image_url: string | null;
  infra_failed: boolean;
  evaluation: {
    overall_score: number;
    product_fidelity: number;
    brand_consistency: number;
    composition_score: number;
    prompt_alignment: number;
    passed: boolean;
    failure_reason: string | null;
  } | null;
};

export type Asset = {
  id: string;
  platform: string;
  placement: string;
  aspect_ratio: string;
  status: string;
  attempts: Attempt[];
};

export const TERMINAL_ASSET_STATUSES = ["approved", "manual_review", "infra_failed"];

async function getToken(): Promise<string> {
  const res = await fetch("/api/token", { cache: "no-store" });
  if (!res.ok) throw new Error("Not signed in");
  const data = await res.json();
  return data.token as string;
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = await getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(init.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${body.slice(0, 300)}`);
  }
  if (res.status === 204) return {} as T;
  return res.json();
}

// --- Brands / Products ---
export const createBrand = (b: {
  name: string;
  description?: string;
  primary_colors?: string[];
  secondary_colors?: string[];
  fonts?: string[];
  tone?: string;
}) =>
  api<Brand>("/brands", {
    method: "POST",
    body: JSON.stringify({
      description: null,
      primary_colors: [],
      secondary_colors: [],
      fonts: [],
      tone: null,
      ...b,
    }),
  });

export const createProduct = (p: { brand_id: string; name: string; description?: string }) =>
  api<Product>("/products", { method: "POST", body: JSON.stringify({ description: null, ...p }) });

export const setProductImages = (productId: string, product_images: string[]) =>
  api<Product>(`/products/${productId}/images`, {
    method: "POST",
    body: JSON.stringify({ product_images }),
  });

// --- Campaigns ---
export const createCampaign = (c: {
  brand_id: string;
  product_id: string;
  brief_text: string;
  target_audience?: string;
}) =>
  api<Campaign>("/campaigns", {
    method: "POST",
    body: JSON.stringify({ target_audience: null, ...c }),
  });

export const getCampaign = (id: string) => api<Campaign>(`/campaigns/${id}`);
export const getConcepts = (id: string) => api<Concept[]>(`/campaigns/${id}/concepts`);

export const selectConcept = (campaignId: string, conceptId: string) =>
  api<Campaign>(`/campaigns/${campaignId}/select-concept`, {
    method: "POST",
    body: JSON.stringify({ concept_id: conceptId }),
  });

export const getAssets = (campaignId: string) => api<Asset[]>(`/campaigns/${campaignId}/assets`);

export const regenerateAsset = (campaignId: string, assetId: string) =>
  api<Asset>(`/campaigns/${campaignId}/assets/${assetId}/regenerate`, { method: "POST" });

// --- Polling helper (the 202 -> poll contract) ---
export async function poll<T>(
  fn: () => Promise<T>,
  done: (value: T) => boolean,
  { interval = 2000, timeout = 10 * 60 * 1000 } = {}
): Promise<T> {
  const started = Date.now();
  for (;;) {
    const value = await fn();
    if (done(value)) return value;
    if (Date.now() - started > timeout) {
      throw new Error("Timed out waiting for the backend to finish.");
    }
    await new Promise((r) => setTimeout(r, interval));
  }
}

export function mediaUrl(imageUrl: string | null): string | null {
  if (!imageUrl) return null;
  const file = imageUrl.split("/").pop();
  return file ? `${API_ORIGIN}/media/${file}` : null;
}

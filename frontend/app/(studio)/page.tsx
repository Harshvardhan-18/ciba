import { CampaignStudio } from "../components/CampaignStudio";

// This is a fully client-side, auth-gated studio — no static prerendering
// (useSession must not run at build time).
export const dynamic = "force-dynamic";

export default function Home() {
  return <CampaignStudio />;
}

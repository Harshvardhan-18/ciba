import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";
import { authOptions } from "./api/auth/[...nextauth]/options";
import LandingPage from "./components/LandingPage";

// The landing page checks the session server-side (uses cookies) and never
// gets statically prerendered.
export const dynamic = "force-dynamic";

export default async function Home() {
  const session = await getServerSession(authOptions);
  // Signed-in users go straight to the studio — the marketing page is for
  // people who haven't started yet.
  if (session?.user) redirect("/studio");
  return <LandingPage />;
}

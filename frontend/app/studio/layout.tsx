"use client";

// Route-group layout for the studio — SessionProvider lives here (NOT in the
// root layout) so the auto-generated _global-error prerender never touches
// next-auth context.
import { SessionProvider } from "next-auth/react";

export default function StudioLayout({ children }: { children: React.ReactNode }) {
  return <SessionProvider>{children}</SessionProvider>;
}

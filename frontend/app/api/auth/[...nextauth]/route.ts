// frontend/app/api/auth/[...nextauth]/route.ts
//
// NextAuth v4 handler mounted at the App Router catch-all.
// Re-exports the handler from the shared options file so the config
// stays in one place (options.ts) and Next.js gets its GET/POST exports here.

import NextAuth from "next-auth";
import { authOptions } from "./options";

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };

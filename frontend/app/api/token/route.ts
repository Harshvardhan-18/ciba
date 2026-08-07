// app/api/token/route.ts
//
// Signs a fresh HS256 JWS (plain JWT, matching app/auth.py on FastAPI) for the
// signed-in user, so the browser can call the backend with
// `Authorization: Bearer <token>`. NextAuth's raw session JWT isn't exposed by
// the client API, so we re-sign the same claims server-side with the same
// NEXTAUTH_SECRET — one secret, one algorithm, two sides.
import { getServerSession } from "next-auth";
import * as jose from "jose";
import { authOptions } from "../auth/[...nextauth]/options";

const SECRET = process.env.NEXTAUTH_SECRET!;

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return Response.json({ error: "Not signed in" }, { status: 401 });
  }
  const user = session.user as typeof session.user & { google_sub?: string };
  if (!user.google_sub || !user.email) {
    return Response.json({ error: "Token missing claims (sub, email)" }, { status: 401 });
  }

  const now = Math.floor(Date.now() / 1000);
  const token = await new jose.SignJWT({
    sub: user.google_sub,
    email: user.email,
    name: user.name ?? undefined,
    picture: (user as unknown as { picture?: string }).picture ?? user.image ?? undefined,
  })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt(now)
    .setExpirationTime(now + 60 * 60) // 1h — enough for a campaign run
    .sign(new TextEncoder().encode(SECRET));

  return Response.json({ token });
}

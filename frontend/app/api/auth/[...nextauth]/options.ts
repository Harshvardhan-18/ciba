// frontend/app/api/auth/[...nextauth]/options.ts
//
// NextAuth configuration.
//
// WHY CUSTOM jwt.encode / jwt.decode?
// ---------------------------------------------------------------------------
// NextAuth v4's default JWT strategy uses JWE (JSON Web Encryption,
// specifically A256GCM) — tokens are *encrypted*, not just signed.
// The FastAPI backend verifies tokens using python-jose's jwt.decode(),
// which expects a plain signed JWT (JWS), not a JWE ciphertext.
//
// Rather than adding JWE decryption to the FastAPI side (which would add
// complexity and a new dependency), we override NextAuth's encode/decode to
// produce/consume a standard HS256-signed JWT instead. The secret is still
// NEXTAUTH_SECRET — same source, same value — so the security model is
// unchanged; the token is signed (authenticity + integrity) but not
// encrypted (contents are base64-visible). This is the same security posture
// as most stateless JWT setups and is acceptable because the token is
// transmitted over HTTPS only.
//
// Claim shape kept aligned with what app/auth.py expects:
//   sub     — Google's stable user ID (set by NextAuth from the Google profile)
//   email   — user's email
//   name    — user's display name
//   picture — Google profile photo URL
//   iat     — issued-at (Unix seconds)
//   exp     — expiry (Unix seconds)
// ---------------------------------------------------------------------------

import type { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import * as jose from "jose";

// NEXTAUTH_SECRET must be set in .env.local (and on the server in production).
// It is the shared secret used for both NextAuth (this file) and the FastAPI
// backend (NEXTAUTH_SECRET env var). One secret, one algorithm, two sides.
const SECRET = process.env.NEXTAUTH_SECRET!;

if (!SECRET) {
  throw new Error("NEXTAUTH_SECRET is not set. Add it to .env.local");
}

// Encode the secret as bytes — jose expects a Uint8Array for symmetric algorithms.
const secretBytes = new TextEncoder().encode(SECRET);

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],

  session: {
    strategy: "jwt",
  },

  jwt: {
    // -----------------------------------------------------------------------
    // encode: produce a plain HS256 signed JWT instead of NextAuth's default
    // JWE (encrypted) token. Uses the `jose` library (already a transitive
    // NextAuth dependency) for consistency — no extra package needed.
    // -----------------------------------------------------------------------
    async encode({ secret: _secret, token, maxAge }) {
      if (!token) throw new Error("Cannot encode undefined token");

      const now = Math.floor(Date.now() / 1000);
      const expiresIn = maxAge ?? 30 * 24 * 60 * 60; // default: 30 days

      return new jose.SignJWT({
        sub: token.sub,
        email: token.email,
        name: token.name,
        picture: token.picture,
        // Pass through any extra fields NextAuth adds (e.g. jti, nbf).
        // Spread first so our explicit fields above take precedence.
        ...token,
      })
        .setProtectedHeader({ alg: "HS256" })
        .setIssuedAt(now)
        .setExpirationTime(now + expiresIn)
        .sign(secretBytes);
    },

    // -----------------------------------------------------------------------
    // decode: verify the HS256 signature and return the payload.
    // Mirror of app/auth.py's jwt.decode() call — same algorithm, same secret.
    // -----------------------------------------------------------------------
    async decode({ secret: _secret, token }) {
      if (!token) return null;
      try {
        const { payload } = await jose.jwtVerify(token, secretBytes, {
          algorithms: ["HS256"],
        });
        return payload as unknown as ReturnType<typeof decode>;
      } catch {
        return null;
      }
    },
  },

  callbacks: {
    // Persist google_sub (provider account ID) into the JWT so FastAPI
    // can key on it. NextAuth sets token.sub to the provider's account ID
    // for the first sign-in; we just make it explicit here.
    async jwt({ token, account, profile }) {
      if (account && profile) {
        // First login: enrich the token with Google profile fields.
        token.sub = account.providerAccountId; // Google's stable user ID
        token.email = profile.email;
        token.name = profile.name;
        token.picture = (profile as { picture?: string }).picture;
      }
      return token;
    },

    async session({ session, token }) {
      // Expose the google_sub on the session object so client code can
      // read it without decoding the JWT manually.
      if (session.user) {
        (session.user as { google_sub?: string }).google_sub = token.sub;
      }
      return session;
    },
  },
};

// Re-export the decode function type for use in tests/type-checking.
declare function decode(params: {
  secret: string;
  token?: string;
}): Promise<Record<string, unknown> | null>;

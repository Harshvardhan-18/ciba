"use client";

// Explicit global error boundary. Rendering its own <html>/<body> (no
// next/font, no SessionProvider) also avoids the auto-generated /_global-error
// prerender that trips over root-layout context.
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          fontFamily: "system-ui, sans-serif",
          background: "#08090d",
          color: "#ede9e1",
          padding: 48,
          minHeight: "100vh",
        }}
      >
        <h1 style={{ fontSize: 28 }}>Something went wrong</h1>
        <p style={{ opacity: 0.7 }}>{error.message}</p>
        <button
          onClick={() => reset()}
          style={{
            marginTop: 16,
            padding: "10px 20px",
            borderRadius: 999,
            border: "1px solid #e3b96b",
            background: "transparent",
            color: "#e3b96b",
            cursor: "pointer",
          }}
        >
          Try again
        </button>
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CIBA — Agentic Creative Campaign Engine",
  description: "Turn a product + brand + brief into a multi-channel ad campaign.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

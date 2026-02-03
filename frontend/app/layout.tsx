import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fathom Financial Agent",
  description:
    "AI-powered financial analysis agent for 10-K reports and complex financial reasoning",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-terminal-bg">{children}</body>
    </html>
  );
}

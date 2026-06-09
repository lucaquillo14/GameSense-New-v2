import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GameSense AI",
  description: "Phase 1 football physical metrics analysis",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}


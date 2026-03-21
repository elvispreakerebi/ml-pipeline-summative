import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { NavSidebar } from "@/components/nav-sidebar";
import { Providers } from "./providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "EmotiCall - Emergency Call Emotion Detection",
  description: "Classify emotions in emergency call audio recordings",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          <NavSidebar />
          <main className="ml-64 min-h-screen p-8">{children}</main>
        </Providers>
      </body>
    </html>
  );
}

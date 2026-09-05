import { Plus_Jakarta_Sans, Chakra_Petch, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import TopNav from "@/components/TopNav";
import AntigravityBackground from "@/components/AntigravityBackground";

const plusJakartaSans = Plus_Jakarta_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const chakraPetch = Chakra_Petch({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata = {
  title: "Unified ML - Tactical AI Engine",
  description: "Build, train, evaluate, quantize, and deploy AI models.",
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      className={`dark ${plusJakartaSans.variable} ${chakraPetch.variable} ${jetbrainsMono.variable}`}
    >
      <body className="min-h-screen relative font-sans antialiased overflow-x-hidden bg-black text-[#E0E0E0]">
        {/* Antigravity Background */}
        <AntigravityBackground color="#94a3b8" count={260} particleSize={1.8} magnetRadius={1} />

        {/* Ambient Grid and Centered Logo Background */}
        <div className="fixed inset-0 pointer-events-none z-0 bg-grid-pattern opacity-20"></div>
        <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden flex items-center justify-center opacity-[0.035]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            alt="Unified ML Watermark"
            className="w-[600px] h-[600px] object-contain select-none"
            src="/logo.png"
          />
        </div>

        <TopNav />

        {children}
      </body>
    </html>
  );
}

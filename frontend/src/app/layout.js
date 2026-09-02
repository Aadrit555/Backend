import "./globals.css";
import TopNav from "@/components/TopNav";
import AntigravityBackground from "@/components/AntigravityBackground";

export const metadata = {
  title: "Unified ML - Model Builder",
  description: "Build, train, evaluate, improve, and deploy AI models.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Geist:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen relative font-body-md text-body-md overflow-x-hidden bg-black text-[#E0E0E0]">
        {/* React Bits Antigravity 3D Interactive Background */}
        <AntigravityBackground color="#94a3b8" count={260} particleSize={1.8} magnetRadius={1} />

        {/* Ambient Grid and Centered Logo Background */}
        <div className="fixed inset-0 pointer-events-none z-0 bg-grid-pattern opacity-20"></div>
        <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden flex items-center justify-center opacity-[0.035]">
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

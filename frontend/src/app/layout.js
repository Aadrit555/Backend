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
        <AntigravityBackground color="#94a3b8" count={260} particleSize={1.8} />

        {/* Ambient Grid and Logo Background */}
        <div className="fixed inset-0 pointer-events-none z-0 bg-grid-pattern opacity-20"></div>
        <div className="fixed right-0 top-0 bottom-0 w-[50vw] pointer-events-none z-0 overflow-hidden flex items-center justify-end opacity-[0.02]">
          <img
            alt="Unified ML Watermark"
            className="w-[800px] h-[800px] object-cover translate-x-1/4"
            src="/logo.png"
          />
        </div>

        <TopNav />

        {children}
      </body>
    </html>
  );
}

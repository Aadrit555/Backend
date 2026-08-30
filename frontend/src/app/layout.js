/**
 * Root layout — BIBLE §6.
 *
 * Applies global styles and wraps all pages.
 * The platform's purpose (BIBLE §0): "Give us your data.
 * Tell us what you want. We'll build the model."
 */

import "./globals.css";

export const metadata = {
  title: "Unified AI/ML Model Builder",
  description:
    "Build, train, evaluate, improve, and deploy AI models by simply providing your data and describing what you want.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <nav style={{ padding: '1rem 2rem', borderBottom: '1px solid #333', display: 'flex', gap: '2rem', alignItems: 'center' }}>
          <h2 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 700 }}>Unified ML</h2>
          <a href="/" style={{ color: '#fff', textDecoration: 'none', fontWeight: 500 }}>Model Builder</a>
          <a href="/data-prep" style={{ color: '#fff', textDecoration: 'none', fontWeight: 500 }}>Data Factory</a>
        </nav>
        {children}
      </body>
    </html>
  );
}

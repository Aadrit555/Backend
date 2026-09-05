/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      "colors": {
        "tertiary-fixed": "#e4e2e1",
        "surface-dim": "#131313",
        "secondary-fixed-dim": "#c8c6c5",
        "surface": "#131313",
        "on-error": "#690005",
        "on-secondary-fixed": "#1c1b1b",
        "background": "#131313",
        "error-container": "#93000a",
        "on-tertiary": "#303030",
        "tertiary-fixed-dim": "#c8c6c6",
        "surface-variant": "#353535",
        "on-error-container": "#ffdad6",
        "tertiary": "#efecec",
        "outline-variant": "#3b494c",
        "primary-fixed": "#9cf0ff",
        "inverse-surface": "#e2e2e2",
        "on-surface": "#e2e2e2",
        "outline": "#849396",
        "on-primary-fixed-variant": "#004f58",
        "surface-container-high": "#2a2a2a",
        "on-tertiary-container": "#595959",
        "on-surface-variant": "#bac9cc",
        "on-tertiary-fixed": "#1b1c1c",
        "on-primary": "#00363d",
        "surface-container-highest": "#353535",
        "primary": "#c3f5ff",
        "error": "#ffb4ab",
        "surface-tint": "#00daf3",
        "secondary-fixed": "#e5e2e1",
        "primary-container": "#00e5ff",
        "on-secondary": "#313030",
        "on-secondary-fixed-variant": "#474746",
        "on-background": "#e2e2e2",
        "surface-container-lowest": "#0e0e0e",
        "on-primary-fixed": "#001f24",
        "on-tertiary-fixed-variant": "#474747",
        "on-primary-container": "#00626e",
        "surface-container": "#1f1f1f",
        "inverse-primary": "#006875",
        "secondary": "#c8c6c5",
        "primary-fixed-dim": "#00daf3",
        "surface-container-low": "#1b1b1b",
        "surface-bright": "#393939",
        "tertiary-container": "#d2d0d0",
        "inverse-on-surface": "#303030",
        "secondary-container": "#474746",
        "on-secondary-container": "#b7b5b4"
      },
      "borderRadius": {
        "DEFAULT": "0.25rem",
        "lg": "0.5rem",
        "xl": "0.75rem",
        "full": "9999px"
      },
      "spacing": {
        "unit": "4px",
        "margin-desktop": "40px",
        "gutter": "16px",
        "margin-mobile": "16px",
        "container-max": "1440px"
      },
      "fontFamily": {
        "sans": ["var(--font-sans)", "system-ui", "-apple-system", "sans-serif"],
        "body-md": ["var(--font-sans)", "system-ui", "-apple-system", "sans-serif"],
        "headline-lg": ["var(--font-display)", "var(--font-sans)", "sans-serif"],
        "label-caps": ["var(--font-mono)", "monospace"],
        "code-sm": ["var(--font-mono)", "monospace"],
        "mono": ["var(--font-mono)", "monospace"],
        "display-lg": ["var(--font-display)", "var(--font-sans)", "sans-serif"]
      },
      "fontSize": {
        "body-md": ["14px", { "lineHeight": "1.6", "fontWeight": "400" }],
        "headline-lg": ["32px", { "lineHeight": "1.2", "letterSpacing": "-0.01em", "fontWeight": "600" }],
        "label-caps": ["11px", { "lineHeight": "16px", "letterSpacing": "0.15em", "fontWeight": "600" }],
        "code-sm": ["12px", { "lineHeight": "20px", "fontWeight": "400" }],
        "headline-lg-mobile": ["24px", { "lineHeight": "1.2", "fontWeight": "600" }],
        "display-lg": ["46px", { "lineHeight": "1.1", "letterSpacing": "-0.02em", "fontWeight": "700" }]
      }
    }
  },
  plugins: [],
};

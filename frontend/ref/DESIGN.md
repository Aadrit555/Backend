---
name: Architectural Logic
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1b1b1b'
  surface-container: '#1f1f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353535'
  on-surface: '#e2e2e2'
  on-surface-variant: '#bac9cc'
  inverse-surface: '#e2e2e2'
  inverse-on-surface: '#303030'
  outline: '#849396'
  outline-variant: '#3b494c'
  surface-tint: '#00daf3'
  primary: '#c3f5ff'
  on-primary: '#00363d'
  primary-container: '#00e5ff'
  on-primary-container: '#00626e'
  inverse-primary: '#006875'
  secondary: '#c8c6c5'
  on-secondary: '#313030'
  secondary-container: '#474746'
  on-secondary-container: '#b7b5b4'
  tertiary: '#efecec'
  on-tertiary: '#303030'
  tertiary-container: '#d2d0d0'
  on-tertiary-container: '#595959'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#9cf0ff'
  primary-fixed-dim: '#00daf3'
  on-primary-fixed: '#001f24'
  on-primary-fixed-variant: '#004f58'
  secondary-fixed: '#e5e2e1'
  secondary-fixed-dim: '#c8c6c5'
  on-secondary-fixed: '#1c1b1b'
  on-secondary-fixed-variant: '#474746'
  tertiary-fixed: '#e4e2e1'
  tertiary-fixed-dim: '#c8c6c6'
  on-tertiary-fixed: '#1b1c1c'
  on-tertiary-fixed-variant: '#474747'
  background: '#131313'
  on-background: '#e2e2e2'
  surface-variant: '#353535'
typography:
  display-lg:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '600'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '500'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: '1.2'
  body-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.6'
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.15em
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 20px
spacing:
  unit: 4px
  gutter: 16px
  margin-desktop: 40px
  margin-mobile: 16px
  container-max: 1440px
---

## Brand & Style

This design system is engineered for machine learning practitioners who demand precision and high-density information environments. The brand personality is **analytical, architectural, and absolute**. It rejects the soft, friendly aesthetics of consumer SaaS in favor of a "terminal-plus" experience—utilitarian but refined.

The visual style is a blend of **Minimalism** and **Modern Brutalism**. It relies on high-contrast typography, razor-sharp edges, and a strict adherence to a grid. The UI acts as a silent scaffold for complex data, using the provided logo (the impossible-geometry 'N/M') as a structural motif. A subtle, large-scale watermark of the logo is positioned in the background of deep-scroll pages at 3% opacity to reinforce the architectural depth without distracting from the data.

## Colors

The palette is strictly functional. The **Pure Black (#000000)** background provides an infinite canvas that eliminates frame-to-bezel distraction on modern displays. 

- **Primary Accent (#00E5FF):** Used exclusively for high-priority interactive states, progress indicators, and successful build paths. 
- **Subtle Gray Dividers (#1A1A1A):** These are the structural bones of the system. Use them for all borders, grid lines, and section separations.
- **Tonal Hierarchy:** Primary text is pure white for maximum legibility. Secondary text uses a mid-tone gray to reduce visual noise in dense technical documentation and metadata.

## Typography

The typography system leverages **Geist** for its clinical, geometric precision and **JetBrains Mono** for technical data and labels. 

- **Headlines:** Should be tight, high-contrast, and used sparingly to define major layout zones.
- **Technical Labels:** The `label-caps` style is mandatory for section headers, table column headers, and status badges. The wide letter-spacing is designed to create a "blueprint" aesthetic.
- **Readability:** Body text is kept at 14px to allow for high information density without sacrificing clarity.

## Layout & Spacing

This system utilizes a **Fixed Grid** within a fluid container. Elements align to a strict 4px baseline rhythm. 

- **Desktop:** 12-column grid with 16px gutters. Panels and code editors should snap to column boundaries.
- **Division:** Use 1px borders (`#1A1A1A`) instead of negative space to define areas. This creates an "instrument cluster" feel. 
- **Density:** Information density is high. Padding inside cards and containers should be a consistent 24px, while vertical spacing between distinct modules should be 40px to provide breathing room.

## Elevation & Depth

Depth is conveyed through **Tonal Layering** rather than shadows. 
- **Level 0 (Background):** Pure black (#000000).
- **Level 1 (Panels/Cards):** Dark gray (#0A0A0A) with a 1px solid border (#1A1A1A).
- **Level 2 (Modals/Popovers):** Slightly lighter gray (#141414) with a slightly more prominent border (#333333).

**No shadows are permitted.** Instead of elevation via Z-axis offsets, use "active" border colors (the cyan accent) to indicate focus or selection.

## Shapes

The shape language is **Sharp**. A 0px radius is preferred for all primary structural elements (containers, inputs, buttons). 

In specific instances where a "technical softness" is required (such as nested status tags), a maximum radius of **2px** may be applied, but this should be the exception. Horizontal and vertical rules should be exactly 1px thick.

## Components

- **Buttons:** Rectangular with 0px radius. Primary buttons use a solid `#00E5FF` background with black text. Secondary buttons are transparent with a `#1A1A1A` border and white text. Hover states should trigger a border color shift rather than a background change.
- **Input Fields:** Bottom-border only or very subtle 1px outlines. Focus state is indicated by the bottom border turning `#00E5FF`. Use JetBrains Mono for input text.
- **Chips/Status Tags:** Small, uppercase text. For "Running" or "Active" states, use a 4x4px solid cyan square icon next to the text instead of a pill shape.
- **Navigation:** The navbar is a 1px bottom-bordered strip. The Unified ML logo sits on the far left, rendered in pure white or the cyan accent.
- **Data Tables:** No alternating row colors. Use 1px horizontal dividers only. Column headers must use the `label-caps` typography style.
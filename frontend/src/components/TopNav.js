"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function TopNav() {
  const pathname = usePathname();

  const navLinks = [
    { name: "MODEL BUILDER", path: "/" },
    { name: "DATA FACTORY", path: "/data-prep" },
  ];

  return (
    <nav className="bg-transparent border-b border-[#1A1A1A] w-full relative z-50">
      <div className="flex items-center h-12 w-full px-margin-desktop max-w-[1440px] mx-auto">
        {/* Left side: Brand */}
        <div className="flex-1 flex items-center justify-start">
          <div className="flex items-center gap-3">
            <img
              alt="Unified ML Logo"
              className="h-6 w-6 object-contain"
              src="/logo.png"
            />
            <span className="text-label-caps font-label-caps font-bold tracking-widest text-on-background uppercase">
              UNIFIED ML
            </span>
          </div>
        </div>

        {/* Center: Navigation Links */}
        <div className="flex items-center h-full gap-12 justify-center">
          {navLinks.map((link) => {
            const isActive = pathname === link.path;
            return (
              <Link
                key={link.path}
                href={link.path}
                className={`h-full flex items-center pt-[2px] font-bold text-label-caps font-label-caps hover:text-white transition-colors duration-200 border-b-2 ${
                  isActive
                    ? "text-[#00E5FF] border-[#00E5FF]"
                    : "text-on-surface-variant border-transparent"
                }`}
              >
                {link.name}
              </Link>
            );
          })}
        </div>

        {/* Right side: Empty space to balance the flex layout */}
        <div className="flex-1"></div>
      </div>
    </nav>
  );
}

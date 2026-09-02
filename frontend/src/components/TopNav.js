"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Cpu, Activity } from "lucide-react";

export default function TopNav() {
  const pathname = usePathname();

  const navLinks = [
    { name: "STUDIO", path: "/" },
    { name: "DATA FACTORY", path: "/data-prep" },
  ];

  return (
    <nav className="bg-[#0A0A0A]/90 backdrop-blur-md border-b border-[#1E1E1E] w-full sticky top-0 z-50">
      <div className="flex items-center justify-between h-12 w-full px-6 max-w-[1440px] mx-auto">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <img
            alt="Unified ML Logo"
            className="h-5 w-5 object-contain"
            src="/logo.png"
          />
          <span className="font-mono text-xs font-bold tracking-widest text-white uppercase">
            UNIFIED AI
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[#1C1C1C] text-[#888888] border border-[#2A2A2A] rounded-xs">
            v0.1
          </span>
        </div>

        {/* Center: Navigation Links */}
        <div className="flex items-center h-full gap-8">
          {navLinks.map((link) => {
            const isActive = pathname === link.path;
            return (
              <Link
                key={link.path}
                href={link.path}
                className={`h-full flex items-center text-xs font-mono font-medium transition-colors border-b-2 ${
                  isActive
                    ? "text-[#00E5FF] border-[#00E5FF]"
                    : "text-[#888888] hover:text-white border-transparent"
                }`}
              >
                {link.name}
              </Link>
            );
          })}
        </div>

        {/* Right side: Hardware status */}
        <div className="flex items-center gap-2 text-[11px] font-mono text-[#777777]">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>BACKEND CONNECTED</span>
        </div>
      </div>
    </nav>
  );
}

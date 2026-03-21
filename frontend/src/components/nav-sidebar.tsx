"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Mic, RefreshCw, BarChart3, Activity } from "lucide-react";

const navItems = [
  { href: "/", label: "Predict", icon: Mic },
  { href: "/retrain", label: "Retrain", icon: RefreshCw },
  { href: "/insights", label: "Insights", icon: BarChart3 },
  { href: "/status", label: "Status", icon: Activity },
];

export function NavSidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r bg-card">
      <div className="flex h-full flex-col">
        <div className="border-b p-6">
          <h1 className="text-lg font-bold">EmotiCall</h1>
          <p className="text-xs text-muted-foreground">
            Emergency Call Emotion Detection
          </p>
        </div>

        <nav className="flex-1 space-y-1 p-4">
          {navItems.map((item) => {
            const isActive =
              pathname === item.href ||
              (item.href !== "/" && pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                )}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="border-t p-4">
          <p className="text-xs text-muted-foreground">
            ML Pipeline Summative
          </p>
        </div>
      </div>
    </aside>
  );
}

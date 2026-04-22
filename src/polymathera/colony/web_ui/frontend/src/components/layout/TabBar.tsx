import { cn } from "@/lib/utils";

export interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
}

interface TabBarProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (id: string) => void;
  notifications?: Record<string, number>;
}

export function TabBar({ tabs, activeTab, onTabChange, notifications }: TabBarProps) {
  return (
    <nav className="flex gap-1 border-b bg-background px-3">
      {tabs.map((tab) => {
        const count = notifications?.[tab.id] || 0;
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={cn(
              "relative flex items-center gap-1.5 px-3 py-2.5 text-sm font-medium transition-colors",
              "hover:text-foreground",
              activeTab === tab.id
                ? "text-foreground"
                : "text-muted-foreground"
            )}
          >
            {tab.icon}
            {tab.label}
            {/* Activity indicator dot */}
            {count > 0 && activeTab !== tab.id && (
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-400" />
            )}
            {/* Active tab underline */}
            {activeTab === tab.id && (
              <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-primary" />
            )}
          </button>
        );
      })}
    </nav>
  );
}

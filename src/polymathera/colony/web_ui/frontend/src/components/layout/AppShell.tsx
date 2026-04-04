import { useEffect, useState } from "react";
import { TabBar, type Tab } from "./TabBar";
import { StatusBar } from "./StatusBar";
import { ErrorBoundary } from "../shared/ErrorBoundary";
import { OverviewTab } from "../dashboard/OverviewTab";
import { AgentsTab } from "../agents/AgentsTab";
import { SessionsTab } from "../sessions/SessionsTab";
import { VCMTab } from "../vcm/VCMTab";
import { PageGraphTab } from "../graph/PageGraphTab";
import { BlackboardTab } from "../blackboard/BlackboardTab";
import { InteractTab } from "../interact/InteractTab";
import { LogsTab } from "../logs/LogsTab";
import { MetricsTab } from "../observability/MetricsTab";
import { TracesTab } from "../observability/TracesTab";
import { SettingsTab } from "../settings/SettingsTab";

const TABS: Tab[] = [
  { id: "overview", label: "Overview" },
  { id: "agents", label: "Agents" },
  { id: "sessions", label: "Sessions" },
  { id: "vcm", label: "VCM" },
  { id: "graph", label: "Page Graph" },
  { id: "blackboard", label: "Blackboard" },
  { id: "interact", label: "Interact" },
  { id: "logs", label: "Logs" },
  { id: "traces", label: "Traces" },
  { id: "metrics", label: "Metrics" },
  { id: "settings", label: "Settings" },
];

// Maps tab id to its component. Lazy-mounted: a tab's component is only created
// the first time that tab is visited, then kept alive (hidden via display:none)
// so that component state (e.g. expand/collapse in Traces) survives tab switches.
const TAB_COMPONENTS: Record<string, React.FC> = {
  overview: OverviewTab,
  agents: AgentsTab,
  sessions: SessionsTab,
  vcm: VCMTab,
  graph: PageGraphTab,
  blackboard: BlackboardTab,
  interact: InteractTab,
  logs: LogsTab,
  traces: TracesTab,
  metrics: MetricsTab,
  settings: SettingsTab,
};

// Tabs that use WebGL or other resources that don't survive display:none.
// These are fully unmounted when not active instead of kept alive.
const UNMOUNT_WHEN_HIDDEN = new Set(["graph"]);

function TabContent({ activeTab }: { activeTab: string }) {
  const [mounted, setMounted] = useState<Set<string>>(() => new Set([activeTab]));

  useEffect(() => {
    setMounted((prev) => {
      if (prev.has(activeTab)) return prev;
      const next = new Set(prev);
      next.add(activeTab);
      return next;
    });
  }, [activeTab]);

  return (
    <>
      {TABS.map(({ id }) => {
        const shouldUnmount = UNMOUNT_WHEN_HIDDEN.has(id);
        // Unmount-when-hidden tabs: only render when active
        // Keep-alive tabs: render once mounted, hide with display:none
        if (shouldUnmount && activeTab !== id) return null;
        if (!shouldUnmount && !mounted.has(id)) return null;
        const Component = TAB_COMPONENTS[id];
        if (!Component) return null;
        return (
          <div key={id} className="h-full overflow-auto" style={{ display: activeTab === id ? "block" : "none" }}>
            <ErrorBoundary name={id}>
              <Component />
            </ErrorBoundary>
          </div>
        );
      })}
    </>
  );
}

export function AppShell() {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b px-4">
        <div className="flex items-center gap-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/10">
            <span className="text-sm font-bold text-primary">C</span>
          </div>
          <h1 className="text-sm font-semibold tracking-tight">
            Colony Dashboard
          </h1>
        </div>
      </header>

      {/* Tabs */}
      <TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Content — no overflow-auto here; each tab controls its own scroll */}
      <main className="flex-1 min-h-0 p-5">
        <TabContent activeTab={activeTab} />
      </main>

      {/* Status bar */}
      <StatusBar />
    </div>
  );
}

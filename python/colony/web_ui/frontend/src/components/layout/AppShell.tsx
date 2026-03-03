import { useState } from "react";
import { TabBar, type Tab } from "./TabBar";
import { StatusBar } from "./StatusBar";
import { OverviewTab } from "../dashboard/OverviewTab";
import { AgentsTab } from "../agents/AgentsTab";
import { SessionsTab } from "../sessions/SessionsTab";
import { VCMTab } from "../vcm/VCMTab";
import { PageGraphTab } from "../graph/PageGraphTab";
import { BlackboardTab } from "../blackboard/BlackboardTab";
import { InteractTab } from "../interact/InteractTab";
import { LogsTab } from "../logs/LogsTab";
import { MetricsTab } from "../observability/MetricsTab";
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
  { id: "metrics", label: "Metrics" },
  { id: "settings", label: "Settings" },
];

function TabContent({ activeTab }: { activeTab: string }) {
  switch (activeTab) {
    case "overview":
      return <OverviewTab />;
    case "agents":
      return <AgentsTab />;
    case "sessions":
      return <SessionsTab />;
    case "vcm":
      return <VCMTab />;
    case "graph":
      return <PageGraphTab />;
    case "blackboard":
      return <BlackboardTab />;
    case "interact":
      return <InteractTab />;
    case "logs":
      return <LogsTab />;
    case "metrics":
      return <MetricsTab />;
    case "settings":
      return <SettingsTab />;
    default:
      return null;
  }
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

      {/* Content */}
      <main className="flex-1 overflow-auto p-5">
        <TabContent activeTab={activeTab} />
      </main>

      {/* Status bar */}
      <StatusBar />
    </div>
  );
}

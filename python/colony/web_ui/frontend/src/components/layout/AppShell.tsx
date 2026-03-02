import { useState } from "react";
import { TabBar, type Tab } from "./TabBar";
import { StatusBar } from "./StatusBar";
import { OverviewTab } from "../dashboard/OverviewTab";
import { AgentsTab } from "../agents/AgentsTab";
import { SessionsTab } from "../sessions/SessionsTab";
import { VCMTab } from "../vcm/VCMTab";
import { MetricsTab } from "../observability/MetricsTab";
import { BlackboardTab } from "../blackboard/BlackboardTab";

const TABS: Tab[] = [
  { id: "overview", label: "Overview" },
  { id: "agents", label: "Agents" },
  { id: "sessions", label: "Sessions" },
  { id: "vcm", label: "VCM" },
  { id: "blackboard", label: "Blackboard" },
  { id: "metrics", label: "Metrics" },
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
    case "blackboard":
      return <BlackboardTab />;
    case "metrics":
      return <MetricsTab />;
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

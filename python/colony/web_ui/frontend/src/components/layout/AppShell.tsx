import { useState } from "react";
import { TabBar, type Tab } from "./TabBar";
import { StatusBar } from "./StatusBar";
import { OverviewTab } from "../dashboard/OverviewTab";
import { AgentsTab } from "../agents/AgentsTab";
import { SessionsTab } from "../sessions/SessionsTab";
import { VCMTab } from "../vcm/VCMTab";

const TABS: Tab[] = [
  { id: "overview", label: "Overview" },
  { id: "agents", label: "Agents" },
  { id: "sessions", label: "Sessions" },
  { id: "vcm", label: "VCM" },
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
    default:
      return null;
  }
}

export function AppShell() {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="flex h-12 items-center justify-between border-b px-4">
        <h1 className="text-lg font-semibold tracking-tight">
          Colony Dashboard
        </h1>
      </header>

      {/* Tabs */}
      <TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Content */}
      <main className="flex-1 overflow-auto p-4">
        <TabContent activeTab={activeTab} />
      </main>

      {/* Status bar */}
      <StatusBar />
    </div>
  );
}

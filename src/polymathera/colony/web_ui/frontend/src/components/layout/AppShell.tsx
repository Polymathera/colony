import { useEffect, useState, useCallback } from "react";
import { LogOut, LayoutDashboard, Bot, Clock, Database, GitFork, ClipboardList, MessageSquare, ScrollText, Activity, Gauge, Settings } from "lucide-react";
import { TabBar, type Tab } from "./TabBar";
import { StatusBar } from "./StatusBar";
import { Sidebar } from "./Sidebar";
import { LandingPage } from "./LandingPage";
import { AuthPage } from "../auth/AuthPage";
import { ErrorBoundary } from "../shared/ErrorBoundary";
import { SubmitRunDialog } from "../dialogs/SubmitRunDialog";
import { OverviewTab } from "../dashboard/OverviewTab";
import { AgentsTab } from "../agents/AgentsTab";
import { SessionsTab } from "../sessions/SessionsTab";
import { VCMTab } from "../vcm/VCMTab";
import { PageGraphTab } from "../graph/PageGraphTab";
import { BlackboardTab } from "../blackboard/BlackboardTab";
import { InteractTab, setInteractSessionId } from "../interact/InteractTab";
import { LogsTab } from "../logs/LogsTab";
import { MetricsTab } from "../observability/MetricsTab";
import { TracesTab } from "../observability/TracesTab";
import { SettingsTab } from "../settings/SettingsTab";
import { useCreateSession } from "@/api/hooks/useSessions";
import { useCurrentUser, useLogout, useColonies } from "@/api/hooks/useAuth";

const TABS: Tab[] = [
  { id: "overview", label: "Overview", icon: <LayoutDashboard size={14} /> },
  { id: "agents", label: "Agents", icon: <Bot size={14} /> },
  { id: "sessions", label: "Sessions", icon: <Clock size={14} /> },
  { id: "vcm", label: "VCM", icon: <Database size={14} /> },
  { id: "graph", label: "Page Graph", icon: <GitFork size={14} /> },
  { id: "blackboard", label: "Blackboard", icon: <ClipboardList size={14} /> },
  { id: "interact", label: "Interact", icon: <MessageSquare size={14} /> },
  { id: "logs", label: "Logs", icon: <ScrollText size={14} /> },
  { id: "traces", label: "Traces", icon: <Activity size={14} /> },
  { id: "metrics", label: "Metrics", icon: <Gauge size={14} /> },
  { id: "settings", label: "Settings", icon: <Settings size={14} /> },
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
  const [activeTab, setActiveTab] = useState(() => localStorage.getItem("colony_active_tab") || "overview");
  const [activeSessionId, setActiveSessionId] = useState<string | null>(() => localStorage.getItem("colony_active_session"));
  const [activeColonyId, setActiveColonyId] = useState<string | null>(() => localStorage.getItem("colony_active_colony"));
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showRunDialog, setShowRunDialog] = useState(false);

  // Persist UI state to localStorage
  useEffect(() => { localStorage.setItem("colony_active_tab", activeTab); }, [activeTab]);
  useEffect(() => {
    if (activeSessionId) localStorage.setItem("colony_active_session", activeSessionId);
    else localStorage.removeItem("colony_active_session");
  }, [activeSessionId]);
  useEffect(() => {
    if (activeColonyId) localStorage.setItem("colony_active_colony", activeColonyId);
    else localStorage.removeItem("colony_active_colony");
  }, [activeColonyId]);

  // Auth state
  const currentUser = useCurrentUser();
  const logout = useLogout();
  const isAuthenticated = !!currentUser.data && !currentUser.isError;
  const colonies = useColonies({ enabled: isAuthenticated });
  const createSession = useCreateSession();
  // Only show loading spinner briefly on initial load, not on refetches.
  // isFetching is true during refetches; isLoading is true only when there's
  // no cached data yet AND the query is in flight.
  const isLoading = currentUser.isLoading && !currentUser.isError;

  // Set default colony when user data loads
  useEffect(() => {
    if (currentUser.data && !activeColonyId) {
      const defaultColony = currentUser.data.colonies.find((c) => c.is_default);
      if (defaultColony) {
        setActiveColonyId(defaultColony.colony_id);
      } else if (currentUser.data.colonies.length > 0) {
        setActiveColonyId(currentUser.data.colonies[0].colony_id);
      }
    }
  }, [currentUser.data, activeColonyId]);

  // Sync session ID to InteractTab (which can't receive props in current tab architecture)
  useEffect(() => {
    setInteractSessionId(activeSessionId);
  }, [activeSessionId]);

  // Set X-Colony-Id header for all API calls when colony changes
  useEffect(() => {
    if (activeColonyId) {
      // Store colony_id so apiFetch can include it
      (window as any).__colony_active_colony_id = activeColonyId;
    }
  }, [activeColonyId]);

  const handleCreateSession = useCallback(async () => {
    const result = await createSession.mutateAsync({});
    if (result.status === "created") {
      setActiveSessionId(result.session_id);
    } else {
      const { showErrorToast } = await import("../shared/ErrorToast");
      showErrorToast(result.message || "Failed to create session");
    }
  }, [createSession]);

  const handleJobSubmitted = useCallback((_jobId: string, sessionId: string) => {
    setActiveSessionId(sessionId);
    setActiveTab("overview");
  }, []);

  const handleLogout = useCallback(async () => {
    await logout.mutateAsync();
    setActiveSessionId(null);
    setActiveColonyId(null);
  }, [logout]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-sm text-muted-foreground">Loading...</div>
      </div>
    );
  }

  // Not authenticated — show auth page
  if (!isAuthenticated) {
    return (
      <div className="flex h-screen flex-col">
        <main className="flex-1 min-h-0">
          <AuthPage onAuthenticated={() => currentUser.refetch()} />
        </main>
        <StatusBar authenticated={isAuthenticated} />
      </div>
    );
  }

  // Authenticated — show full dashboard
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
          {activeSessionId && (
            <span className="text-xs text-muted-foreground font-mono">
              {activeSessionId.slice(0, 16)}...
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Colony selector */}
          {colonies.data && colonies.data.length > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-muted-foreground">Colony:</span>
              <select
                className="rounded border border-border bg-background px-2 py-1 text-xs"
                value={activeColonyId || ""}
                onChange={(e) => setActiveColonyId(e.target.value)}
              >
                {colonies.data.map((c) => (
                  <option key={c.colony_id} value={c.colony_id}>
                    {c.name} {c.is_default ? "(default)" : ""}
                  </option>
                ))}
              </select>
            </div>
          )}
          {/* User info + logout */}
          <span className="text-xs text-muted-foreground">
            {currentUser.data?.username}
          </span>
          <button
            onClick={handleLogout}
            className="rounded px-2 py-1 text-xs text-muted-foreground hover:text-foreground hover:bg-accent/50"
          >
            <LogOut size={14} className="inline -mt-0.5" /> Logout
          </button>
        </div>
      </header>

      {/* Body: Sidebar + Main */}
      <div className="flex flex-1 min-h-0">
        {/* Sidebar */}
        <Sidebar
          activeSessionId={activeSessionId}
          onSelectSession={setActiveSessionId}
          onStartRun={() => setShowRunDialog(true)}
          colonyReady={!!activeColonyId}
          collapsed={sidebarCollapsed}
          onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
        />

        {/* Main content */}
        <div className="flex flex-1 flex-col min-w-0">
          {activeSessionId ? (
            <>
              {/* Tabs */}
              <TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

              {/* Content — no overflow-auto here; each tab controls its own scroll */}
              <main className="flex-1 min-h-0 p-5">
                <TabContent activeTab={activeTab} />
              </main>
            </>
          ) : createSession.isPending ? (
            <main className="flex flex-1 items-center justify-center min-h-0">
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                Creating session...
              </div>
            </main>
          ) : (
            /* Landing page when no session selected */
            <main className="flex-1 min-h-0">
              <LandingPage
                onSelectSession={setActiveSessionId}
                onCreateSession={handleCreateSession}
              />
            </main>
          )}
        </div>
      </div>

      {/* Status bar */}
      <StatusBar authenticated={isAuthenticated} />

      {/* Dialogs */}
      <SubmitRunDialog
        open={showRunDialog}
        onClose={() => setShowRunDialog(false)}
        onSubmitted={handleJobSubmitted}
      />
    </div>
  );
}

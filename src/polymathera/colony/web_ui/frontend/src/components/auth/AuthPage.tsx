/**
 * Sign-in landing page. VCS-OAuth only — no password forms.
 *
 * A single "Sign in with GitHub" button kicks off a full-page
 * navigation to ``/api/v1/auth/github/sign-in``. The backend 302s
 * to GitHub; on callback the user lands back at ``/`` with auth
 * cookies set + ``useCurrentUser`` refetches automatically.
 *
 * Future: render one button per registered provider (GitLab,
 * Bitbucket). The ``startVcsSignIn(provider_id)`` helper is
 * provider-agnostic; this component will read the enabled-providers
 * list from a new ``/api/v1/auth/providers`` route once PR 7 adds it.
 */
import { Github } from "lucide-react";
import { startVcsSignIn } from "@/api/hooks/useAuth";


export function AuthPage() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-8">
      {/* Logo */}
      <div className="text-center">
        <h1
          className="text-5xl font-bold tracking-widest text-primary"
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          THE COLONY
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Civilization-Building AI
        </p>
        <p className="mt-4 text-[10px] uppercase tracking-widest text-muted-foreground/60">
          By Polymathera
        </p>
      </div>

      {/* Sign-in button — single VCS provider for now. */}
      <div className="w-full max-w-sm space-y-3">
        <button
          type="button"
          onClick={() => startVcsSignIn("github")}
          className="flex w-full items-center justify-center gap-2 rounded bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          <Github size={16} /> Sign in with GitHub
        </button>
        <p className="text-center text-[10px] text-muted-foreground">
          You'll be redirected to GitHub to authorise Colony. We never see
          your password. After approval you'll land back here with your
          tenants and Colony-marked repos auto-discovered.
        </p>
      </div>
    </div>
  );
}


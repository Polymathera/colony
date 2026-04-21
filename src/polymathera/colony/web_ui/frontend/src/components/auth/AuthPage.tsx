import { useState } from "react";
import { useSignup, useLogin } from "@/api/hooks/useAuth";

interface AuthPageProps {
  onAuthenticated: () => void;
}

export function AuthPage({ onAuthenticated }: AuthPageProps) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const login = useLogin();
  const signup = useSignup();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!username.trim() || !password.trim()) {
      setError("Username and password are required");
      return;
    }

    try {
      if (mode === "login") {
        await login.mutateAsync({ username: username.trim(), password });
      } else {
        if (password.length < 6) {
          setError("Password must be at least 6 characters");
          return;
        }
        await signup.mutateAsync({ username: username.trim(), password });
      }
      onAuthenticated();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Authentication failed";
      // Extract detail from API error
      if (msg.includes("409")) setError("Username already exists");
      else if (msg.includes("401")) setError("Invalid username or password");
      else setError(msg);
    }
  };

  const isPending = login.isPending || signup.isPending;

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

      {/* Auth form */}
      <div className="w-full max-w-sm">
        {/* Mode toggle */}
        <div className="flex rounded border border-border mb-4">
          <button
            className={`flex-1 px-4 py-2 text-xs font-medium transition-colors ${
              mode === "login"
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
            onClick={() => { setMode("login"); setError(null); }}
          >
            Log In
          </button>
          <button
            className={`flex-1 px-4 py-2 text-xs font-medium transition-colors ${
              mode === "signup"
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
            onClick={() => { setMode("signup"); setError(null); }}
          >
            Sign Up
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              autoComplete="username"
              className="w-full rounded border border-border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === "signup" ? "Min 6 characters" : "Enter password"}
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              className="w-full rounded border border-border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none"
            />
          </div>

          {error && (
            <div className="rounded bg-red-500/10 px-3 py-2 text-xs text-red-400">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isPending}
            className="w-full rounded bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
          >
            {isPending
              ? (mode === "login" ? "Logging in..." : "Creating account...")
              : (mode === "login" ? "Log In" : "Create Account")}
          </button>
        </form>

        {mode === "signup" && (
          <p className="mt-3 text-center text-[10px] text-muted-foreground">
            A default workspace will be created for you automatically.
          </p>
        )}
      </div>
    </div>
  );
}

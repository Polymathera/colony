const BASE_URL = "/api/v1";

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  // Include active colony ID if set (used by AuthMiddleware for ExecutionContext)
  const colonyId = (window as any).__colony_active_colony_id as string | undefined;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(colonyId ? { "X-Colony-Id": colonyId } : {}),
    ...((init?.headers as Record<string, string>) ?? {}),
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000); // 10s timeout

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers,
      credentials: "same-origin", // Include cookies (auth tokens)
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
  if (!res.ok) {
    let detail = `API ${res.status}: ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
      else if (body.message) detail = body.message;
    } catch {
      // body wasn't JSON — use statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

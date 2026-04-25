const BASE_URL = "/api/v1";

// Default request timeout. Generous on purpose: session/agent creation
// involves Ray actor spawn + capability initialization which routinely
// takes 15–30s on cold cluster, and the previous 10s ceiling caused
// silent aborts where the backend completed the work but the frontend
// gave up before the response landed (leaving a "signal is aborted
// without reason" toast and a phantom session in "Recent Sessions"
// that never auto-opened). React Query handles per-call cancellation
// independently, so a long ceiling here is a backstop, not the primary
// timeout mechanism.
const DEFAULT_TIMEOUT_MS = 60_000;

export interface ApiFetchInit extends RequestInit {
  /** Override the default request timeout (ms). Pass 0 to disable. */
  timeoutMs?: number;
}

export async function apiFetch<T>(path: string, init?: ApiFetchInit): Promise<T> {
  // Include active colony ID if set (used by AuthMiddleware for ExecutionContext)
  const colonyId = (window as any).__colony_active_colony_id as string | undefined;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(colonyId ? { "X-Colony-Id": colonyId } : {}),
    ...((init?.headers as Record<string, string>) ?? {}),
  };

  // Compose two abort sources into one signal that we hand to fetch:
  //   1. The caller's signal — React Query passes its query/mutation
  //      cancellation signal here. Honoring it is what lets a
  //      cancelled query stop in flight instead of running to
  //      completion just to be ignored.
  //   2. Our own timeout — fires after ``timeoutMs``. The default is
  //      large enough for cold-start session creation; per-call
  //      overrides exist for endpoints with different latency
  //      profiles. Pass ``timeoutMs: 0`` to disable.
  // Without honoring (1), apiFetch would silently drop React Query's
  // cancellation. Without (2), a hung backend would never time out.
  const controller = new AbortController();
  const callerSignal = init?.signal;
  if (callerSignal) {
    if (callerSignal.aborted) {
      controller.abort();
    } else {
      callerSignal.addEventListener(
        "abort",
        () => controller.abort(),
        { once: true },
      );
    }
  }
  const timeoutMs = init?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timeout = timeoutMs > 0
    ? setTimeout(() => controller.abort(), timeoutMs)
    : null;

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers,
      credentials: "same-origin", // Include cookies (auth tokens)
      signal: controller.signal,
    });
  } catch (err) {
    // Surface a helpful message instead of "signal is aborted without
    // reason" when our timeout fired. The caller's MutationCache /
    // QueryCache onError handlers render this in a toast.
    if (
      err instanceof DOMException
      && err.name === "AbortError"
      && controller.signal.aborted
      && !callerSignal?.aborted
    ) {
      throw new Error(
        `Request timed out after ${timeoutMs}ms (${path}). The backend may still complete the operation.`,
      );
    }
    throw err;
  } finally {
    if (timeout !== null) clearTimeout(timeout);
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

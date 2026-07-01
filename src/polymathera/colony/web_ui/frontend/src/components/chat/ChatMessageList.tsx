import { useEffect, useMemo, useRef } from "react";
import { ChatMessage, type ChatMessageData } from "./ChatMessage";

interface ChatMessageListProps {
  messages: ChatMessageData[];
  onReply?: (message: ChatMessageData, content: string) => void;
  emptyText?: string;
  sessionId?: string | null;
}

export function ChatMessageList({ messages, onReply, emptyText, sessionId }: ChatMessageListProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // Track whether the user is currently near the bottom. Updated by
  // the scroll listener on every user-initiated scroll. The
  // ``messages``-change effect reads this — NOT the live
  // ``scrollHeight - scrollTop - clientHeight`` measurement, because
  // by the time React's effect runs the new message is already in
  // the DOM and has grown ``scrollHeight``, making the live
  // computation always report "not near bottom". The prior
  // implementation had this bug — the user was at the bottom but new
  // messages did not trigger auto-scroll. The ref preserves the
  // user's last-known scroll INTENT (true at session start; flips on
  // scroll events).
  const nearBottomRef = useRef<boolean>(true);

  // "Busy" = the Colony has at least one action currently in flight.
  // Drives the bottom-of-timeline spinner ("we're still working on
  // your request"). Computed from the same ``messages`` array the
  // timeline renders — the per-status-entry ``status_phase=running``
  // is the canonical signal. A "complete"/"failed" record flips the
  // matching entry's phase, dropping it from this set.
  const busy = useMemo(
    () => messages.some(
      (m) => m.kind === "status" && m.status_phase === "running",
    ),
    [messages],
  );

  useEffect(() => {
    if (nearBottomRef.current) {
      endRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, busy]);

  const handleScroll = () => {
    const container = containerRef.current;
    if (!container) return;
    // Same 100px threshold as before — the user is "at the bottom"
    // if they're within 100px of the scroll-bottom. Computed at
    // scroll-event time (when scrollHeight reflects what the user
    // sees), NOT at effect-time (when scrollHeight already includes
    // the new message).
    nearBottomRef.current =
      container.scrollHeight - container.scrollTop - container.clientHeight < 100;
  };

  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
        {emptyText || "Send a message to begin."}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="h-full overflow-auto p-3"
    >
      {/* Unified timeline: every entry shares the same left-rail
          visual (via ChatMessage's own ``before:`` pseudo-element
          on its left column). No outer border / per-entry card —
          the rail is the visual spine; entries hang off it as
          inline content. */}
      {messages.map((msg) => (
        <ChatMessage
          key={msg.id}
          message={msg}
          // Interactive UI (response buttons, compose textarea, etc.)
          // is suppressed in the timeline — active requests render
          // in :class:`ActiveRequestsOverlay` so the user can act on
          // them without scrolling. The timeline shows the question
          // historically; the overlay carries the widget.
          interactive={false}
          onReply={onReply}
          sessionId={sessionId}
        />
      ))}
      {/* Bottom-of-timeline working indicator. Sits AS A RAIL NODE so
          the spine continues to its dot, and falls off cleanly when
          the last running action finishes. The dot is a spinning
          circle (not a static color) so it visually reads "in
          progress" without needing any text. */}
      {busy && (
        <div
          className="relative flex gap-2 py-1.5 pl-6 text-[11px] text-muted-foreground
                     before:absolute before:left-[11px] before:top-0 before:bottom-1/2 before:w-px before:bg-border"
          aria-live="polite"
        >
          <span
            className="absolute left-2 top-2 z-10 inline-flex h-2.5 w-2.5 items-center justify-center rounded-full ring-2 ring-card"
            aria-hidden
          >
            <span className="h-2.5 w-2.5 animate-spin rounded-full border border-blue-400 border-t-transparent" />
          </span>
          <span className="leading-5">Working…</span>
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}

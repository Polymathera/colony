/**
 * Pinned-overlay panel for currently-active operator requests
 * (human_approval / human_help / guardrail_waiver).
 *
 * Lives in the slot above the chat input that the prior
 * ``ActionStatusBanner`` + ``MissionStatusBanner`` vacated when status
 * was unified into the timeline. Why an overlay:
 *
 *   - Active requests MUST stay in the operator's view. The chat
 *     timeline auto-scrolls to the bottom as new messages + status
 *     entries arrive, which would push an active question off-screen
 *     and the operator would never see it.
 *   - Active requests need ROOM — the human_help guidance textarea, the
 *     reject/abort compose-explanation textarea, and the typed-approval
 *     button row don't fit in a one-line banner.
 *
 * Implementation:
 *
 *   - Source of truth is :data:`ChatPanel.messages` — the same array
 *     the timeline renders. ActiveRequestsOverlay filters to the entries
 *     where ``awaiting_reply === true`` and ``kind`` is one of the three
 *     typed request kinds.
 *   - Each active request renders via :class:`ChatMessage` with
 *     ``interactive=true``, so the response widgets (buttons, compose
 *     mode, guidance textarea) are owned in ONE place — no duplication.
 *   - The timeline's :class:`ChatMessageList` passes ``interactive=false``
 *     so the same entries appear as historical context but without the
 *     interactive widgets. Operator can read the question's history on
 *     scroll; they always act here in the overlay.
 */
import { ChatMessage, type ChatMessageData } from "./ChatMessage";

const INTERACTIVE_KINDS = new Set([
  "human_approval",
  "human_help",
  "guardrail_waiver",
]);

interface ActiveRequestsOverlayProps {
  messages: ChatMessageData[];
  onReply: (
    message: ChatMessageData,
    content: string,
    extra?: { explanation?: string; guidance?: string },
  ) => void;
}

export function ActiveRequestsOverlay({
  messages,
  onReply,
}: ActiveRequestsOverlayProps) {
  // Filter at render time — cheap and avoids state-sync bugs that a
  // memoised cached subset would introduce. The expected steady-state
  // is 0–1 active request; even a few dozen is fine to walk every
  // render.
  const active = messages.filter(
    (m) =>
      m.awaiting_reply === true &&
      typeof m.kind === "string" &&
      INTERACTIVE_KINDS.has(m.kind),
  );
  if (active.length === 0) return null;

  return (
    <div
      className="max-h-[50vh] overflow-auto border-t border-border bg-card/95 backdrop-blur-sm"
      // ``role/aria-label`` make the overlay discoverable to screen
      // readers as a distinct region rather than getting concatenated
      // with the timeline above.
      role="region"
      aria-label="Awaiting your response"
    >
      <div className="border-b border-amber-500/30 bg-amber-500/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-amber-300">
        {active.length === 1
          ? "Awaiting your response"
          : `Awaiting your response (${active.length})`}
      </div>
      <div>
        {active.map((msg) => (
          <ChatMessage
            key={`overlay_${msg.id}`}
            message={msg}
            onReply={onReply}
            interactive={true}
          />
        ))}
      </div>
    </div>
  );
}

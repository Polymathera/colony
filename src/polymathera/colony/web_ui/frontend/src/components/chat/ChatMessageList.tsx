import { useEffect, useRef } from "react";
import { ChatMessage, type ChatMessageData } from "./ChatMessage";

interface ChatMessageListProps {
  messages: ChatMessageData[];
  onReply?: (requestId: string, agentId: string, content: string) => void;
  emptyText?: string;
}

export function ChatMessageList({ messages, onReply, emptyText }: ChatMessageListProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive, but only if the user
  // is already near the bottom (within 100px). This prevents yanking the
  // scroll position when the user is reading history.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
    if (isNearBottom) {
      endRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
        {emptyText || "Send a message to begin."}
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full overflow-auto space-y-2 p-3">
      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} onReply={onReply} />
      ))}
      <div ref={endRef} />
    </div>
  );
}

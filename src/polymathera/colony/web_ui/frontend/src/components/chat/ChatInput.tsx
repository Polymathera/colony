import { useState, useRef, useCallback } from "react";
import { Send } from "lucide-react";
import { ChatAutocomplete } from "./ChatAutocomplete";
import {
  getAutocompleteTrigger,
  getAvailableCommands,
  type AutocompleteSuggestion,
} from "./CommandParser";

interface ChatInputProps {
  onSend: (content: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSend, disabled, placeholder }: ChatInputProps) {
  const [value, setValue] = useState("");
  const [suggestions, setSuggestions] = useState<AutocompleteSuggestion[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setValue("");
    setSuggestions([]);
    setShowAutocomplete(false);
    inputRef.current?.focus();
  }, [value, onSend]);

  const updateAutocomplete = useCallback((text: string, cursorPos: number) => {
    const { trigger, query } = getAutocompleteTrigger(text, cursorPos);

    if (!trigger) {
      setShowAutocomplete(false);
      setSuggestions([]);
      return;
    }

    let items: AutocompleteSuggestion[] = [];

    if (trigger === "/") {
      const commands = getAvailableCommands();
      items = Object.entries(commands)
        .filter(([name]) => name.startsWith(query.toLowerCase()))
        .map(([name, info]) => ({
          label: `/${name}`,
          value: `/${name} `,
          type: "command" as const,
          description: info.description,
        }));
    } else if (trigger === "@") {
      // Static suggestions for now — Phase 3+ will populate from live agent list
      items = [
        { label: "@tool:web-search", value: "@tool:web-search ", type: "tool" as const, description: "Web search" },
        { label: "@tool:repl", value: "@tool:repl ", type: "tool" as const, description: "Code execution" },
      ].filter((s) => s.label.toLowerCase().includes(query.toLowerCase()));
    } else if (trigger === "#") {
      const refTypes = [
        { prefix: "repo:", type: "reference" as const, description: "VCM repository" },
        { prefix: "file:", type: "reference" as const, description: "File path" },
        { prefix: "dir:", type: "reference" as const, description: "Directory" },
        { prefix: "lang:", type: "reference" as const, description: "Language filter" },
        { prefix: "page:", type: "reference" as const, description: "VCM page ID" },
      ];
      items = refTypes
        .filter((r) => r.prefix.startsWith(query.toLowerCase()) || query.startsWith(r.prefix))
        .map((r) => ({
          label: `#${r.prefix}`,
          value: `#${r.prefix}`,
          type: r.type,
          description: r.description,
        }));
    }

    setSuggestions(items);
    setSelectedIndex(0);
    setShowAutocomplete(items.length > 0);
  }, []);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    setValue(newValue);
    updateAutocomplete(newValue, e.target.selectionStart || newValue.length);
  }, [updateAutocomplete]);

  const handleSelectSuggestion = useCallback((suggestion: AutocompleteSuggestion) => {
    // Replace from the trigger character to cursor with the suggestion value
    const { trigger } = getAutocompleteTrigger(value, value.length);
    if (!trigger) return;

    // Find the trigger position
    let triggerPos = value.length - 1;
    for (let i = value.length - 1; i >= 0; i--) {
      if (value[i] === trigger) {
        triggerPos = i;
        break;
      }
    }

    const before = value.slice(0, triggerPos);
    const newValue = before + suggestion.value;
    setValue(newValue);
    setShowAutocomplete(false);
    setSuggestions([]);
    inputRef.current?.focus();
  }, [value]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (showAutocomplete && suggestions.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, suggestions.length - 1));
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
        return;
      }
      if (e.key === "Tab" || e.key === "Enter") {
        e.preventDefault();
        handleSelectSuggestion(suggestions[selectedIndex]);
        return;
      }
      if (e.key === "Escape") {
        setShowAutocomplete(false);
        return;
      }
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [showAutocomplete, suggestions, selectedIndex, handleSelectSuggestion, handleSend]);

  return (
    <div className="relative border-t border-border p-3">
      {/* Autocomplete dropdown */}
      <ChatAutocomplete
        suggestions={suggestions}
        selectedIndex={selectedIndex}
        onSelect={handleSelectSuggestion}
        visible={showAutocomplete}
      />

      <div className="flex gap-2">
        <textarea
          ref={inputRef}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || "Type a message, /command, or @agent..."}
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none rounded border border-border bg-background px-3 py-2 text-xs focus:border-primary focus:outline-none disabled:opacity-50"
          style={{ minHeight: "36px", maxHeight: "120px" }}
          onInput={(e) => {
            const target = e.target as HTMLTextAreaElement;
            target.style.height = "auto";
            target.style.height = Math.min(target.scrollHeight, 120) + "px";
          }}
        />
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="shrink-0 rounded bg-primary px-3 py-2 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Send size={14} />
        </button>
      </div>
    </div>
  );
}

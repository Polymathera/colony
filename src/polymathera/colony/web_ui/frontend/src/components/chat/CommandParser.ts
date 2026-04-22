/**
 * Client-side parsing for chat syntax: /commands, @mentions, #references.
 *
 * Parsing is done client-side for autocomplete and pre-validation.
 * The raw content is still sent to the server — the session agent
 * re-parses server-side for authoritative routing.
 */

export interface ParsedCommand {
  name: string;           // "analyze", "map", "abort", etc.
  args: string[];         // positional args
  flags: Record<string, string>;  // --key=value or --key value
}

export interface ParsedMention {
  type: "agent_type" | "agent_id" | "capability" | "tool";
  value: string;          // the name/id after @
  startIndex: number;     // position in the input string
  endIndex: number;
}

export interface ParsedReference {
  type: "repo" | "page" | "file" | "dir" | "lang";
  value: string;          // the value after #type:
  startIndex: number;
  endIndex: number;
}

export interface ParseResult {
  command: ParsedCommand | null;
  mentions: ParsedMention[];
  references: ParsedReference[];
  isCommand: boolean;
}

const COMMANDS: Record<string, { description: string; usage: string }> = {
  analyze:  { description: "Start a structured analysis run", usage: "/analyze <type> [--max-agents N] [--quality N]" },
  map:      { description: "Map content to VCM", usage: "/map <url> [--branch name]" },
  abort:    { description: "Abort the current or specified run", usage: "/abort [run_id]" },
  status:   { description: "Show current run/session status", usage: "/status" },
  agents:   { description: "List active agents in this session", usage: "/agents" },
  set:      { description: "Set a session parameter", usage: "/set <param>=<value>" },
  help:     { description: "Show available commands", usage: "/help [command]" },
  context:  { description: "Show or modify active VCM context", usage: "/context [add|remove] [#ref]" },
};

export function getAvailableCommands(): Record<string, { description: string; usage: string }> {
  return COMMANDS;
}

/**
 * Parse a chat input string for commands, mentions, and references.
 */
export function parseInput(input: string): ParseResult {
  const trimmed = input.trim();
  const result: ParseResult = {
    command: null,
    mentions: [],
    references: [],
    isCommand: false,
  };

  // Parse /command
  if (trimmed.startsWith("/")) {
    result.isCommand = true;
    result.command = parseCommand(trimmed);
  }

  // Parse @mentions
  result.mentions = parseMentions(input);

  // Parse #references
  result.references = parseReferences(input);

  return result;
}

function parseCommand(input: string): ParsedCommand {
  // Split by whitespace, respecting quoted strings
  const tokens = tokenize(input.slice(1)); // remove leading /
  const name = tokens[0] || "";
  const args: string[] = [];
  const flags: Record<string, string> = {};

  for (let i = 1; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.startsWith("--")) {
      const eqIndex = token.indexOf("=");
      if (eqIndex > 0) {
        flags[token.slice(2, eqIndex)] = token.slice(eqIndex + 1);
      } else {
        // Next token is the value, if it exists and isn't a flag
        const nextToken = tokens[i + 1];
        if (nextToken && !nextToken.startsWith("--")) {
          flags[token.slice(2)] = nextToken;
          i++;
        } else {
          flags[token.slice(2)] = "true";
        }
      }
    } else {
      args.push(token);
    }
  }

  return { name, args, flags };
}

function parseMentions(input: string): ParsedMention[] {
  const mentions: ParsedMention[] = [];
  const regex = /@(tool:)?(\w[\w.-]*)/g;
  let match;

  while ((match = regex.exec(input)) !== null) {
    const isTool = !!match[1];
    const value = match[2];
    const startIndex = match.index;
    const endIndex = startIndex + match[0].length;

    let type: ParsedMention["type"];
    if (isTool) {
      type = "tool";
    } else if (value.startsWith("agent_") || /^[a-f0-9]{12,}$/.test(value)) {
      type = "agent_id";
    } else {
      // Could be agent_type or capability — autocomplete will disambiguate
      type = "agent_type";
    }

    mentions.push({ type, value, startIndex, endIndex });
  }

  return mentions;
}

function parseReferences(input: string): ParsedReference[] {
  const refs: ParsedReference[] = [];
  const regex = /#(repo|page|file|dir|lang):([^\s]+)/g;
  let match;

  while ((match = regex.exec(input)) !== null) {
    const type = match[1] as ParsedReference["type"];
    const value = match[2];
    refs.push({
      type,
      value,
      startIndex: match.index,
      endIndex: match.index + match[0].length,
    });
  }

  return refs;
}

function tokenize(input: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let inQuote = false;
  let quoteChar = "";

  for (const ch of input) {
    if (inQuote) {
      if (ch === quoteChar) {
        inQuote = false;
      } else {
        current += ch;
      }
    } else if (ch === '"' || ch === "'") {
      inQuote = true;
      quoteChar = ch;
    } else if (ch === " " || ch === "\t") {
      if (current) {
        tokens.push(current);
        current = "";
      }
    } else {
      current += ch;
    }
  }

  if (current) tokens.push(current);
  return tokens;
}

/**
 * Get autocomplete suggestions for the current cursor position.
 */
export interface AutocompleteSuggestion {
  label: string;
  value: string;
  type: "command" | "agent" | "capability" | "tool" | "reference";
  description?: string;
}

export function getAutocompleteTrigger(input: string, cursorPos: number): { trigger: "@" | "#" | "/" | null; query: string } {
  // Walk backwards from cursor to find trigger character
  const before = input.slice(0, cursorPos);

  // Check for / at start of input
  if (before === "/" || (before.startsWith("/") && !before.includes(" "))) {
    return { trigger: "/", query: before.slice(1) };
  }

  // Check for @ or # trigger
  for (let i = before.length - 1; i >= 0; i--) {
    const ch = before[i];
    if (ch === " " || ch === "\n") break;
    if (ch === "@") return { trigger: "@", query: before.slice(i + 1) };
    if (ch === "#") return { trigger: "#", query: before.slice(i + 1) };
  }

  return { trigger: null, query: "" };
}

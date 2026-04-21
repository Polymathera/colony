import { useState, useRef } from "react";
import { useMapRepo } from "@/api/hooks/useVCM";

interface MapContentDialogProps {
  open: boolean;
  onClose: () => void;
}

const SOURCE_TYPES = [
  { id: "git", label: "Git Repository", description: "Clone and page a git repo", available: true },
  { id: "pdf", label: "PDF Documents", description: "Extract and page PDF content", available: false },
  { id: "papers", label: "Scientific Papers", description: "Parse academic papers", available: false },
  { id: "docs", label: "Documentation", description: "Import documentation sites", available: false },
];

function Tooltip({ text }: { text: string }) {
  return (
    <span className="relative group ml-1 cursor-help">
      <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-muted-foreground/40 text-[9px] text-muted-foreground">
        ?
      </span>
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block w-52 rounded bg-popover border border-border px-2.5 py-1.5 text-[10px] text-popover-foreground shadow-lg z-50 leading-relaxed">
        {text}
      </span>
    </span>
  );
}

export function MapContentDialog({ open, onClose }: MapContentDialogProps) {
  const mapRepo = useMapRepo();
  const [sourceType, setSourceType] = useState("git");
  const [inputMode, setInputMode] = useState<"url" | "upload">("url");
  const [originUrl, setOriginUrl] = useState("");
  const [branch, setBranch] = useState("main");
  const [flushThreshold, setFlushThreshold] = useState(20);
  const [flushTokenBudget, setFlushTokenBudget] = useState(4096);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!open) return null;

  const handleSubmitUrl = () => {
    if (!originUrl.trim()) return;

    mapRepo.mutate({
      origin_url: originUrl.trim(),
      branch,
      flush_threshold: flushThreshold,
      flush_token_budget: flushTokenBudget,
    });
    onClose();
  };

  const handleUpload = () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("flush_threshold", String(flushThreshold));
    formData.append("flush_token_budget", String(flushTokenBudget));

    // Fire and forget — status visible in VCM tab
    fetch("/api/v1/vcm/upload-and-map", {
      method: "POST",
      body: formData,
      credentials: "same-origin",
    });
    onClose();
  };

  const handleSubmit = inputMode === "url" ? handleSubmitUrl : handleUpload;
  const canSubmit = inputMode === "url" ? !!originUrl.trim() : !!selectedFile;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-lg rounded-lg border border-border bg-card shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <h2 className="text-sm font-semibold">Map Content to VCM</h2>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground text-lg leading-none"
          >
            &times;
          </button>
        </div>

        {/* Body */}
        <div className="space-y-4 px-5 py-4">
          {/* Source type selector */}
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">
              Content Source
            </label>
            <div className="grid grid-cols-2 gap-2">
              {SOURCE_TYPES.map((st) => (
                <button
                  key={st.id}
                  onClick={() => st.available && setSourceType(st.id)}
                  disabled={!st.available}
                  className={`rounded border px-3 py-2 text-left transition-colors ${
                    sourceType === st.id
                      ? "border-primary bg-primary/10 text-primary"
                      : st.available
                        ? "border-border text-foreground hover:border-primary/50"
                        : "border-border/50 text-muted-foreground/50 cursor-not-allowed"
                  }`}
                >
                  <div className="text-xs font-medium">{st.label}</div>
                  <div className="text-[10px] text-muted-foreground mt-0.5">
                    {st.available ? st.description : "Coming soon"}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Git-specific fields */}
          {sourceType === "git" && (
            <>
              {/* Input mode toggle: URL vs Upload */}
              <div className="flex rounded border border-border">
                <button
                  className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                    inputMode === "url"
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  onClick={() => setInputMode("url")}
                >
                  Git URL
                </button>
                <button
                  className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                    inputMode === "upload"
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  onClick={() => setInputMode("upload")}
                >
                  Upload Archive
                </button>
              </div>

              {inputMode === "url" ? (
                <div>
                  <label className="block text-xs font-medium text-muted-foreground mb-1">
                    Repository URL
                    <Tooltip text="HTTPS URL of a git repository. The cluster will clone it internally." />
                  </label>
                  <input
                    type="text"
                    value={originUrl}
                    onChange={(e) => setOriginUrl(e.target.value)}
                    placeholder="https://github.com/org/repo"
                    className="w-full rounded border border-border bg-background px-3 py-1.5 text-xs font-mono focus:border-primary focus:outline-none"
                  />
                  <div className="mt-2">
                    <label className="block text-xs font-medium text-muted-foreground mb-1">
                      Branch
                    </label>
                    <input
                      type="text"
                      value={branch}
                      onChange={(e) => setBranch(e.target.value)}
                      className="w-40 rounded border border-border bg-background px-2 py-1.5 text-xs font-mono focus:border-primary focus:outline-none"
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <label className="block text-xs font-medium text-muted-foreground mb-1">
                    Archive File
                    <Tooltip text="Upload a .zip or .tar.gz of your codebase. It will be extracted and mapped in the cluster." />
                  </label>
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full rounded border border-dashed border-border bg-background px-3 py-4 text-center cursor-pointer hover:border-primary/50 transition-colors"
                  >
                    {selectedFile ? (
                      <span className="text-xs text-foreground font-mono">{selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(1)} MB)</span>
                    ) : (
                      <span className="text-xs text-muted-foreground">Click to select .zip or .tar.gz</span>
                    )}
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".zip,.tar.gz,.tgz"
                    className="hidden"
                    onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
                  />
                </div>
              )}

              {/* Paging config */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-muted-foreground mb-1">
                    Flush Threshold
                    <Tooltip text="Number of source files grouped together before creating a VCM page. Lower = more smaller pages. Higher = fewer larger pages." />
                  </label>
                  <input
                    type="number"
                    value={flushThreshold}
                    onChange={(e) => setFlushThreshold(Number(e.target.value))}
                    min={1}
                    className="w-full rounded border border-border bg-background px-2 py-1.5 text-xs font-mono focus:border-primary focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-muted-foreground mb-1">
                    Token Budget
                    <Tooltip text="Maximum tokens per VCM page. When exceeded, a new page starts. Controls how much code each agent sees at once." />
                  </label>
                  <input
                    type="number"
                    value={flushTokenBudget}
                    onChange={(e) => setFlushTokenBudget(Number(e.target.value))}
                    min={512}
                    step={512}
                    className="w-full rounded border border-border bg-background px-2 py-1.5 text-xs font-mono focus:border-primary focus:outline-none"
                  />
                </div>
              </div>
            </>
          )}

        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 border-t border-border px-5 py-3">
          <button
            onClick={onClose}
            className="rounded px-4 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="rounded bg-primary px-4 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
          >
            Map Content
          </button>
        </div>
      </div>
    </div>
  );
}

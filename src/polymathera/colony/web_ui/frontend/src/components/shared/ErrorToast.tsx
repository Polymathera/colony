import { useEffect, useState } from "react";

interface Toast {
  id: number;
  message: string;
}

let _nextId = 0;
let _addToast: ((message: string) => void) | null = null;

/** Call from anywhere to show an error toast. */
export function showErrorToast(message: string) {
  _addToast?.(message);
}

export function ErrorToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    _addToast = (message: string) => {
      const id = _nextId++;
      setToasts((prev) => [...prev, { id, message }]);
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 6000);
    };
    return () => { _addToast = null; };
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-12 right-4 z-50 space-y-2 max-w-sm">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="rounded-lg border border-red-500/30 bg-red-950/90 px-4 py-3 text-xs text-red-200 shadow-lg backdrop-blur animate-in fade-in slide-in-from-bottom-2"
        >
          <div className="flex items-start gap-2">
            <span className="text-red-400 shrink-0">&#x2717;</span>
            <span className="break-words">{t.message}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

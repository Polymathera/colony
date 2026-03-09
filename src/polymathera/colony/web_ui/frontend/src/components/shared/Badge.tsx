import { cn } from "@/lib/utils";

const variants: Record<string, string> = {
  default: "bg-muted text-muted-foreground",
  success: "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/20",
  warning: "bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/20",
  error: "bg-red-500/15 text-red-400 ring-1 ring-red-500/20",
  info: "bg-blue-500/15 text-blue-400 ring-1 ring-blue-500/20",
};

interface BadgeProps {
  children: React.ReactNode;
  variant?: keyof typeof variants;
  className?: string;
}

export function Badge({ children, variant = "default", className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium",
        variants[variant],
        className
      )}
    >
      {children}
    </span>
  );
}

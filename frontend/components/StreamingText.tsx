"use client";

import { cn } from "@/lib/utils";
import type { StreamPhase } from "@/lib/types";

interface StreamingTextProps {
  phase: StreamPhase;
  className?: string;
}

const phaseLabels: Record<StreamPhase, string> = {
  idle: "",
  retrieving: "Retrieving documents",
  reasoning: "Reasoning",
  verifying: "Verifying calculations",
  complete: "Complete",
  error: "Error",
};

export function StreamingText({ phase, className }: StreamingTextProps) {
  if (phase === "idle" || phase === "complete") {
    return null;
  }

  return (
    <div className={cn("flex items-center gap-2 text-sm", className)}>
      {phase !== "error" && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-terminal-green opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-terminal-green"></span>
        </span>
      )}
      <span
        className={cn(
          phase === "error" ? "text-terminal-red" : "text-terminal-dim"
        )}
      >
        {phaseLabels[phase]}
        {phase !== "error" && (
          <span className="inline-block w-4 overflow-hidden align-bottom">
            <span className="animate-ellipsis">...</span>
          </span>
        )}
      </span>
    </div>
  );
}

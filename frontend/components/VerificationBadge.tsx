"use client";

import { CheckCircle, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface VerificationBadgeProps {
  hasErrors: boolean;
  errorCount: number;
}

export function VerificationBadge({
  hasErrors,
  errorCount,
}: VerificationBadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium",
        hasErrors
          ? "bg-terminal-red/10 text-terminal-red border border-terminal-red/30"
          : "bg-terminal-green/10 text-terminal-green border border-terminal-green/30"
      )}
    >
      {hasErrors ? (
        <>
          <AlertTriangle className="w-3.5 h-3.5" />
          <span>
            {errorCount} arithmetic error{errorCount !== 1 ? "s" : ""}
          </span>
        </>
      ) : (
        <>
          <CheckCircle className="w-3.5 h-3.5" />
          <span>Math verified</span>
        </>
      )}
    </div>
  );
}

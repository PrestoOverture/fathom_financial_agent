"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Brain } from "lucide-react";

interface ReasoningTraceProps {
  reasoning: string;
  isStreaming?: boolean;
}

export function ReasoningTrace({
  reasoning,
  isStreaming = false,
}: ReasoningTraceProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!reasoning) return null;

  // Split reasoning into steps (numbered or newline-separated)
  const steps = reasoning
    .split(/\n/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  return (
    <div className="border border-terminal-yellow/30 rounded-md overflow-hidden bg-terminal-yellow/5">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-terminal-yellow hover:bg-terminal-yellow/10 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="w-4 h-4" />
        ) : (
          <ChevronRight className="w-4 h-4" />
        )}
        <Brain className="w-4 h-4" />
        <span>Reasoning trace</span>
        {isStreaming && (
          <span className="ml-auto flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-terminal-yellow opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-terminal-yellow"></span>
            </span>
            <span className="text-xs text-terminal-dim">thinking</span>
          </span>
        )}
      </button>

      {isExpanded && (
        <div className="border-t border-terminal-yellow/20 bg-terminal-bg/50 px-3 py-2">
          <div className="space-y-1 text-sm text-terminal-dim font-mono">
            {steps.map((step, idx) => (
              <p key={idx} className="leading-relaxed">
                {step}
              </p>
            ))}
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-terminal-yellow animate-blink"></span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

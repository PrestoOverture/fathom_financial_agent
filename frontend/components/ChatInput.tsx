"use client";

import { useState, useRef, useEffect, FormEvent, KeyboardEvent } from "react";
import { Send, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { StreamPhase } from "@/lib/types";

interface ChatInputProps {
  onSend: (message: string) => void;
  onCancel: () => void;
  phase: StreamPhase;
  disabled?: boolean;
}

export function ChatInput({
  onSend,
  onCancel,
  phase,
  disabled = false,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isStreaming =
    phase === "retrieving" || phase === "reasoning" || phase === "verifying";

  useEffect(() => {
    // Auto-focus on mount
    textareaRef.current?.focus();
  }, []);

  useEffect(() => {
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled && !isStreaming) {
      onSend(input.trim());
      setInput("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as FormEvent);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="flex items-start gap-2 p-3 rounded-lg bg-terminal-surface border border-terminal-border focus-within:border-terminal-green/50 transition-colors">
        <span className="text-terminal-green font-bold select-none pt-1">
          {">"}
        </span>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a financial question..."
          disabled={disabled || isStreaming}
          rows={1}
          className={cn(
            "flex-1 bg-transparent text-terminal-text placeholder:text-terminal-dim",
            "outline-none resize-none min-h-[24px] max-h-[200px]",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        />

        {isStreaming ? (
          <button
            type="button"
            onClick={onCancel}
            className="p-1.5 rounded-md bg-terminal-red/10 text-terminal-red hover:bg-terminal-red/20 transition-colors"
            title="Cancel"
          >
            <X className="w-4 h-4" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim() || disabled}
            className={cn(
              "p-1.5 rounded-md transition-colors",
              input.trim() && !disabled
                ? "bg-terminal-green/10 text-terminal-green hover:bg-terminal-green/20"
                : "text-terminal-dim cursor-not-allowed"
            )}
            title="Send (Enter)"
          >
            <Send className="w-4 h-4" />
          </button>
        )}
      </div>

      <p className="mt-2 text-xs text-terminal-dim text-center">
        Press <kbd className="px-1 py-0.5 rounded bg-terminal-surface border border-terminal-border">Enter</kbd> to send,{" "}
        <kbd className="px-1 py-0.5 rounded bg-terminal-surface border border-terminal-border">Shift+Enter</kbd> for new line
      </p>
    </form>
  );
}

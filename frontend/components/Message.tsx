"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, AlertCircle } from "lucide-react";
import type { ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";
import { SourcesPanel } from "./SourcesPanel";
import { ReasoningTrace } from "./ReasoningTrace";
import { VerificationBadge } from "./VerificationBadge";

interface MessageProps {
  message: ChatMessage;
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 py-4",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-md bg-terminal-green/10 border border-terminal-green/30 flex items-center justify-center">
          <Bot className="w-4 h-4 text-terminal-green" />
        </div>
      )}

      <div
        className={cn(
          "flex-1 max-w-[85%] space-y-3",
          isUser && "flex flex-col items-end"
        )}
      >
        {isUser ? (
          <div className="px-4 py-2 rounded-lg bg-terminal-cyan/10 border border-terminal-cyan/30 text-terminal-text">
            {message.content}
          </div>
        ) : (
          <>
            {/* Error display */}
            {message.error && (
              <div className="flex items-start gap-2 px-3 py-2 rounded-md bg-terminal-red/10 border border-terminal-red/30 text-terminal-red">
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium">{message.error.type}</p>
                  <p className="text-terminal-red/80">{message.error.message}</p>
                </div>
              </div>
            )}

            {/* Sources panel */}
            {message.chunks && message.sourceCount && (
              <SourcesPanel
                chunks={message.chunks}
                sourceCount={message.sourceCount}
              />
            )}

            {/* Reasoning trace */}
            {message.reasoning && (
              <ReasoningTrace
                reasoning={message.reasoning}
                isStreaming={message.isStreaming && !message.answer}
              />
            )}

            {/* Answer content */}
            {message.answer && (
              <div className="prose prose-invert prose-terminal max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.answer}
                </ReactMarkdown>
              </div>
            )}

            {/* Verification badge */}
            {message.verification && (
              <VerificationBadge
                hasErrors={message.verification.hasErrors}
                errorCount={message.verification.errorCount}
              />
            )}
          </>
        )}
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-md bg-terminal-cyan/10 border border-terminal-cyan/30 flex items-center justify-center">
          <User className="w-4 h-4 text-terminal-cyan" />
        </div>
      )}
    </div>
  );
}

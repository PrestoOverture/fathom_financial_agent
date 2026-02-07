"use client";

import { useRef, useEffect } from "react";
import { Trash2 } from "lucide-react";
import { useSSEStream } from "@/hooks/useSSEStream";
import { Message } from "./Message";
import { ChatInput } from "./ChatInput";
import { StreamingText } from "./StreamingText";

export function Chat() {
  const { messages, currentPhase, sendMessage, cancelStream, clearMessages } =
    useSSEStream();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom on new messages
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentPhase]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-terminal-border bg-terminal-surface/50">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold text-terminal-text">
            Fathom Financial Agent
          </h1>
        </div>

        {messages.length > 0 && (
          <button
            onClick={clearMessages}
            className="p-2 rounded-md text-terminal-dim hover:text-terminal-red hover:bg-terminal-red/10 transition-colors"
            title="Clear chat"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </header>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="max-w-md space-y-4">
              <h2 className="text-xl font-semibold text-terminal-text">
                Financial Intelligence Agent
              </h2>
              <p className="text-terminal-dim">
                Ask questions about 10-K reports and financial data. The agent
                uses RAG to retrieve relevant documents, reasons through the
                problem step-by-step, and verifies all arithmetic calculations.
              </p>
              <div className="grid gap-2 text-sm">
                <p className="text-terminal-dim">Try asking:</p>
                <button
                  onClick={() =>
                    sendMessage(
                      "What percent of Ulta Beauty's total spend on stock repurchases for FY 2023 occurred in Q4 of FY2023?"
                    )
                  }
                  className="px-3 py-2 rounded-md bg-terminal-surface border border-terminal-border text-terminal-blue hover:border-terminal-blue/50 transition-colors text-left"
                >
                  What percent of Ulta Beauty&apos;s total spend on stock repurchases for FY 2023 occurred in Q4 of FY2023?
                </button>
                <button
                  onClick={() =>
                    sendMessage(
                      "What percentage of Amazon's total revenue came from AWS in 2023?"
                    )
                  }
                  className="px-3 py-2 rounded-md bg-terminal-surface border border-terminal-border text-terminal-blue hover:border-terminal-blue/50 transition-colors text-left"
                >
                  What percentage of Amazon&apos;s total revenue came from AWS in 2023?
                </button>
                <button
                  onClick={() =>
                    sendMessage(
                      "What is the FY2018 - FY2020 3 year average of capex as a % of revenue for MGM Resorts?"
                    )
                  }
                  className="px-3 py-2 rounded-md bg-terminal-surface border border-terminal-border text-terminal-blue hover:border-terminal-blue/50 transition-colors text-left"
                >
                  What is the FY2018 - FY2020 3 year average of capex as a % of revenue for MGM Resorts?
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="py-4 space-y-2">
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}

            {/* Streaming phase indicator */}
            {currentPhase !== "idle" && currentPhase !== "complete" && (
              <div className="py-2 pl-11">
                <StreamingText phase={currentPhase} />
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="px-4 py-4 border-t border-terminal-border bg-terminal-surface/30">
        <ChatInput
          onSend={sendMessage}
          onCancel={cancelStream}
          phase={currentPhase}
        />
      </div>
    </div>
  );
}

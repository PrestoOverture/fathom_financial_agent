"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { SSEClient } from "@/lib/sse-client";
import type { ChatMessage, StreamPhase, SSECallbacks } from "@/lib/types";
import { generateId } from "@/lib/utils";

const TOKEN_RENDER_INTERVAL_MS = 25; // Smooth typewriter cadence

// Extract only the reasoning portion (before "Answer:")
function extractReasoningOnly(text: string): string {
  // Find where "Answer:" starts (case-insensitive)
  const answerMatch = text.match(/\n\s*Answer\s*:/i);
  if (answerMatch && answerMatch.index !== undefined) {
    return text.slice(0, answerMatch.index).trim();
  }
  // Also handle if "Reasoning:" label is present at the start
  let result = text;
  const reasoningMatch = result.match(/^Reasoning\s*:\s*/i);
  if (reasoningMatch) {
    result = result.slice(reasoningMatch[0].length);
  }
  return result.trim();
}

interface UseSSEStreamReturn {
  messages: ChatMessage[];
  currentPhase: StreamPhase;
  sendMessage: (question: string) => Promise<void>;
  cancelStream: () => void;
  clearMessages: () => void;
}

export function useSSEStream(): UseSSEStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentPhase, setCurrentPhase] = useState<StreamPhase>("idle");
  const clientRef = useRef<SSEClient | null>(null);

  // Token buffer for smooth streaming
  const tokenQueueRef = useRef<string[]>([]);
  const displayedReasoningRef = useRef<string>("");
  const fullTextRef = useRef<string>(""); // Full raw text including Answer
  const finalReasoningLogsRef = useRef<string>(""); // Parsed reasoning_logs from backend
  const renderLoopRef = useRef<number | null>(null);
  const lastRenderTimeRef = useRef<number>(0);
  const currentAssistantIdRef = useRef<string | null>(null);

  // Cleanup render loop
  const stopRenderLoop = useCallback(() => {
    if (renderLoopRef.current !== null) {
      cancelAnimationFrame(renderLoopRef.current);
      renderLoopRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRenderLoop();
    };
  }, [stopRenderLoop]);

  const sendMessage = useCallback(
    async (question: string) => {
      // Add user message
      const userMessage: ChatMessage = {
        id: generateId(),
        role: "user",
        content: question,
        timestamp: new Date(),
      };

      // Create assistant message placeholder
      const assistantId = generateId();
      currentAssistantIdRef.current = assistantId;
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setCurrentPhase("retrieving");

      // Reset token buffer state
      tokenQueueRef.current = [];
      displayedReasoningRef.current = "";
      fullTextRef.current = "";
      finalReasoningLogsRef.current = "";
      lastRenderTimeRef.current = 0;

      // Create SSE client and callbacks
      const client = new SSEClient();
      clientRef.current = client;

      const updateAssistant = (updates: Partial<ChatMessage>) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, ...updates } : m))
        );
      };

      // Render loop using requestAnimationFrame for smooth updates
      const renderLoop = (timestamp: number) => {
        // Throttle to TOKEN_RENDER_INTERVAL_MS
        if (timestamp - lastRenderTimeRef.current >= TOKEN_RENDER_INTERVAL_MS) {
          if (tokenQueueRef.current.length > 0) {
            // Take 1-3 tokens per frame for natural variation
            const tokensToTake = Math.min(
              tokenQueueRef.current.length,
              Math.ceil(Math.random() * 2) + 1
            );
            const tokens = tokenQueueRef.current.splice(0, tokensToTake);
            displayedReasoningRef.current += tokens.join("");

            // Only show reasoning portion (filter out Answer:)
            const reasoningOnly = extractReasoningOnly(
              displayedReasoningRef.current
            );
            updateAssistant({
              reasoning: reasoningOnly,
            });
          }
          lastRenderTimeRef.current = timestamp;
        }

        // Continue loop if streaming
        if (clientRef.current !== null || tokenQueueRef.current.length > 0) {
          renderLoopRef.current = requestAnimationFrame(renderLoop);
        }
      };

      const callbacks: SSECallbacks = {
        onGraphStart: () => {
          setCurrentPhase("retrieving");
        },

        onRetrieveUpdate: (event) => {
          updateAssistant({
            chunks: event.chunks,
            sourceCount: event.source_count,
          });
          setCurrentPhase("reasoning");
          // Start the render loop when reasoning begins
          lastRenderTimeRef.current = 0;
          renderLoopRef.current = requestAnimationFrame(renderLoop);
        },

        onReasoningDelta: (event) => {
          // Only push to queue - render loop handles display
          tokenQueueRef.current.push(event.delta);
          fullTextRef.current += event.delta;
        },

        onReasonUpdate: (event) => {
          // Stop render loop and use parsed reasoning_logs from backend
          stopRenderLoop();
          tokenQueueRef.current = [];
          finalReasoningLogsRef.current = event.reasoning_logs;
          updateAssistant({
            reasoning: event.reasoning_logs,
            answer: event.answer,
          });
          setCurrentPhase("verifying");
        },

        onVerifyUpdate: (event) => {
          updateAssistant({
            verification: {
              hasErrors: event.arithmetic_errors_found,
              errorCount: event.error_count,
            },
          });
        },

        onGraphComplete: (event) => {
          stopRenderLoop();
          updateAssistant({
            content: event.answer,
            answer: event.answer,
            reasoning: finalReasoningLogsRef.current,
            isStreaming: false,
          });
          setCurrentPhase("complete");
          currentAssistantIdRef.current = null;
        },

        onError: (event) => {
          stopRenderLoop();
          updateAssistant({
            error: {
              type: event.error_type,
              message: event.message,
            },
            isStreaming: false,
          });
          setCurrentPhase("error");
          currentAssistantIdRef.current = null;
        },
      };

      try {
        await client.stream(question, callbacks);
      } catch (error) {
        stopRenderLoop();
        const errorMessage =
          error instanceof Error ? error.message : "Unknown error";
        updateAssistant({
          error: {
            type: "NetworkError",
            message: errorMessage,
          },
          isStreaming: false,
        });
        setCurrentPhase("error");
        currentAssistantIdRef.current = null;
      } finally {
        clientRef.current = null;
      }
    },
    [stopRenderLoop]
  );

  const cancelStream = useCallback(() => {
    clientRef.current?.abort();
    stopRenderLoop();
    // On cancel, show whatever reasoning we have (filtered)
    const reasoningOnly = extractReasoningOnly(fullTextRef.current);
    setMessages((prev) =>
      prev.map((m) =>
        m.isStreaming ? { ...m, isStreaming: false, reasoning: reasoningOnly } : m
      )
    );
    setCurrentPhase("idle");
    currentAssistantIdRef.current = null;
  }, [stopRenderLoop]);

  const clearMessages = useCallback(() => {
    stopRenderLoop();
    tokenQueueRef.current = [];
    displayedReasoningRef.current = "";
    fullTextRef.current = "";
    finalReasoningLogsRef.current = "";
    setMessages([]);
    setCurrentPhase("idle");
    currentAssistantIdRef.current = null;
  }, [stopRenderLoop]);

  return {
    messages,
    currentPhase,
    sendMessage,
    cancelStream,
    clearMessages,
  };
}

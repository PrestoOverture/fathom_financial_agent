import type { SSEEvent, SSECallbacks } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class SSEClient {
  private abortController: AbortController | null = null;

  async stream(question: string, callbacks: SSECallbacks): Promise<void> {
    this.abortController = new AbortController();

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        let currentEventType: string | null = null;

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith("data: ") && currentEventType) {
            const jsonStr = line.slice(6);
            try {
              const event = JSON.parse(jsonStr) as SSEEvent;
              this.dispatchEvent(event, callbacks);
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
            currentEventType = null;
          }
        }
      }
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        // Stream was cancelled, not an error
        return;
      }
      throw error;
    }
  }

  private dispatchEvent(event: SSEEvent, callbacks: SSECallbacks): void {
    switch (event.event_type) {
      case "graph_start":
        callbacks.onGraphStart?.(event);
        break;
      case "retrieve_update":
        callbacks.onRetrieveUpdate?.(event);
        break;
      case "reasoning_delta":
        callbacks.onReasoningDelta?.(event);
        break;
      case "reason_update":
        callbacks.onReasonUpdate?.(event);
        break;
      case "verify_update":
        callbacks.onVerifyUpdate?.(event);
        break;
      case "graph_complete":
        callbacks.onGraphComplete?.(event);
        break;
      case "error":
        callbacks.onError?.(event);
        break;
    }
  }

  abort(): void {
    this.abortController?.abort();
    this.abortController = null;
  }
}

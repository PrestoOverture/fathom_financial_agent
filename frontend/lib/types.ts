// SSE Event Types from backend (api/sse.py)

export interface BaseEvent {
  event_type: string;
  timestamp: string;
  request_id: string;
}

export interface GraphStartEvent extends BaseEvent {
  event_type: "graph_start";
  question: string;
}

// Retrieved chunk with content
export interface RetrievedChunk {
  doc_name: string;
  content: string;
  score: number;
}

export interface RetrieveUpdateEvent extends BaseEvent {
  event_type: "retrieve_update";
  source_count: number;
  chunks: RetrievedChunk[];
}

export interface ReasoningDeltaEvent extends BaseEvent {
  event_type: "reasoning_delta";
  delta: string;
}

export interface ReasonUpdateEvent extends BaseEvent {
  event_type: "reason_update";
  answer: string;
  reasoning_logs: string;
}

export interface VerifyUpdateEvent extends BaseEvent {
  event_type: "verify_update";
  arithmetic_errors_found: boolean;
  error_count: number;
}

export interface GraphCompleteEvent extends BaseEvent {
  event_type: "graph_complete";
  answer: string;
  has_errors: boolean;
}

export interface ErrorEvent extends BaseEvent {
  event_type: "error";
  error_type: string;
  message: string;
}

export type SSEEvent =
  | GraphStartEvent
  | RetrieveUpdateEvent
  | ReasoningDeltaEvent
  | ReasonUpdateEvent
  | VerifyUpdateEvent
  | GraphCompleteEvent
  | ErrorEvent;

// Stream phases for UI state
export type StreamPhase =
  | "idle"
  | "retrieving"
  | "reasoning"
  | "verifying"
  | "complete"
  | "error";

// Chat message state
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  // Assistant-specific fields
  chunks?: RetrievedChunk[];
  sourceCount?: number;
  reasoning?: string;
  answer?: string;
  verification?: {
    hasErrors: boolean;
    errorCount: number;
  };
  isStreaming?: boolean;
  error?: {
    type: string;
    message: string;
  };
}

// SSE Client callback types
export interface SSECallbacks {
  onGraphStart?: (event: GraphStartEvent) => void;
  onRetrieveUpdate?: (event: RetrieveUpdateEvent) => void;
  onReasoningDelta?: (event: ReasoningDeltaEvent) => void;
  onReasonUpdate?: (event: ReasonUpdateEvent) => void;
  onVerifyUpdate?: (event: VerifyUpdateEvent) => void;
  onGraphComplete?: (event: GraphCompleteEvent) => void;
  onError?: (event: ErrorEvent) => void;
}

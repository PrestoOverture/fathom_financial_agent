"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronDown, ChevronRight, FileText, Database } from "lucide-react";
import type { RetrievedChunk } from "@/lib/types";

interface SourcesPanelProps {
  chunks: RetrievedChunk[];
  sourceCount: number;
}

export function SourcesPanel({ chunks, sourceCount }: SourcesPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set());

  if (chunks.length === 0) return null;

  // Get unique document names
  const uniqueDocs = Array.from(new Set(chunks.map((c) => c.doc_name)));

  const toggleChunk = (index: number) => {
    setExpandedChunks((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  return (
    <div className="border border-terminal-border rounded-md overflow-hidden">
      {/* Main header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-terminal-blue hover:bg-terminal-surface transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="w-4 h-4" />
        ) : (
          <ChevronRight className="w-4 h-4" />
        )}
        <Database className="w-4 h-4" />
        <span>
          {sourceCount} chunk{sourceCount !== 1 ? "s" : ""} from{" "}
          {uniqueDocs.length} document{uniqueDocs.length !== 1 ? "s" : ""}
        </span>
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="border-t border-terminal-border bg-terminal-bg">
          {chunks.map((chunk, idx) => (
            <div
              key={idx}
              className="border-b border-terminal-border last:border-b-0"
            >
              {/* Chunk header */}
              <button
                onClick={() => toggleChunk(idx)}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-terminal-surface/50 transition-colors"
              >
                {expandedChunks.has(idx) ? (
                  <ChevronDown className="w-3 h-3 text-terminal-dim" />
                ) : (
                  <ChevronRight className="w-3 h-3 text-terminal-dim" />
                )}
                <FileText className="w-3 h-3 text-terminal-blue" />
                <span className="text-terminal-text truncate flex-1 text-left">
                  {chunk.doc_name}
                </span>
                <span className="text-terminal-dim text-xs">
                  {(chunk.score * 100).toFixed(1)}% match
                </span>
              </button>

              {/* Chunk content */}
              {expandedChunks.has(idx) && (
                <div className="px-3 py-2 bg-terminal-surface/30 border-t border-terminal-border/50">
                  <div className="prose prose-invert prose-sm prose-ledger max-w-none text-terminal-dim">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {chunk.content}
                    </ReactMarkdown>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        terminal: {
          bg: "#0a0e14",
          surface: "#0d1117",
          border: "#21262d",
          text: "#e6edf3",
          dim: "#7d8590",
          green: "#3fb950",
          yellow: "#d29922",
          red: "#f85149",
          cyan: "#39c5cf",
          blue: "#58a6ff",
        },
      },
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
      animation: {
        blink: "blink 1s step-end infinite",
        ellipsis: "ellipsis 1.5s steps(4, end) infinite",
      },
      keyframes: {
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        ellipsis: {
          "0%": { width: "0" },
          "100%": { width: "1em" },
        },
      },
      typography: {
        terminal: {
          css: {
            "--tw-prose-body": "#e6edf3",
            "--tw-prose-headings": "#e6edf3",
            "--tw-prose-links": "#58a6ff",
            "--tw-prose-bold": "#e6edf3",
            "--tw-prose-code": "#3fb950",
            "--tw-prose-pre-bg": "#0d1117",
            "--tw-prose-pre-code": "#e6edf3",
            "--tw-prose-th-borders": "#21262d",
            "--tw-prose-td-borders": "#21262d",
            "code::before": { content: '""' },
            "code::after": { content: '""' },
            code: {
              backgroundColor: "#21262d",
              padding: "0.125rem 0.375rem",
              borderRadius: "0.25rem",
              fontWeight: "400",
            },
            "thead th": {
              borderBottomColor: "#21262d",
            },
            "tbody td, tfoot td": {
              borderBottomColor: "#21262d",
            },
          },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
export default config;

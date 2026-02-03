import { Chat } from "@/components/Chat";

export default function Home() {
  return (
    <main className="h-screen bg-terminal-bg">
      <div className="h-full max-w-4xl mx-auto">
        <Chat />
      </div>
    </main>
  );
}

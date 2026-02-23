"use client";

import { useEffect, useMemo, useState } from "react";

type ChatResponse = {
  answer: string;
  citations?: Array<{ source: string; excerpt: string; score: number }>;
};

const STORAGE_KEY = "persona_chat_messages_v1";
const USER_AVATAR = "/avatars/me.jpg";          // 你的头像（放到 public/avatars/me.jpg）
const ASSIST_AVATAR = "/avatars/assistant.jpg";  // 助手头像（放到 public/avatars/assistant.jpg）


export default function HomePage() {
  const apiBase = useMemo(() => {
    return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
  }, []);

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; text: string; citations?: any[]}>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);



  // 1) 初次加载：从 localStorage 恢复聊天记录
  useEffect(() => {
    try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) setMessages(parsed);
    }
  } catch {
    // ignore
  }
}, []);
  // 2) messages 变化：写回 localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // 忽略写入失败（比如浏览器禁用存储）
    }
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || loading) return;

    setError(null);
    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as ChatResponse;

      setMessages((m) => [...m, { role: "assistant", text: data.answer, citations: data.citations ?? [] }]);

    } catch (e: any) {
      setError(e?.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  }

  function clearChat() {
    setMessages([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}
  }

function readAsDataURL(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}




  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, textAlign: "center" }}>本地文档问答系统</h1>
      <p style={{ marginTop: 8, opacity: 0.8 }}>
        
      </p>

      <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
        <button
          onClick={clearChat}
          style={{ padding: "8px 12px", borderRadius: 10, border: "1px solid #ccc", background: "#fff" }}
        >
          清空聊天
        </button>
      </div>

      <div style={{ marginTop: 16, border: "1px solid #ddd", borderRadius: 12, padding: 16 }}>
        <div style={{ minHeight: 240 }}>
          {messages.length === 0 ? (
            <div style={{ opacity: 0.7 }}>在下面输入问题，比如：你做过哪些项目？</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

              {messages.map((m, idx) => {
  const isUser = m.role === "user";
  const avatar = isUser ? USER_AVATAR : ASSIST_AVATAR;

  return (
    <div
      key={idx}
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: isUser ? "row-reverse" : "row",
          alignItems: "flex-start",
          gap: 10,
          maxWidth: "85%",
        }}
      >
        <img
          src={avatar}
          alt={isUser ? "you" : "assistant"}
          style={{
            width: 34,
            height: 34,
            borderRadius: "50%",
            objectFit: "cover",
            border: "1px solid #ddd",
            flexShrink: 0,
          }}
        />

        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {/* 名字行：左右对齐 */}
          <div
            style={{
              fontSize: 12,
              opacity: 0.7,
              textAlign: isUser ? "right" : "left",
            }}
          >
            {isUser ? "YOU" : "Assistant"}
          </div>

          {/* 气泡 */}
          <div
            style={{
              whiteSpace: "pre-wrap",
              padding: "10px 12px",
              borderRadius: 14,
              border: "1px solid #e5e5e5",
              background: isUser ? "#f6f6f6" : "#fff",
              lineHeight: 1.6,
            }}
          >
            {m.text}
          </div>

          {/* 引用证据（只在 assistant 显示） */}
          {m.role === "assistant" && m.citations?.length ? (
            <div style={{ marginTop: 6, paddingLeft: 12, borderLeft: "3px solid #ddd" }}>
              <div style={{ fontSize: 13, fontWeight: 700, opacity: 0.85 }}>引用证据</div>

              <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }}>
                {m.citations.map((c, i) => (
                  <div key={i} style={{ fontSize: 13, opacity: 0.9 }}>
                    <div>
                      <b>{c.source}</b>{" "}
                      <span style={{ opacity: 0.7 }}>score={Number(c.score).toFixed(3)}</span>
                    </div>
                    <div style={{ opacity: 0.85 }}>{c.excerpt}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
})}


            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") send();
            }}
            placeholder="输入问题…"
            style={{ flex: 1, border: "1px solid #ccc", borderRadius: 10, padding: "10px 12px" }}
          />
          <button
            onClick={send}
            disabled={loading}
            style={{
              padding: "10px 14px",
              borderRadius: 10,
              border: "1px solid #ccc",
              background: loading ? "#eee" : "#fff",
            }}
          >
            {loading ? "发送中…" : "发送"}
          </button>
        </div>

        {error && <div style={{ color: "crimson", marginTop: 10 }}>Error: {error}</div>}
      </div>
    </div>
  );
}

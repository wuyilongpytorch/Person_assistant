"use client";

import { useEffect, useMemo, useState } from "react";

type FileItem = { name: string; size: number; mtime: number };

export default function Sidebar() {
  const apiBase = useMemo(() => process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000", []);

  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [indexStatus, setIndexStatus] = useState<string>("");

  // ✅ 新增：可拉伸宽度
  const [width, setWidth] = useState<number>(320);
  const MIN_W = 240;
  const MAX_W = 520;

  async function fetchFiles() {
    try {
      const res = await fetch(`${apiBase}/files`);
      const data = await res.json();
      setFiles(data.files ?? []);
    } catch {
      // ignore
    }
  }

  async function fetchIndexStatus() {
    try {
      const res = await fetch(`${apiBase}/index/status`);
      const data = await res.json();
      setIndexStatus(`files=${data.files}, chunks=${data.chunks}`);
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    fetchFiles();
    fetchIndexStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function upload() {
    if (!file) return;
    setStatus("上传中…");

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${apiBase}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setStatus(`✅ 上传成功：${data.filename}`);
      setFile(null);

      await fetchFiles();
      await fetchIndexStatus();
    } catch (e: any) {
      setStatus(`❌ 上传失败：${e?.message ?? "unknown error"}`);
    }
  }

  async function rebuildIndex() {
    setIndexStatus("重建索引中…");
    try {
      const res = await fetch(`${apiBase}/index/rebuild`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setIndexStatus(`✅ index rebuilt: files=${data.files}, chunks=${data.chunks}`);
    } catch (e: any) {
      setIndexStatus(`❌ 重建失败：${e?.message ?? "unknown error"}`);
    }
  }

  // ✅ 新增：拖拽逻辑
  function startResize(e: React.MouseEvent<HTMLDivElement>) {
    e.preventDefault();

    const startX = e.clientX;
    const startW = width;

    function onMove(ev: MouseEvent) {
      const next = Math.min(MAX_W, Math.max(MIN_W, startW + (ev.clientX - startX)));
      setWidth(next);
    }

    function onUp() {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }

  return (
    <aside
      style={{
        width,
        borderRight: "1px solid #eee",
        padding: 16,
        height: "100vh",
        position: "sticky",
        top: 0,
        boxSizing: "border-box",
      }}
    >
      {/* ✅ 拖拽条：放在右侧边缘 */}
      <div
        onMouseDown={startResize}
        style={{
          position: "absolute",
          top: 0,
          right: -4,
          width: 8,
          height: "100%",
          cursor: "col-resize",
          // 可选：让拖拽区域更容易点到，但视觉不明显
          background: "transparent",
        }}
        aria-label="Resize sidebar"
      />

      <div style={{ fontWeight: 800, fontSize: 18 }}>菜单</div>

      <nav style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 8 }}>
        <a href="/" style={{ textDecoration: "underline" }}>
          首页
        </a>
      </nav>

      <div style={{ marginTop: 18, paddingTop: 16, borderTop: "1px solid #eee" }}>
        <div style={{ fontWeight: 700 }}>上传资料</div>
        <div style={{ marginTop: 8, fontSize: 13, opacity: 0.8 }}>支持： md / txt</div>

        <input
          id="file-input"
          type="file"
          accept=".pdf,.md,.txt"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          style={{ display: "none" }}
        />

        <label
          htmlFor="file-input"
          style={{
            marginTop: 10,
            width: "100%",
            padding: "10px 12px",
            borderRadius: 10,
            border: "1px solid #ccc",
            background: "#fff",
            display: "block",
            textAlign: "center",
            cursor: "pointer",
            userSelect: "none",
            boxSizing: "border-box",
          }}
        >
          {file ? `已选择：${file.name}` : "选择文件"}
        </label>

        <button
          onClick={upload}
          disabled={!file}
          style={{
            marginTop: 10,
            width: "100%",
            padding: "10px 12px",
            borderRadius: 10,
            border: "1px solid #ccc",
            background: file ? "#fff" : "#eee",
            boxSizing: "border-box",
          }}
        >
          上传
        </button>

        {status && <div style={{ marginTop: 10, fontSize: 13 }}>{status}</div>}
      </div>

      <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid #eee" }}>
        <div style={{ fontWeight: 700 }}>文件列表</div>

        <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
          <button
            onClick={fetchFiles}
            style={{ flex: 1, padding: "8px 10px", borderRadius: 10, border: "1px solid #ccc", background: "#fff" }}
          >
            刷新
          </button>
          <button
            onClick={rebuildIndex}
            style={{ flex: 1, padding: "8px 10px", borderRadius: 10, border: "1px solid #ccc", background: "#fff" }}
          >
            重建索引
          </button>
        </div>

        <div style={{ marginTop: 10, fontSize: 13, opacity: 0.85 }}>索引状态：{indexStatus || "—"}</div>

        <div style={{ marginTop: 10, maxHeight: 220, overflow: "auto", border: "1px solid #eee", borderRadius: 10, padding: 10 }}>
          {files.length === 0 ? (
            <div style={{ opacity: 0.7, fontSize: 13 }}>暂无文件</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 13, lineHeight: 1.6 }}>
              {files.map((f) => (
                <li key={f.name}>
                  {f.name} <span style={{ opacity: 0.7 }}>({Math.round(f.size / 1024)} KB)</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </aside>
  );
}

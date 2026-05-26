import { useEffect, useRef, useState } from "react";
import { fetchModels, streamChat } from "../api";

export function Chat() {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [input, setInput] = useState("");
  const [prompt, setPrompt] = useState("");
  const [reply, setReply] = useState("");
  const [tps, setTps] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchModels()
      .then((ms) => {
        setModels(ms.map((m) => m.id));
        if (ms[0]) setModel(ms[0].id);
      })
      .catch((e) => setError(`could not load models: ${e}`));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [reply, prompt]);

  async function send() {
    if (!input.trim() || busy) return;
    setPrompt(input);
    setReply("");
    setTps(null);
    setError("");
    setBusy(true);
    setInput("");
    let count = 0;
    const t0 = performance.now();
    try {
      await streamChat(model, input, (tok) => {
        count += 1;
        setReply((r) => r + tok);
      });
      const secs = (performance.now() - t0) / 1000;
      if (secs > 0) setTps(count / secs);
    } catch (e) {
      setError(`request failed: ${e}`);
    } finally {
      setBusy(false);
    }
  }

  function onKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <>
      <div className="section-head">
        <h2>Inference Console</h2>
        <span className="rule" />
        <span className="meta">/v1/chat/completions · stream</span>
      </div>

      <div className="term fade-up">
        <div className="term-bar">
          <span className="lights">
            <i /> <i /> <i />
          </span>
          <span className="term-title">session://chat</span>
          <select
            className="term-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {models.length === 0 && <option value="">no models</option>}
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="transcript" ref={scrollRef}>
          {!prompt && !reply && !error && (
            <div className="empty">
              <span>// awaiting input</span>
              <span>// pick a model, type a prompt, ⌘/Ctrl+Enter to send</span>
            </div>
          )}
          {prompt && <div className="line-user">{prompt}</div>}
          {(reply || busy) && (
            <div className="line-asst">
              {reply}
              {busy && <span className="caret" />}
            </div>
          )}
          {error && <div className="errline">⚠ {error}</div>}
        </div>

        <div className="composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            rows={2}
            placeholder="message the model…"
          />
          <div style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-end" }}>
            <button className="send" onClick={send} disabled={busy || !model}>
              {busy ? "···" : "Send"}
            </button>
            {tps !== null && (
              <span className="readout">
                <b>{tps.toFixed(1)}</b> tok/s
              </span>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

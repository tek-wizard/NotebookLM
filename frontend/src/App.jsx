import { startTransition, useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

const promptSuggestions = [
  "Give me a crisp summary of this resource.",
  "List the most important concepts in bullet points.",
  "Pull out any action items or decisions.",
  "Explain the resource like I'm new to the topic.",
];

const timeFormatter = new Intl.DateTimeFormat([], {
  hour: "numeric",
  minute: "2-digit",
});

// Friendly labels for the Advanced RAG stages reported in the query `trace`.
const STAGE_LABELS = {
  query_rewrite: "Query rewrite",
  subquery: "Sub-queries",
  hyde: "HyDE",
  rerank: "Re-rank",
  corrective_rag: "Corrective RAG",
  llm_judge: "LLM judge",
  full_context: "Full context",
  small_talk: "Small talk",
};

function buildMessage(role, content, trace = null) {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    content,
    trace,
    timestamp: Date.now(),
  };
}

function buildWelcomeMessage(resourceLabel) {
  return buildMessage(
    "assistant",
    resourceLabel
      ? `Your resource "${resourceLabel}" is indexed. Ask anything about it and I will answer from that material.`
      : "Drop a PDF or TXT file, or paste raw text to create a grounded chat session.",
  );
}

function formatTime(timestamp) {
  return timeFormatter.format(timestamp);
}

function formatSessionId(sessionId) {
  if (!sessionId) {
    return "No active session";
  }

  return `${sessionId.slice(0, 8)}...${sessionId.slice(-4)}`;
}

function formatFileSize(bytes) {
  if (!bytes && bytes !== 0) {
    return "";
  }

  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isSupportedFile(file) {
  return Boolean(file?.name?.toLowerCase().match(/\.(pdf|txt)$/));
}

async function requestApi(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);

  if (!response.ok) {
    let detail = "Request failed";

    try {
      const errorPayload = await response.json();
      detail = errorPayload.detail || errorPayload.message || detail;
    } catch {
      detail = response.statusText || detail;
    }

    throw new Error(detail);
  }

  return response.json();
}

async function destroySession(sessionId) {
  if (!sessionId) {
    return;
  }

  try {
    await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
      method: "DELETE",
    });
  } catch {
    // Cleanup is best-effort. The UI should not block on it.
  }
}

function PipelineTrace({ trace }) {
  // De-duplicate the ordered stage list while keeping first-seen order.
  const stages = [...new Set(trace?.stages ?? [])].filter(
    (stage) => STAGE_LABELS[stage],
  );

  if (stages.length === 0) {
    return null;
  }

  return (
    <div className="pipeline-trace" aria-label="Advanced RAG pipeline stages">
      {stages.map((stage) => (
        <span key={stage} className="pipeline-chip">
          {STAGE_LABELS[stage]}
        </span>
      ))}
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      <article className={`message-bubble ${isUser ? "user" : "assistant"}`}>
        <p>{message.content}</p>
        {!isUser && message.trace ? <PipelineTrace trace={message.trace} /> : null}
        <span>{formatTime(message.timestamp)}</span>
      </article>
    </div>
  );
}

function App() {
  const [sessionId, setSessionId] = useState("");
  const [resourceMeta, setResourceMeta] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [textResource, setTextResource] = useState("");
  const [uploadMode, setUploadMode] = useState("file");
  const [composerValue, setComposerValue] = useState("");
  const [messages, setMessages] = useState(() => [buildWelcomeMessage("")]);
  const [uploadError, setUploadError] = useState("");
  const [chatError, setChatError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isUploadingFile, setIsUploadingFile] = useState(false);
  const [isUploadingText, setIsUploadingText] = useState(false);
  const [isSending, setIsSending] = useState(false);

  const fileInputRef = useRef(null);
  const threadEndRef = useRef(null);

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  function primeFile(file) {
    if (!file) {
      return;
    }

    if (!isSupportedFile(file)) {
      setUploadError("Only PDF and TXT files are supported.");
      return;
    }

    setUploadError("");
    setSelectedFile(file);
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function handleFileInputChange(event) {
    primeFile(event.target.files?.[0]);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    primeFile(event.dataTransfer.files?.[0]);
  }

  function handleDragOver(event) {
    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setIsDragging(false);
    }
  }

  async function activateResource(nextSessionId, nextResourceMeta, cleanupSessionId) {
    startTransition(() => {
      setSessionId(nextSessionId);
      setResourceMeta(nextResourceMeta);
      setMessages([buildWelcomeMessage(nextResourceMeta.label)]);
      setComposerValue("");
      setChatError("");
    });

    await destroySession(cleanupSessionId);
  }

  async function uploadFile() {
    if (!selectedFile || isUploadingFile) {
      return;
    }

    if (!isSupportedFile(selectedFile)) {
      setUploadError("Only PDF and TXT files are supported.");
      return;
    }

    setUploadError("");
    setIsUploadingFile(true);

    const previousSessionId = sessionId;
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const payload = await requestApi("/sessions/upload-file", {
        method: "POST",
        body: formData,
      });

      await activateResource(
        payload.session_id,
        {
          kind: "file",
          label: selectedFile.name,
          detail: formatFileSize(selectedFile.size),
        },
        previousSessionId,
      );

      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      setUploadError(error.message);
    } finally {
      setIsUploadingFile(false);
    }
  }

  async function uploadText() {
    if (!textResource.trim() || isUploadingText) {
      return;
    }

    setUploadError("");
    setIsUploadingText(true);

    const previousSessionId = sessionId;
    const formData = new FormData();
    formData.append("text", textResource.trim());

    try {
      const payload = await requestApi("/sessions/upload-text", {
        method: "POST",
        body: formData,
      });

      const preview = textResource.trim().slice(0, 48);

      await activateResource(
        payload.session_id,
        {
          kind: "text",
          label: preview.length < textResource.trim().length ? `${preview}...` : preview,
          detail: `${textResource.trim().split(/\s+/).length} words`,
        },
        previousSessionId,
      );

      setTextResource("");
    } catch (error) {
      setUploadError(error.message);
    } finally {
      setIsUploadingText(false);
    }
  }

  async function submitQuestion(questionText) {
    const trimmedQuestion = questionText.trim();

    if (!trimmedQuestion || !sessionId || isSending) {
      return;
    }

    setChatError("");
    setIsSending(true);
    setComposerValue("");
    setMessages((currentMessages) => [
      ...currentMessages,
      buildMessage("user", trimmedQuestion),
    ]);

    try {
      const payload = await requestApi("/sessions/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          query: trimmedQuestion,
        }),
      });

      setMessages((currentMessages) => [
        ...currentMessages,
        buildMessage(
          "assistant",
          payload.response || "I could not find an answer.",
          payload.trace ?? null,
        ),
      ]);
    } catch (error) {
      setChatError(error.message);
      setMessages((currentMessages) => [
        ...currentMessages,
        buildMessage(
          "assistant",
          "I could not complete that request. Please try again after checking the uploaded resource.",
        ),
      ]);
    } finally {
      setIsSending(false);
    }
  }

  async function resetWorkspace() {
    const activeSessionId = sessionId;

    setSelectedFile(null);
    setTextResource("");
    setComposerValue("");
    setResourceMeta(null);
    setSessionId("");
    setUploadError("");
    setChatError("");
    setMessages([buildWelcomeMessage("")]);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    await destroySession(activeSessionId);
  }

  function handleComposerSubmit(event) {
    event.preventDefault();
    void submitQuestion(composerValue);
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion(composerValue);
    }
  }

  function handleSuggestionClick(prompt) {
    if (sessionId) {
      void submitQuestion(prompt);
      return;
    }

    setComposerValue(prompt);
  }

  const uploadBusy = isUploadingFile || isUploadingText;

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <main className="app-shell">
        <aside className="resource-panel">
          <div className="panel-topline">
            <span className="eyebrow">Grounded AI Workspace</span>
            <span className={`status-pill ${sessionId ? "ready" : "idle"}`}>
              {sessionId ? "Resource live" : "Awaiting context"}
            </span>
          </div>

          <section className="glass-card upload-card">
            <div className="section-heading">
              <div>
                <p className="section-kicker">Resource intake</p>
                <h2>Index a source</h2>
              </div>
              <span className="mini-status">
                {uploadBusy ? "Syncing..." : "Ready"}
              </span>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt"
              className="hidden-input"
              onChange={handleFileInputChange}
            />

            <div className="mode-switch" role="tablist" aria-label="Upload mode">
              <button
                type="button"
                className={`mode-button ${uploadMode === "file" ? "active" : ""}`}
                onClick={() => setUploadMode("file")}
              >
                File
              </button>
              <button
                type="button"
                className={`mode-button ${uploadMode === "text" ? "active" : ""}`}
                onClick={() => setUploadMode("text")}
              >
                Raw text
              </button>
            </div>

            {uploadMode === "file" ? (
              <div className="upload-pane file-pane">
                <div className="action-row">
                  <p>Drag in a PDF or TXT file, or pick one manually.</p>
                </div>

                <button
                  type="button"
                  className={`dropzone compact ${isDragging ? "dragging" : ""}`}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onClick={openFilePicker}
                >
                  <span className="dropzone-mark" />
                  <strong>Drop PDF or TXT here</strong>
                  <span>Click to pick from your device</span>
                </button>

                <div className="file-meta compact elevated">
                  <div>
                    <span className="file-meta-label">Selected file</span>
                    <strong>{selectedFile?.name || "Nothing selected yet"}</strong>
                  </div>
                  <div className="file-meta-actions">
                    <span>{selectedFile ? formatFileSize(selectedFile.size) : ""}</span>
                    <button
                      type="button"
                      className="ghost-button slim"
                      onClick={openFilePicker}
                      disabled={uploadBusy}
                    >
                      Browse
                    </button>
                  </div>
                </div>

                <button
                  type="button"
                  className="primary-button"
                  onClick={uploadFile}
                  disabled={!selectedFile || uploadBusy}
                >
                  {isUploadingFile ? "Uploading file..." : "Upload file"}
                </button>
              </div>
            ) : (
              <div className="upload-pane text-pane">
                <div className="action-row">
                  <p>Paste notes, transcripts, or rough research directly.</p>
                  <span className="text-counter">
                    {textResource.trim().split(/\s+/).filter(Boolean).length} words
                  </span>
                </div>

                <textarea
                  className="resource-textarea compact"
                  placeholder="Paste source text here..."
                  value={textResource}
                  onChange={(event) => setTextResource(event.target.value)}
                  disabled={uploadBusy}
                />

                <button
                  type="button"
                  className="secondary-button"
                  onClick={uploadText}
                  disabled={!textResource.trim() || uploadBusy}
                >
                  {isUploadingText ? "Uploading text..." : "Upload text"}
                </button>
              </div>
            )}
          </section>

          <section className="resource-card compact">
            <div className="session-header">
              <div>
                <p className="section-kicker">Session</p>
                <h2>{resourceMeta?.label || "No active resource yet"}</h2>
              </div>
              <span className={`mini-status ${sessionId ? "ready" : ""}`}>
                {sessionId ? "Live" : "Idle"}
              </span>
            </div>

            <div className="session-grid">
              <article className="session-stat">
                <span>Type</span>
                <strong>{resourceMeta?.kind || "None"}</strong>
              </article>
              <article className="session-stat">
                <span>Detail</span>
                <strong>{resourceMeta?.detail || "Upload a source to begin"}</strong>
              </article>
              <article className="session-stat">
                <span>Session ID</span>
                <strong>{formatSessionId(sessionId)}</strong>
              </article>
            </div>

            <button
              type="button"
              className="ghost-button slim wide"
              onClick={resetWorkspace}
              disabled={!sessionId && !selectedFile && !textResource}
            >
              Clear workspace
            </button>

            {uploadError ? <p className="error-banner compact">{uploadError}</p> : null}
          </section>
        </aside>

        <section className="chat-panel">
          <header className="chat-header">
            <div>
              <span className="eyebrow">Context chat</span>
              <h2>Conversation thread</h2>
            </div>
            <div className="chat-header-meta">
              <span className={`status-pill ${isSending ? "thinking" : "idle"}`}>
                {isSending ? "Thinking" : sessionId ? "Ready for questions" : "Upload to unlock"}
              </span>
            </div>
          </header>

          <div className="chat-surface">
            <div className="chat-thread">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}

              {isSending ? (
                <div className="message-row assistant">
                  <article className="message-bubble assistant typing-bubble">
                    <div className="typing-dots" aria-label="Assistant is typing">
                      <span />
                      <span />
                      <span />
                    </div>
                  </article>
                </div>
              ) : null}

              <div ref={threadEndRef} />
            </div>

            <div className="prompt-rail">
              {promptSuggestions.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="prompt-chip"
                  onClick={() => handleSuggestionClick(prompt)}
                  disabled={isSending}
                >
                  {prompt}
                </button>
              ))}
            </div>

            <form className="composer" onSubmit={handleComposerSubmit}>
              <textarea
                className="composer-input"
                placeholder={
                  sessionId
                    ? "Ask a grounded question about the uploaded resource..."
                    : "Upload a resource first, then chat here..."
                }
                value={composerValue}
                onChange={(event) => setComposerValue(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                disabled={!sessionId || isSending}
              />

              <button
                type="submit"
                className="send-button"
                disabled={!sessionId || !composerValue.trim() || isSending}
              >
                {isSending ? "Sending..." : "Send"}
              </button>
            </form>

            {chatError ? <p className="error-banner inline">{chatError}</p> : null}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

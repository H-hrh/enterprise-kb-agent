/* 企业知识库智能体 - 前端逻辑（原生 JS，无需构建工具）
 *
 * 主要功能：
 * 1. 文档上传 / 列表 / 删除
 * 2. 提问并以 fetch 流式读取 SSE，逐字渲染回答
 * 3. 渲染引用来源卡片，答案中的 [n] 角标可点击定位
 */

// ---------- 全局状态 ----------
// 会话 ID：存 localStorage，同一浏览器跨刷新保持（"新对话"可换 ID），首次访问自动生成
let sessionId = localStorage.getItem("kb_session_id");
if (!sessionId) {
  sessionId = "sess_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
  localStorage.setItem("kb_session_id", sessionId);
}

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const docListEl = document.getElementById("docList");
const docCountEl = document.getElementById("docCount");
const healthBadge = document.getElementById("healthBadge");
const keyBanner = document.getElementById("keyBanner");
const toastEl = document.getElementById("toast");

let sending = false;

// ---------- 工具函数 ----------

/** HTML 转义，防止文本里的 < > 等字符破坏页面 */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/** 把答案正文中的 [1][2] 替换为可点击的引用角标 */
function renderCitations(html) {
  return html.replace(/\[(\d+)\]/g, (_, n) => `<sup class="cite" data-n="${n}">[${n}]</sup>`);
}

let toastTimer = null;
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.add("hidden"), 5000);
}

// ---------- 健康检查 & 文档列表 ----------

async function loadHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    if (!data.api_key_configured) {
      healthBadge.textContent = "未配置 API Key";
      healthBadge.className = "badge badge-warn";
      keyBanner.classList.remove("hidden");
    } else {
      healthBadge.textContent = `服务正常 · ${data.document_count} 篇文档 / ${data.chunk_count} 个片段`;
      healthBadge.className = "badge badge-ok";
    }
  } catch (e) {
    healthBadge.textContent = "服务未连接";
    healthBadge.className = "badge badge-warn";
  }
}

async function loadDocs() {
  const resp = await fetch("/api/documents");
  const docs = await resp.json();
  docCountEl.textContent = docs.length;

  if (docs.length === 0) {
    docListEl.innerHTML = '<p class="empty-tip">还没有文档，先上传一份吧</p>';
    return;
  }

  docListEl.innerHTML = "";
  for (const doc of docs) {
    const item = document.createElement("div");
    item.className = "doc-item";
    const date = new Date(doc.created_at * 1000).toLocaleString("zh-CN", {
      month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
    });
    item.innerHTML = `
      <button class="doc-del" title="删除文档">✕</button>
      <div class="doc-name">📄 ${escapeHtml(doc.file_name)}</div>
      <div class="doc-meta">${doc.chunk_count} 个片段 · ${date}<span class="doc-hint">点击查看片段</span></div>`;

    // 点击卡片 → 展开/收起片段预览（首次点击时懒加载）
    item.addEventListener("click", async () => {
      const existing = item.querySelector(".doc-chunks");
      if (existing) {
        existing.remove();
        item.classList.remove("expanded");
        return;
      }
      item.classList.add("expanded");
      const panel = document.createElement("div");
      panel.className = "doc-chunks";
      panel.innerHTML = '<p class="chunk-tip">加载片段中…</p>';
      // 点击预览面板内部不触发收起（方便选中复制文本）
      panel.addEventListener("click", (e) => e.stopPropagation());
      item.appendChild(panel);
      try {
        const r = await fetch(`/api/documents/${doc.doc_id}/chunks`);
        if (!r.ok) throw new Error();
        const data = await r.json();
        panel.innerHTML = data.chunks
          .map(
            (c) => `
            <div class="chunk-item">
              <span class="chunk-no">#${c.index + 1}</span>
              <span class="chunk-text">${escapeHtml(c.text)}</span>
            </div>`
          )
          .join("");
      } catch {
        panel.innerHTML = '<p class="chunk-tip">片段加载失败，请重试</p>';
      }
    });

    item.querySelector(".doc-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`确定删除《${doc.file_name}》吗？相关片段将一并移除。`)) return;
      const r = await fetch(`/api/documents/${doc.doc_id}`, { method: "DELETE" });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        showToast(err.detail || "删除失败");
        return;
      }
      loadDocs();
      loadHealth();
    });
    docListEl.appendChild(item);
  }
}

// ---------- 文档上传 ----------

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  uploadBtn.disabled = true;
  uploadBtn.textContent = "上传中…";

  const form = new FormData();
  form.append("file", file);
  try {
    const resp = await fetch("/api/documents/upload", { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) {
      showToast(data.detail || "上传失败");
    } else {
      showToast(`《${data.file_name}》上传成功，切分为 ${data.chunk_count} 个片段`);
      loadDocs();
      loadHealth();
    }
  } catch (e) {
    showToast("上传失败：" + e.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "⬆ 上传文档";
    fileInput.value = ""; // 允许重复选择同一文件
  }
});

// ---------- 聊天（SSE 流式）----------

function appendMessage(role) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "🧑" : "🤖";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  if (role === "assistant") ensureBubbleDelegation(bubble);
  return { msg, bubble };
}

/**
 * 用"事件委托"绑定气泡内的交互（引用角标、来源卡片）。
 * 流式渲染会不断重写气泡内的 HTML，逐个元素绑监听器会被销毁；
 * 委托把监听器挂在 bubble 上，子元素怎么重建都不受影响。
 */
function ensureBubbleDelegation(bubble) {
  if (bubble.dataset.delegated) return;
  bubble.dataset.delegated = "1";
  bubble.addEventListener("click", (e) => {
    // 点击答案里的 [n] 角标 → 定位并高亮对应来源卡片
    const cite = e.target.closest(".cite");
    if (cite) {
      const card = bubble.querySelector(`.source-card[data-n="${cite.dataset.n}"]`);
      if (card) {
        card.classList.add("open", "highlight");
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        setTimeout(() => card.classList.remove("highlight"), 1600);
      }
      return;
    }
    // 点击来源卡片 → 展开/收起原文
    const card = e.target.closest(".source-card");
    if (card) card.classList.toggle("open");
  });
}

/** 渲染气泡基础结构（来源卡片 + 空的回答区），返回回答区元素 */
function renderAnswerArea(state) {
  state.bubble.innerHTML =
    (state.sourcesHtml || "") + '<div class="answer-text typing-cursor"></div>';
  state.answerEl = state.bubble.querySelector(".answer-text");
  return state.answerEl;
}

/** 处理一条 SSE 事件 */
function handleEvent(evt, state) {
  if (evt.type === "status") {
    state.statusEl.textContent = evt.data;
  } else if (evt.type === "sources") {
    state.statusEl.textContent = "";
    state.sources = evt.data || [];
    renderSources(state);
  } else if (evt.type === "token") {
    state.answerText += evt.data;
    // 流式过程中只更新回答区的文本（textContent 自动转义，防 XSS）
    const el = state.answerEl || renderAnswerArea(state);
    el.textContent = state.answerText;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else if (evt.type === "done") {
    // 结束：把 [n] 替换为可点击角标（点击交互由事件委托处理）
    const el = state.answerEl || renderAnswerArea(state);
    el.classList.remove("typing-cursor");
    el.innerHTML = renderCitations(escapeHtml(state.answerText));
  } else if (evt.type === "error") {
    state.statusEl.textContent = "";
    showToast(evt.data);
    if (!state.answerText) {
      state.bubble.innerHTML = `<span style="color:var(--danger)">❌ ${escapeHtml(evt.data)}</span>`;
    }
  }
}

function renderSources(state) {
  if (!state.sources.length) return;
  const cards = state.sources
    .map(
      (s) => `
      <div class="source-card" data-n="${s.index}">
        <div class="source-head">
          <span class="source-num">[${s.index}]</span>
          <span>📄 ${escapeHtml(s.file_name)}</span>
          <span class="source-score">相关性 ${s.score.toFixed(3)}</span>
        </div>
        <div class="source-text">${escapeHtml(s.text)}</div>
      </div>`
    )
    .join("");
  state.sourcesHtml = `<div class="sources">${cards}</div>`;
  // 卡片的点击展开由 ensureBubbleDelegation 统一处理，这里无需逐个绑定
  renderAnswerArea(state);
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || sending) return;
  if (!navigator.onLine) return;

  sending = true;
  sendBtn.disabled = true;

  // 用户气泡
  appendMessage("user").bubble.textContent = text;
  inputEl.value = "";
  inputEl.style.height = "auto";

  // 助手气泡（含状态行）
  const { bubble } = appendMessage("assistant");
  const statusEl = document.createElement("div");
  statusEl.className = "status-line";
  statusEl.innerHTML = '<span class="dots">正在思考</span>';
  bubble.appendChild(statusEl);

  const state = { bubble, statusEl, answerText: "", sources: [], sourcesHtml: "" };

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });

    if (!resp.ok) {
      // HTTP 错误（如 422/502），FastAPI 返回 {detail: "..."}
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `请求失败（HTTP ${resp.status}）`);
    }

    // 逐块读取 SSE 流
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE 每条消息以两个换行分隔
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // 最后一段可能不完整，留到下一轮

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        try {
          const evt = JSON.parse(line.slice(5).trim());
          handleEvent(evt, state);
        } catch (e) {
          console.error("事件解析失败：", e, line);
        }
      }
    }
  } catch (e) {
    statusEl.textContent = "";
    showToast(e.message);
    if (!state.answerText) {
      bubble.innerHTML = `<span style="color:var(--danger)">❌ ${escapeHtml(e.message)}</span>`;
    }
  } finally {
    sending = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => {
  // 中文输入法选词的 Enter 不发送（isComposing 表示拼音还没上屏）
  if (e.isComposing || e.keyCode === 229) return;
  // Enter 发送，Shift+Enter 换行
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// 输入框自动增高
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
});

// ---------- 启动 ----------
loadHealth();
loadDocs();
inputEl.focus();

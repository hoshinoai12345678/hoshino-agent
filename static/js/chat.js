/* =========================================================
 * 聊天逻辑 - 消息收发、面板更新
 * ========================================================= */
(function () {
  "use strict";

  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const resetBtn = document.getElementById("resetBtn");
  const forgetBtn = document.getElementById("forgetBtn");
  const devSwitch = document.getElementById("devSwitch");

  let isStreaming = false;
  let devMode = false;

  // 情景记忆分页状态
  const EPISODIC_PAGE_SIZE = 10;
  let episodicOffset = 0;

  // ---- 发送消息 ----
  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isStreaming) return;

    // 渲染用户消息
    appendMessage("user", text);
    inputEl.value = "";
    inputEl.style.height = "auto";

    // 创建 AI 消息占位
    const aiMsg = appendMessage("ai", "");
    isStreaming = true;
    sendBtn.disabled = true;

    // 打字指示器
    const typing = document.createElement("div");
    typing.className = "typing";
    typing.innerHTML = "<span></span><span></span><span></span>";
    aiMsg.querySelector(".bubble").appendChild(typing);

    try {
      let replyText = "";
      let thinkingText = "";
      const toolCalls = [];

      for await (const chunk of API.chatStream(text, devMode)) {
        // 移除打字指示器
        if (typing.parentNode) typing.remove();

        if (chunk.type === "thinking" && devMode) {
          thinkingText += chunk.content;
          renderThinking(aiMsg, thinkingText);
        } else if (chunk.type === "tool_call") {
          toolCalls.push(chunk);
          renderToolCall(aiMsg, chunk);
        } else if (chunk.type === "reply") {
          replyText += chunk.content;
          renderReply(aiMsg, replyText);
        } else if (chunk.type === "meta") {
          updatePanels(chunk.data);
        }
      }

      // 滚动到底部
      scrollToBottom();
    } catch (e) {
      if (typing.parentNode) typing.remove();
      renderReply(aiMsg, `❌ 出错了: ${e.message}`);
    } finally {
      // 对话结束后刷新画像面板（长期记忆可能已更新）
      // 放在 finally 且独立 try-catch，避免刷新失败影响回复显示
      loadMemories().catch(e => console.error("刷新画像失败:", e));
      isStreaming = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  // ---- 渲染消息 ----
  function appendMessage(role, content) {
    const msg = document.createElement("div");
    msg.className = `msg ${role}`;
    const avatar = role === "ai" ? "💕" : "你";
    msg.innerHTML = `
      <div class="avatar">${avatar}</div>
      <div class="bubble">${API.renderMarkdown(content)}</div>
    `;
    messagesEl.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  function renderReply(msgEl, text) {
    const bubble = msgEl.querySelector(".bubble");
    // 保留思考链和工具调用
    const extras = bubble.querySelectorAll(".thinking-block, .tool-block");
    bubble.innerHTML = API.renderMarkdown(text);
    extras.forEach(e => bubble.insertBefore(e, bubble.firstChild));
  }

  function renderThinking(msgEl, text) {
    const bubble = msgEl.querySelector(".bubble");
    let block = bubble.querySelector(".thinking-block");
    if (!block) {
      block = document.createElement("div");
      block.className = "thinking-block";
      bubble.insertBefore(block, bubble.firstChild);
    }
    block.textContent = text;
  }

  function renderToolCall(msgEl, chunk) {
    const bubble = msgEl.querySelector(".bubble");
    const block = document.createElement("div");
    block.className = "tool-block";
    block.innerHTML = `
      <div class="tool-name">🔧 ${chunk.name}</div>
      <div class="tool-result">${API.escapeHtml(chunk.result)}</div>
    `;
    bubble.insertBefore(block, bubble.firstChild);
  }

  // ---- 面板更新 ----
  function updatePanels(data) {
    if (data.emotion) updateEmotionPanel(data.emotion);
    if (data.tool_calls) updateToolPanel(data.tool_calls);
    if (data.memory_ops) updateMemoryPanel(data.memory_ops);
    updateStats(data);
  }

  function updateEmotionPanel(emotion) {
    const fav = emotion.favorability || 0;
    const favPercent = ((fav + 100) / 200 * 100).toFixed(0);

    document.getElementById("favValue").textContent = fav;
    document.getElementById("favLabel").textContent = emotion.favorability_level || "";
    document.getElementById("favBar").style.width = favPercent + "%";

    const p = emotion.pleasure || 0;
    const a = emotion.arousal || 0;
    const d = emotion.dominance || 0;
    document.getElementById("pleasureVal").textContent = p.toFixed(2);
    document.getElementById("arousalVal").textContent = a.toFixed(2);
    document.getElementById("dominanceVal").textContent = d.toFixed(2);
    document.getElementById("pleasureBar").style.width = ((p + 1) / 2 * 100) + "%";
    document.getElementById("arousalBar").style.width = ((a + 1) / 2 * 100) + "%";
    document.getElementById("dominanceBar").style.width = ((d + 1) / 2 * 100) + "%";
    document.getElementById("emotionLabel").textContent = emotion.emotion_label || "";
  }

  function updateToolPanel(tools) {
    const el = document.getElementById("toolList");
    if (!tools || tools.length === 0) {
      el.innerHTML = '<div class="empty">暂无工具调用</div>';
      return;
    }
    el.innerHTML = tools.map(t => `
      <div class="memory-item">
        <div class="mem-type">🔧 ${t.name}</div>
        <div>${API.escapeHtml(t.result || "")}</div>
      </div>
    `).join("");
  }

  function updateMemoryPanel(ops) {
    const el = document.getElementById("memoryList");
    if (!ops || ops.length === 0) return;
    const html = ops.map(op => {
      if (op.op === "save_memory") {
        return `<div class="memory-item">
          <div class="mem-type">📝 新记忆</div>
          <div>${API.escapeHtml(op.content || "")}</div>
        </div>`;
      } else if (op.op === "emotion_update") {
        return `<div class="memory-item">
          <div class="mem-type">💓 情绪更新</div>
          <div>${API.escapeHtml(JSON.stringify(op.detail))}</div>
        </div>`;
      } else if (op.op === "consolidate") {
        return `<div class="memory-item">
          <div class="mem-type">🔄 记忆巩固</div>
          <div>已执行记忆巩固</div>
        </div>`;
      }
      return "";
    }).join("");
    el.innerHTML = html + el.innerHTML;
  }

  function updateStats(data) {
    if (data.working_memory_size !== undefined)
      document.getElementById("statWorking").textContent = data.working_memory_size;
    if (data.episodic_count !== undefined)
      document.getElementById("statEpisodic").textContent = data.episodic_count;
    if (data.semantic_count !== undefined)
      document.getElementById("statSemantic").textContent = data.semantic_count;
  }

  // ---- 初始化状态 ----
  async function loadState() {
    try {
      const state = await API.getState();
      updatePanels(state);
    } catch (e) {
      console.error("加载状态失败:", e);
    }
  }

  // ---- 加载对话历史（刷新页面后恢复聊天框）----
  async function loadHistory() {
    try {
      const data = await API.getHistory();
      if (data.messages && data.messages.length > 0) {
        messagesEl.innerHTML = "";
        for (const m of data.messages) {
          // working memory 中 role 为 "user" / "assistant"，前端映射为 "user" / "ai"
          const role = m.role === "user" ? "user" : "ai";
          appendMessage(role, m.content);
        }
      }
    } catch (e) {
      console.error("加载历史失败:", e);
    }
  }

  // ---- 加载记忆（首屏，重置分页）----
  async function loadMemories() {
    episodicOffset = 0;
    try {
      const data = await API.getMemories(EPISODIC_PAGE_SIZE, 0);
      // 记忆操作日志面板（仅实时 ops，由 updateMemoryPanel 填充）
      updateStats({
        episodic_count: data.episodic_count,
        semantic_count: data.semantic_count,
      });
      // 渲染用户画像 + 最近记忆到画像面板
      renderProfile(data.semantic || []);
      renderEpisodic(data.episodic || [], false);
      updateLoadMoreBtn(data.episodic_count, (data.episodic || []).length);
    } catch (e) {
      console.error("加载记忆失败:", e);
    }
  }

  // ---- 加载更多情景记忆（追加下一页）----
  async function loadMoreEpisodic() {
    try {
      const data = await API.getMemories(EPISODIC_PAGE_SIZE, episodicOffset);
      renderEpisodic(data.episodic || [], true);
      updateLoadMoreBtn(data.episodic_count, (data.episodic || []).length);
    } catch (e) {
      console.error("加载更多失败:", e);
    }
  }

  // ---- 更新"加载更多"按钮显隐与文案 ----
  function updateLoadMoreBtn(totalCount, currentPageSize) {
    const btn = document.getElementById("loadMoreEpisodic");
    if (!btn) return;
    // 本次已加载到 episodicOffset + currentPageSize
    const loaded = episodicOffset + currentPageSize;
    if (loaded < totalCount) {
      btn.style.display = "block";
      btn.textContent = `加载更多（剩余 ${totalCount - loaded} 条）`;
      episodicOffset = loaded;  // 推进偏移量，供下次请求使用
    } else {
      btn.style.display = "none";
    }
  }

  // ---- 渲染用户画像 ----
  function renderProfile(profile) {
    const el = document.getElementById("profileList");
    const countEl = document.getElementById("profileCount");
    if (!profile || profile.length === 0) {
      el.innerHTML = '<div class="empty">暂无用户画像，多聊几句让爱了解你~</div>';
      countEl.textContent = "";
      return;
    }
    countEl.textContent = `（共 ${profile.length} 条）`;

    const categoryNames = {
      basic: "基本信息",
      preference: "偏好",
      personality: "性格",
      interest: "兴趣",
    };
    const categoryIcons = {
      basic: "👤",
      preference: "💖",
      personality: "🌟",
      interest: "🎯",
    };

    // 按类别分组
    const groups = {};
    profile.forEach(p => {
      const cat = p.category || "other";
      (groups[cat] = groups[cat] || []).push(p);
    });

    const html = Object.keys(groups).map(cat => {
      const items = groups[cat];
      const name = categoryNames[cat] || cat;
      const icon = categoryIcons[cat] || "📝";
      const entries = items.map(i => {
        const conf = (i.confidence != null ? (i.confidence * 100).toFixed(0) : "?") + "%";
        return `<div class="profile-entry">
          <span class="profile-key">${API.escapeHtml(i.key)}</span>
          <span class="profile-sep">:</span>
          <span class="profile-val">${API.escapeHtml(i.value)}</span>
          <span class="profile-conf" title="置信度">${conf}</span>
        </div>`;
      }).join("");
      return `<div class="profile-group">
        <div class="profile-cat">${icon} ${name}</div>
        ${entries}
      </div>`;
    }).join("");

    el.innerHTML = html;
  }

  // ---- 渲染最近情景记忆（支持首屏与追加）----
  function renderEpisodic(memories, append) {
    const el = document.getElementById("episodicList");
    const hintEl = document.getElementById("episodicHint");
    if (!memories || memories.length === 0) {
      if (!append) {
        el.innerHTML = '<div class="empty">暂无对话记忆</div>';
        hintEl.textContent = "";
      }
      return;
    }
    // 首屏：清空并显示"最近 N 条"；追加：更新计数
    if (!append) {
      el.innerHTML = "";
    }

    const typeIcons = {
      dialogue: "💬",
      fact: "📌",
      emotion: "💓",
    };

    const html = memories.map(m => {
      const icon = typeIcons[m.event_type] || "📝";
      const ts = m.timestamp ? formatTime(m.timestamp) : "";
      const imp = m.importance != null
        ? `<span class="mem-importance" title="重要性">${(m.importance * 100).toFixed(0)}%</span>`
        : "";
      return `<div class="memory-item">
        <div class="mem-head">
          <span class="mem-type">${icon} ${m.event_type || "dialogue"}</span>
          <span class="mem-time">${ts}</span>
          ${imp}
        </div>
        <div class="mem-content">${API.escapeHtml(m.content || "")}</div>
      </div>`;
    }).join("");
    el.insertAdjacentHTML("beforeend", html);

    // 提示文案：显示已加载数 / 总数（总数由 updateLoadMoreBtn 维护，这里用当前 DOM 条数近似）
    const loadedCount = el.querySelectorAll(".memory-item").length;
    hintEl.textContent = `（已加载 ${loadedCount} 条）`;
  }

  // ---- 时间格式化 ----
  function formatTime(ts) {
    const d = new Date(ts * 1000);
    const now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    const pad = n => String(n).padStart(2, "0");
    if (sameDay) {
      return `今天 ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    }
    return `${d.getMonth() + 1}/${d.getDate()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  // ---- 重置 ----
  async function reset() {
    if (!confirm("确定要重置对话吗？工作记忆和情绪将清空，长期记忆保留。")) return;
    try {
      await API.reset();
      messagesEl.innerHTML = "";
      appendMessage("ai", "爱回来啦！让我们重新开始吧~♪ 有什么想跟爱说的吗？");
      await loadState();
    } catch (e) {
      alert("重置失败: " + e.message);
    }
  }

  // ---- 忘记我（清空长期记忆）----
  async function forget() {
    if (!confirm("确定要让我忘记你吗？\n\n将清空：情景记忆（对话事件）+ 语义记忆（用户画像）\n保留：工作记忆、情绪、角色知识库\n\n此操作不可恢复！")) return;
    try {
      const res = await API.forget();
      // 刷新记忆面板和统计
      document.getElementById("memoryList").innerHTML = '<div class="empty">暂无记忆操作</div>';
      await loadMemories();
      await loadState();
      appendMessage("ai", `咦……爱好像忘记了什么重要的东西~♪\n（已清空情景记忆 ${res.episodic_deleted} 条，用户画像 ${res.semantic_deleted} 条）`);
      scrollToBottom();
    } catch (e) {
      alert("忘记失败: " + e.message);
    }
  }

  // ---- 工具函数 ----
  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
  }

  // ---- 事件绑定 ----
  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("input", autoResize);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  resetBtn.addEventListener("click", reset);
  if (forgetBtn) forgetBtn.addEventListener("click", forget);

  // "加载更多"情景记忆按钮
  const loadMoreBtn = document.getElementById("loadMoreEpisodic");
  if (loadMoreBtn) loadMoreBtn.addEventListener("click", loadMoreEpisodic);
  devSwitch.addEventListener("change", (e) => {
    devMode = e.target.checked;
    console.log("开发者模式:", devMode);
  });

  // ---- 标签切换（用 display 控制）----
  document.querySelectorAll(".sidebar-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".sidebar-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      document.querySelectorAll(".tab-panel").forEach(p => p.style.display = "none");
      document.getElementById(tab.dataset.panel).style.display = "block";
    });
  });

  // ---- 认证逻辑 ----
  const authOverlay = document.getElementById("authOverlay");
  const authForm = document.getElementById("authForm");
  const authUsername = document.getElementById("authUsername");
  const authPassword = document.getElementById("authPassword");
  const authSubmitBtn = document.getElementById("authSubmitBtn");
  const authSubtitle = document.getElementById("authSubtitle");
  const authSwitchText = document.getElementById("authSwitchText");
  const authSwitchLink = document.getElementById("authSwitchLink");
  const authError = document.getElementById("authError");
  const authGuestBtn = document.getElementById("authGuestBtn");
  const userBadge = document.getElementById("userBadge");
  const logoutBtn = document.getElementById("logoutBtn");

  let authMode = "login";   // "login" | "register"

  function showAuthOverlay() {
    authOverlay.style.display = "flex";
    authError.textContent = "";
    authUsername.value = "";
    authPassword.value = "";
    authUsername.focus();
  }

  function hideAuthOverlay() {
    authOverlay.style.display = "none";
  }

  function updateUserBadge(username) {
    if (username) {
      userBadge.textContent = "👤 " + username;
      userBadge.style.display = "inline-block";
      logoutBtn.style.display = "inline-block";
    } else {
      userBadge.style.display = "none";
      logoutBtn.style.display = "none";
    }
  }

  function switchAuthMode(e) {
    e.preventDefault();
    if (authMode === "login") {
      authMode = "register";
      authSubtitle.textContent = "注册一个新账号~♪";
      authSubmitBtn.textContent = "注册";
      authSwitchText.textContent = "已有账号？";
      authSwitchLink.textContent = "登录";
    } else {
      authMode = "login";
      authSubtitle.textContent = "登录后开始聊天~♪";
      authSubmitBtn.textContent = "登录";
      authSwitchText.textContent = "没有账号？";
      authSwitchLink.textContent = "注册";
    }
    authError.textContent = "";
    authUsername.focus();
  }

  async function handleAuthSubmit(e) {
    e.preventDefault();
    const username = authUsername.value.trim();
    const password = authPassword.value;
    if (!username || !password) {
      authError.textContent = "请输入用户名和密码";
      return;
    }
    authSubmitBtn.disabled = true;
    authError.textContent = "";
    try {
      if (authMode === "register") {
        await API.register(username, password);
      }
      // 注册后直接登录（后端设置 HttpOnly Cookie）
      await API.login(username, password);
      hideAuthOverlay();
      updateUserBadge(username);
      await reloadAll();
    } catch (err) {
      authError.textContent = err.message || (authMode === "register" ? "注册失败" : "登录失败");
    } finally {
      authSubmitBtn.disabled = false;
    }
  }

  async function handleGuest() {
    hideAuthOverlay();
    updateUserBadge(null);
    await reloadAll();
  }

  async function handleLogout() {
    try { await API.logout(); } catch (e) { console.error("登出失败:", e); }
    updateUserBadge(null);
    messagesEl.innerHTML = "";
    showAuthOverlay();
  }

  async function reloadAll() {
    // 切换 session 后重新加载所有面板数据
    loadState();
    loadHistory();
    loadMemories();
    inputEl.focus();
  }

  async function checkAuth() {
    try {
      const me = await API.getMe();
      if (me && me.username) {
        hideAuthOverlay();
        updateUserBadge(me.username);
        await reloadAll();
        return;
      }
    } catch (e) {
      console.error("检查登录态失败:", e);
    }
    // 未登录：显示遮罩，等待用户登录/注册/访客
    showAuthOverlay();
  }

  // 认证事件绑定
  if (authForm) authForm.addEventListener("submit", handleAuthSubmit);
  if (authSwitchLink) authSwitchLink.addEventListener("click", switchAuthMode);
  if (authGuestBtn) authGuestBtn.addEventListener("click", handleGuest);
  if (logoutBtn) logoutBtn.addEventListener("click", handleLogout);

  // ---- 初始化 ----
  checkAuth();
})();

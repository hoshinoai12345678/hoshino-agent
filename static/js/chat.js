/* =========================================================
 * 聊天逻辑 - 多角色并发架构（微信风格）
 *
 * 核心设计：每个角色拥有独立的 session 状态（消息缓冲、流式标志、
 * 草稿、面板数据）。切换联系人不阻塞后台流式回复；不同角色的
 * SSE 流在后台并发执行，互不耽误。后端 Agent 实例按
 * (session_id, character_id) 隔离，FastAPI async 天然支持并发请求。
 * ========================================================= */
(function () {
  "use strict";

  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const resetBtn = document.getElementById("resetBtn");
  const forgetBtn = document.getElementById("forgetBtn");
  const devSwitch = document.getElementById("devSwitch");
  const contactsListEl = document.getElementById("contactsList");

  let devMode = false;

  // 当前查看的角色
  let currentCharacterId = null;
  let charactersCache = [];

  // 角色头像图片路径（新增角色只需在 data/characters/{id}/ 放头像并在此映射）
  const CHARACTER_AVATARS = {
    hoshino_ai: "/static/images/hoshino_ai.jpg",
    zi_ling: "/static/images/zi_ling.jpg",
  };
  const DEFAULT_AVATAR = "/static/images/default.jpg";

  // 角色欢迎语映射（切换角色时显示，可按角色定制语气）
  const CHARACTER_WELCOME = {
    hoshino_ai: "我是新B小町的星野爱！✨\n今天也一起开心地聊天吧~♪\n有什么想跟爱说的吗？",
    zi_ling: "紫灵见过道友。\n修行路漫漫，今日有缘相会，不知有何指教？",
  };

  const EPISODIC_PAGE_SIZE = 10;

  // 消息 id 计数器（用于 DOM 元素与缓冲对象对应）
  let _msgIdSeq = 0;

  // ---- 每角色独立 session 状态 ----
  // 每个角色的流式回复在后台独立进行，切换联系人不会中断。
  const sessions = Object.create(null);

  function getSession(cid) {
    if (!sessions[cid]) {
      sessions[cid] = {
        isStreaming: false,        // 该角色是否正在流式回复（独立于当前查看的角色）
        messages: [],              // 消息缓冲：{id, role, content, thinking, toolCalls, streaming}
        inputDraft: "",            // 输入框草稿（切换时保存/恢复）
        scrollTop: 0,              // 滚动位置
        // 面板数据（按角色隔离，切换时恢复显示）
        emotion: null,
        toolCalls: [],
        memoryOps: [],
        stats: {},
        profile: [],
        episodic: [],
        episodicCount: 0,
        semanticCount: 0,
        episodicOffset: 0,
        historyLoaded: false,
        memoriesLoaded: false,
        stateLoaded: false,
      };
    }
    return sessions[cid];
  }

  // ---- 角色管理 ----

  function getCharacterAvatar(characterId) {
    return CHARACTER_AVATARS[characterId] || DEFAULT_AVATAR;
  }

  function getCharacterWelcome(characterId) {
    return CHARACTER_WELCOME[characterId] || `你好，我是${getCharacterName(characterId)}。`;
  }

  function getCharacterName(characterId) {
    const c = charactersCache.find(c => c.id === characterId);
    return c ? c.name : characterId;
  }

  /** 加载角色列表，渲染联系人列表 */
  async function loadCharacters() {
    try {
      const data = await API.getCharacters();
      charactersCache = data.characters || [];
      renderContactsList();

      // 设置默认角色
      const def = data.default || (charactersCache[0] && charactersCache[0].id);
      if (def && !currentCharacterId) {
        await switchCharacter(def);
      }
    } catch (e) {
      console.error("加载角色列表失败:", e);
      contactsListEl.innerHTML = '<div class="empty">加载失败</div>';
    }
  }

  /** 渲染联系人列表 */
  function renderContactsList() {
    contactsListEl.innerHTML = "";
    for (const c of charactersCache) {
      const item = document.createElement("div");
      item.className = "contact-item";
      item.dataset.characterId = c.id;
      const descParts = [];
      if (c.source) descParts.push(`《${c.source}》`);
      if (c.occupation) descParts.push(c.occupation);
      item.innerHTML = `
        <div class="contact-avatar-wrap">
          <img class="contact-avatar" src="${getCharacterAvatar(c.id)}" alt="${API.escapeHtml(c.name)}" />
          <span class="typing-dot" style="display:none;"></span>
        </div>
        <div class="contact-info">
          <div class="contact-name">${API.escapeHtml(c.name)}</div>
          <div class="contact-desc">${API.escapeHtml(descParts.join(" · ") || "AI角色")}</div>
        </div>
      `;
      item.addEventListener("click", () => switchCharacter(c.id));
      contactsListEl.appendChild(item);
    }
    // 同步当前高亮与输入指示
    if (currentCharacterId) {
      highlightContact(currentCharacterId);
      syncTypingDots();
    }
  }

  /** 更新联系人项的"正在输入"小点显隐 */
  function syncTypingDots() {
    contactsListEl.querySelectorAll(".contact-item").forEach(item => {
      const cid = item.dataset.characterId;
      const dot = item.querySelector(".typing-dot");
      const sess = sessions[cid];
      const streaming = sess && sess.isStreaming;
      if (dot) dot.style.display = streaming ? "block" : "none";
      // 当前正在查看的角色不需要小点（聊天区已有打字指示器）
      if (cid === currentCharacterId && dot) dot.style.display = "none";
    });
  }

  function highlightContact(characterId) {
    contactsListEl.querySelectorAll(".contact-item").forEach(item => {
      item.classList.toggle("active", item.dataset.characterId === characterId);
    });
  }

  /** 更新顶栏显示（头像、名字、状态） */
  function updateTopbar(characterId) {
    const name = getCharacterName(characterId);
    const avatar = getCharacterAvatar(characterId);
    const char = charactersCache.find(c => c.id === characterId);

    document.getElementById("topbarAvatar").src = avatar;
    document.getElementById("topbarName").textContent = name;
    const statusParts = [];
    if (char && char.source) statusParts.push(`《${char.source}》`);
    if (char && char.occupation) statusParts.push(char.occupation);
    document.getElementById("topbarStatus").textContent = statusParts.length > 0 ? "在线 · " + statusParts.join(" ") : "在线";
    inputEl.placeholder = `跟${name}说点什么...`;
  }

  /**
   * 切换角色（点击联系人）
   * 关键：不阻塞后台流式回复。切换时：
   * 1. 保存当前角色的输入草稿与滚动位置
   * 2. 切换视图：重渲染目标角色的消息缓冲
   * 3. 恢复目标角色的草稿、面板数据、滚动位置
   * 4. 后台流继续写入原角色的缓冲，不受影响
   */
  async function switchCharacter(characterId) {
    if (!characterId) return;
    if (characterId === currentCharacterId) return;

    // 保存当前角色的草稿与滚动位置（不中断其后台流）
    if (currentCharacterId) {
      const oldSess = getSession(currentCharacterId);
      oldSess.inputDraft = inputEl.value;
      oldSess.scrollTop = messagesEl.scrollTop;
    }

    currentCharacterId = characterId;
    highlightContact(characterId);
    updateTopbar(characterId);

    // 切换视图：从目标角色缓冲重渲染
    const sess = getSession(characterId);
    renderAllMessages(sess);
    renderPanelsFromSession(sess);

    // 恢复草稿
    inputEl.value = sess.inputDraft || "";
    autoResize();
    sendBtn.disabled = sess.isStreaming;  // 当前角色正在回复则禁用发送（但不影响其他角色）

    // 恢复滚动位置（下一帧，等 DOM 渲染完）
    requestAnimationFrame(() => {
      messagesEl.scrollTop = sess.scrollTop || 0;
      if (sess.isStreaming) scrollToBottom();  // 流式中切回自动滚到底
    });

    syncTypingDots();

    // 首次进入该角色：加载历史与面板数据
    if (!sess.historyLoaded) {
      sess.historyLoaded = true;
      await loadHistory(characterId);
    }
    if (!sess.stateLoaded) {
      sess.stateLoaded = true;
      loadState(characterId);
    }
    if (!sess.memoriesLoaded) {
      sess.memoriesLoaded = true;
      loadMemories(characterId);
    }

    inputEl.focus();
  }

  // ---- 发送消息（并发核心）----
  /**
   * 每个角色的流式回复独立进行。发送时只检查【当前角色】是否在回复，
   * 不检查其他角色——其他角色的后台流不受影响。
   */
  async function sendMessage() {
    const cid = currentCharacterId;
    if (!cid) return;
    const sess = getSession(cid);
    if (sess.isStreaming) return;  // 同一角色不并发发送（避免对话错乱）

    const text = inputEl.value.trim();
    if (!text) return;

    // 渲染用户消息（写入缓冲 + DOM）
    pushMessage(sess, "user", text);
    sess.inputDraft = "";
    inputEl.value = "";
    inputEl.style.height = "auto";

    // 创建 AI 占位消息（流式标志位 true）
    const aiMsg = pushMessage(sess, "ai", "", { streaming: true });
    sess.isStreaming = true;
    sendBtn.disabled = true;
    syncTypingDots();

    // 打字指示器
    const typingEl = document.createElement("div");
    typingEl.className = "typing";
    typingEl.innerHTML = "<span></span><span></span><span></span>";
    const aiBubble = getMsgBubbleEl(aiMsg.id);
    if (aiBubble) aiBubble.appendChild(typingEl);

    // 捕获切换时的"当前查看"判定：流始终写入 cid 的缓冲，
    // 但 DOM 只在用户正在查看 cid 时实时更新。
    try {
      let replyText = "";
      let thinkingText = "";

      for await (const chunk of API.chatStream(text, devMode, cid)) {
        // 仅当用户当前正在查看该角色时才更新 DOM
        const viewing = (currentCharacterId === cid);

        if (chunk.type === "thinking" && devMode) {
          thinkingText += chunk.content;
          aiMsg.thinking = thinkingText;
          if (viewing) renderThinkingInto(aiMsg.id, thinkingText);
        } else if (chunk.type === "tool_call") {
          aiMsg.toolCalls.push(chunk);
          if (viewing) renderToolCallInto(aiMsg.id, chunk);
        } else if (chunk.type === "reply") {
          replyText += chunk.content;
          aiMsg.content = replyText;
          if (viewing) renderReplyInto(aiMsg.id, replyText);
        } else if (chunk.type === "meta") {
          applyMetaToSession(cid, chunk.data);
          if (viewing) updatePanelsFromData(chunk.data);
        }

        if (viewing) scrollToBottom();
      }

      // 流结束
      aiMsg.streaming = false;
      const finalBubble = getMsgBubbleEl(aiMsg.id);
      if (finalBubble) {
        const t = finalBubble.querySelector(".typing");
        if (t) t.remove();
      }
    } catch (e) {
      aiMsg.streaming = false;
      aiMsg.content = `❌ 出错了: ${e.message}`;
      if (currentCharacterId === cid) renderReplyInto(aiMsg.id, aiMsg.content);
      const b = getMsgBubbleEl(aiMsg.id);
      if (b) { const t = b.querySelector(".typing"); if (t) t.remove(); }
    } finally {
      sess.isStreaming = false;
      if (currentCharacterId === cid) sendBtn.disabled = false;
      syncTypingDots();
      // 对话结束后刷新该角色画像面板（长期记忆可能已更新）
      loadMemories(cid).catch(err => console.error("刷新画像失败:", err));
      if (currentCharacterId === cid) inputEl.focus();
    }
  }

  // ---- 消息缓冲与渲染 ----

  /** 推入一条消息到 session 缓冲，并（若当前查看该角色）渲染到 DOM */
  function pushMessage(sess, role, content, opts = {}) {
    const msg = {
      id: ++_msgIdSeq,
      role,
      content,
      thinking: "",
      toolCalls: [],
      streaming: !!opts.streaming,
    };
    sess.messages.push(msg);
    if (sessions[currentCharacterId] === sess) {
      renderMessage(msg);
      scrollToBottom();
    }
    return msg;
  }

  /** 从 session 缓冲全量重渲染消息区（切换角色时调用） */
  function renderAllMessages(sess) {
    messagesEl.innerHTML = "";
    if (!sess.messages || sess.messages.length === 0) {
      // 显示欢迎语占位
      const welcome = getCharacterWelcome(currentCharacterId);
      const msg = {
        id: ++_msgIdSeq,
        role: "ai",
        content: welcome,
        thinking: "",
        toolCalls: [],
        streaming: false,
      };
      sess.messages.push(msg);
      renderMessage(msg);
      return;
    }
    for (const msg of sess.messages) renderMessage(msg);
  }

  /** 渲染单条消息到 DOM */
  function renderMessage(msg) {
    const el = document.createElement("div");
    el.className = `msg ${msg.role}`;
    el.dataset.msgId = msg.id;
    if (msg.role === "ai") {
      el.innerHTML = `
        <img class="avatar" src="${getCharacterAvatar(currentCharacterId)}" alt="avatar" />
        <div class="bubble"></div>
      `;
    } else {
      el.innerHTML = `
        <div class="avatar">你</div>
        <div class="bubble"></div>
      `;
    }
    messagesEl.appendChild(el);

    const bubble = el.querySelector(".bubble");
    // 先渲染思考链
    if (msg.thinking) renderThinkingInto(msg.id, msg.thinking);
    // 再渲染工具调用
    for (const tc of msg.toolCalls) renderToolCallInto(msg.id, tc);
    // 最后渲染正文
    if (msg.content) bubble.insertAdjacentHTML("beforeend", API.renderMarkdown(msg.content));
  }

  function getMsgEl(msgId) {
    return messagesEl.querySelector(`.msg[data-msg-id="${msgId}"]`);
  }
  function getMsgBubbleEl(msgId) {
    const el = getMsgEl(msgId);
    return el ? el.querySelector(".bubble") : null;
  }

  function renderReplyInto(msgId, text) {
    const bubble = getMsgBubbleEl(msgId);
    if (!bubble) return;
    // 保留思考链与工具调用块
    const extras = bubble.querySelectorAll(".thinking-block, .tool-block, .typing");
    const extrasArr = Array.from(extras);
    bubble.innerHTML = API.renderMarkdown(text);
    extrasArr.forEach(e => bubble.insertBefore(e, bubble.firstChild));
  }

  function renderThinkingInto(msgId, text) {
    const bubble = getMsgBubbleEl(msgId);
    if (!bubble) return;
    let block = bubble.querySelector(".thinking-block");
    if (!block) {
      block = document.createElement("div");
      block.className = "thinking-block";
      bubble.insertBefore(block, bubble.firstChild);
    }
    block.textContent = text;
  }

  function renderToolCallInto(msgId, chunk) {
    const bubble = getMsgBubbleEl(msgId);
    if (!bubble) return;
    const block = document.createElement("div");
    block.className = "tool-block";
    block.innerHTML = `
      <div class="tool-name">🔧 ${API.escapeHtml(chunk.name || "")}</div>
      <div class="tool-result">${API.escapeHtml(chunk.result || "")}</div>
    `;
    bubble.insertBefore(block, bubble.firstChild);
  }

  // ---- 面板数据：写入 session + 按需渲染 DOM ----

  function applyMetaToSession(cid, data) {
    const sess = getSession(cid);
    if (data.emotion) sess.emotion = data.emotion;
    if (data.tool_calls) sess.toolCalls = data.tool_calls;
    if (data.memory_ops) sess.memoryOps = sess.memoryOps.concat(data.memory_ops);
    if (data.working_memory_size !== undefined) sess.stats.working_memory_size = data.working_memory_size;
    if (data.episodic_count !== undefined) sess.episodicCount = data.episodic_count;
    if (data.semantic_count !== undefined) sess.semanticCount = data.semantic_count;
  }

  function updatePanelsFromData(data) {
    if (data.emotion) updateEmotionPanel(data.emotion);
    if (data.tool_calls) updateToolPanel(data.tool_calls);
    if (data.memory_ops) updateMemoryPanel(data.memory_ops);
    updateStats(data);
  }

  /** 切换角色时从 session 恢复面板显示 */
  function renderPanelsFromSession(sess) {
    if (sess.emotion) updateEmotionPanel(sess.emotion);
    else resetEmotionPanel();
    updateToolPanel(sess.toolCalls);
    // 记忆操作日志保持各角色独立：切回时从缓冲恢复该角色的日志
    renderMemoryOpsList(sess.memoryOps);
    updateStats({
      working_memory_size: sess.stats.working_memory_size,
      episodic_count: sess.episodicCount,
      semantic_count: sess.semanticCount,
    });
    // 用户画像与情景记忆列表
    renderProfile(sess.profile);
    renderEpisodic(sess.episodic, false);
    updateLoadMoreBtn(sess.episodicCount, (sess.episodic || []).length);
  }

  /** 从缓冲全量渲染记忆操作日志（最新在顶部，与流式增量顺序一致） */
  function renderMemoryOpsList(ops) {
    const el = document.getElementById("memoryList");
    if (!ops || ops.length === 0) {
      el.innerHTML = '<div class="empty">暂无记忆操作</div>';
      return;
    }
    el.innerHTML = "";
    // 倒序渲染：最新的 op 在顶部（与流式 updateMemoryPanel 的 prepend 行为一致）
    for (let i = ops.length - 1; i >= 0; i--) {
      const html = memoryOpToHtml(ops[i]);
      if (html) el.insertAdjacentHTML("beforeend", html);
    }
  }

  /** 单条记忆操作转 HTML */
  function memoryOpToHtml(op) {
    if (op.op === "save_memory") {
      return `<div class="memory-item">
        <div class="mem-type">📝 新记忆</div>
        <div>${API.escapeHtml(op.content || "")}</div>
      </div>`;
    } else if (op.op === "save_profile") {
      return `<div class="memory-item">
        <div class="mem-type">👤 画像更新</div>
        <div>${API.escapeHtml((op.key || "") + ": " + (op.value || ""))}</div>
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
  }

  function resetEmotionPanel() {
    document.getElementById("favValue").textContent = "—";
    document.getElementById("favLabel").textContent = "";
    document.getElementById("favBar").style.width = "0%";
    ["pleasureVal", "arousalVal", "dominanceVal"].forEach(id => document.getElementById(id).textContent = "0.00");
    ["pleasureBar", "arousalBar", "dominanceBar"].forEach(id => document.getElementById(id).style.width = "50%");
    document.getElementById("emotionLabel").textContent = "—";
  }

  function updateEmotionPanel(emotion) {
    const fav = emotion.favorability || 0;
    const favPercent = ((fav + 100) / 200 * 100).toFixed(0);
    document.getElementById("favValue").textContent = fav;
    document.getElementById("favLabel").textContent = emotion.favorability_level || "";
    document.getElementById("favBar").style.width = favPercent + "%";
    const p = emotion.pleasure || 0, a = emotion.arousal || 0, d = emotion.dominance || 0;
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
        <div class="mem-type">🔧 ${API.escapeHtml(t.name || "")}</div>
        <div>${API.escapeHtml(t.result || "")}</div>
      </div>
    `).join("");
  }

  /** 流式增量追加记忆操作日志（最新在顶部） */
  function updateMemoryPanel(ops) {
    const el = document.getElementById("memoryList");
    if (!ops || ops.length === 0) return;
    // 首次写入时清掉空占位
    const empty = el.querySelector(".empty");
    if (empty) empty.remove();
    // 倒序插入：数组里最后一个 op 是最新的，应排在最顶部
    for (let i = ops.length - 1; i >= 0; i--) {
      const html = memoryOpToHtml(ops[i]);
      if (html) el.insertAdjacentHTML("afterbegin", html);
    }
  }

  function updateStats(data) {
    if (data.working_memory_size !== undefined)
      document.getElementById("statWorking").textContent = data.working_memory_size;
    if (data.episodic_count !== undefined)
      document.getElementById("statEpisodic").textContent = data.episodic_count;
    if (data.semantic_count !== undefined)
      document.getElementById("statSemantic").textContent = data.semantic_count;
  }

  // ---- 数据加载（按角色）----

  async function loadState(cid = currentCharacterId) {
    try {
      const state = await API.getState(cid);
      applyMetaToSession(cid, state);
      if (currentCharacterId === cid) updatePanelsFromData(state);
    } catch (e) {
      console.error("加载状态失败:", e);
    }
  }

  async function loadHistory(cid = currentCharacterId) {
    try {
      const data = await API.getHistory(cid);
      const sess = getSession(cid);
      if (data.messages && data.messages.length > 0) {
        // 用历史替换欢迎语占位
        sess.messages = [];
        for (const m of data.messages) {
          const role = m.role === "user" ? "user" : "ai";
          sess.messages.push({
            id: ++_msgIdSeq,
            role,
            content: m.content,
            thinking: "",
            toolCalls: [],
            streaming: false,
          });
        }
        if (currentCharacterId === cid) renderAllMessages(sess);
      }
    } catch (e) {
      console.error("加载历史失败:", e);
    }
  }

  async function loadMemories(cid = currentCharacterId) {
    const sess = getSession(cid);
    sess.episodicOffset = 0;
    try {
      const data = await API.getMemories(EPISODIC_PAGE_SIZE, 0, cid);
      sess.profile = data.semantic || [];
      sess.episodic = data.episodic || [];
      sess.episodicCount = data.episodic_count || 0;
      sess.semanticCount = data.semantic_count || 0;
      if (currentCharacterId === cid) {
        updateStats({
          episodic_count: sess.episodicCount,
          semantic_count: sess.semanticCount,
        });
        renderProfile(sess.profile);
        renderEpisodic(sess.episodic, false);
        updateLoadMoreBtn(sess.episodicCount, sess.episodic.length);
      }
    } catch (e) {
      console.error("加载记忆失败:", e);
    }
  }

  async function loadMoreEpisodic() {
    const cid = currentCharacterId;
    const sess = getSession(cid);
    try {
      const data = await API.getMemories(EPISODIC_PAGE_SIZE, sess.episodicOffset, cid);
      sess.episodic = sess.episodic.concat(data.episodic || []);
      renderEpisodic(data.episodic || [], true);
      updateLoadMoreBtn(sess.episodicCount, (data.episodic || []).length);
    } catch (e) {
      console.error("加载更多失败:", e);
    }
  }

  function updateLoadMoreBtn(totalCount, currentPageSize) {
    const btn = document.getElementById("loadMoreEpisodic");
    if (!btn) return;
    const sess = getSession(currentCharacterId);
    const loaded = sess.episodicOffset + currentPageSize;
    if (loaded < totalCount) {
      btn.style.display = "block";
      btn.textContent = `加载更多（剩余 ${totalCount - loaded} 条）`;
      sess.episodicOffset = loaded;
    } else {
      btn.style.display = "none";
    }
  }

  function renderProfile(profile) {
    const el = document.getElementById("profileList");
    const countEl = document.getElementById("profileCount");
    if (!profile || profile.length === 0) {
      el.innerHTML = '<div class="empty">暂无用户画像，多聊几句让角色了解你~</div>';
      countEl.textContent = "";
      return;
    }
    countEl.textContent = `（共 ${profile.length} 条）`;
    const categoryNames = { basic: "基本信息", preference: "偏好", personality: "性格", interest: "兴趣" };
    const categoryIcons = { basic: "👤", preference: "💖", personality: "🌟", interest: "🎯" };
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
      return `<div class="profile-group"><div class="profile-cat">${icon} ${name}</div>${entries}</div>`;
    }).join("");
    el.innerHTML = html;
  }

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
    if (!append) el.innerHTML = "";
    const typeIcons = { dialogue: "💬", fact: "📌", emotion: "💓" };
    const html = memories.map(m => {
      const icon = typeIcons[m.event_type] || "📝";
      const ts = m.timestamp ? formatTime(m.timestamp) : "";
      const imp = m.importance != null
        ? `<span class="mem-importance" title="重要性">${(m.importance * 100).toFixed(0)}%</span>`
        : "";
      return `<div class="memory-item">
        <div class="mem-head">
          <span class="mem-type">${icon} ${API.escapeHtml(m.event_type || "dialogue")}</span>
          <span class="mem-time">${ts}</span>
          ${imp}
        </div>
        <div class="mem-content">${API.escapeHtml(m.content || "")}</div>
      </div>`;
    }).join("");
    el.insertAdjacentHTML("beforeend", html);
    const loadedCount = el.querySelectorAll(".memory-item").length;
    hintEl.textContent = `（已加载 ${loadedCount} 条）`;
  }

  function formatTime(ts) {
    const d = new Date(ts * 1000);
    const now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    const pad = n => String(n).padStart(2, "0");
    if (sameDay) return `今天 ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    return `${d.getMonth() + 1}/${d.getDate()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  // ---- 重置 / 忘记我（按角色）----

  async function reset() {
    const cid = currentCharacterId;
    if (!cid) return;
    const name = getCharacterName(cid);
    if (!confirm(`确定要重置与${name}的对话吗？工作记忆和情绪将清空，长期记忆保留。`)) return;
    try {
      await API.reset(cid);
      const sess = getSession(cid);
      sess.messages = [];
      sess.emotion = null;
      sess.memoryOps = [];
      sess.toolCalls = [];
      if (currentCharacterId === cid) {
        renderAllMessages(sess);
        resetEmotionPanel();
        document.getElementById("toolList").innerHTML = '<div class="empty">暂无工具调用</div>';
        document.getElementById("memoryList").innerHTML = '<div class="empty">暂无记忆操作</div>';
      }
      await loadState(cid);
    } catch (e) {
      alert("重置失败: " + e.message);
    }
  }

  async function forget() {
    const cid = currentCharacterId;
    if (!cid) return;
    const name = getCharacterName(cid);
    if (!confirm(`确定要让${name}忘记你吗？\n\n将清空：情景记忆 + 用户画像 + 情绪好感（重置到初见陌生人的态度）\n保留：工作记忆、角色知识库\n\n此操作不可恢复！`)) return;
    try {
      const res = await API.forget(cid);
      const sess = getSession(cid);
      sess.profile = [];
      sess.episodic = [];
      sess.episodicCount = 0;
      sess.semanticCount = 0;
      if (currentCharacterId === cid) {
        document.getElementById("memoryList").innerHTML = '<div class="empty">暂无记忆操作</div>';
        await loadMemories(cid);
        await loadState(cid);
        appendNotice(`${name}好像忘记了什么...\n（已清空情景记忆 ${res.episodic_deleted} 条，用户画像 ${res.semantic_deleted} 条，情绪好感已重置）`);
      }
    } catch (e) {
      alert("忘记失败: " + e.message);
    }
  }

  /** 追加一条系统提示消息到当前查看角色 */
  function appendNotice(text) {
    const sess = getSession(currentCharacterId);
    pushMessage(sess, "ai", text);
    scrollToBottom();
  }

  // ---- 工具 ----

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
  }

  // ---- 事件绑定 ----

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("input", () => {
    // 实时保存草稿到当前角色
    if (currentCharacterId) getSession(currentCharacterId).inputDraft = inputEl.value;
    autoResize();
  });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  resetBtn.addEventListener("click", reset);
  if (forgetBtn) forgetBtn.addEventListener("click", forget);

  const loadMoreBtn = document.getElementById("loadMoreEpisodic");
  if (loadMoreBtn) loadMoreBtn.addEventListener("click", loadMoreEpisodic);
  devSwitch.addEventListener("change", (e) => {
    devMode = e.target.checked;
    console.log("开发者模式:", devMode);
  });

  document.querySelectorAll(".sidebar-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".sidebar-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      document.querySelectorAll(".tab-panel").forEach(p => p.style.display = "none");
      document.getElementById(tab.dataset.panel).style.display = "block";
    });
  });

  // ---- 认证 ----
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

  let authMode = "login";

  function showAuthOverlay() {
    authOverlay.style.display = "flex";
    authError.textContent = "";
    authUsername.value = "";
    authPassword.value = "";
    authUsername.focus();
  }
  function hideAuthOverlay() { authOverlay.style.display = "none"; }

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
      authSubtitle.textContent = "注册一个新账号";
      authSubmitBtn.textContent = "注册";
      authSwitchText.textContent = "已有账号？";
      authSwitchLink.textContent = "登录";
    } else {
      authMode = "login";
      authSubtitle.textContent = "登录后开始聊天";
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
      if (authMode === "register") await API.register(username, password);
      await API.login(username, password);
      hideAuthOverlay();
      updateUserBadge(username);
      await loadCharacters();
    } catch (err) {
      authError.textContent = err.message || (authMode === "register" ? "注册失败" : "登录失败");
    } finally {
      authSubmitBtn.disabled = false;
    }
  }

  async function handleGuest() {
    hideAuthOverlay();
    updateUserBadge(null);
    await loadCharacters();
  }

  async function handleLogout() {
    try { await API.logout(); } catch (e) { console.error("登出失败:", e); }
    updateUserBadge(null);
    messagesEl.innerHTML = "";
    showAuthOverlay();
  }

  async function checkAuth() {
    try {
      const me = await API.getMe();
      if (me && me.username) {
        hideAuthOverlay();
        updateUserBadge(me.username);
        await loadCharacters();
        return;
      }
    } catch (e) {
      console.error("检查登录态失败:", e);
    }
    showAuthOverlay();
  }

  if (authForm) authForm.addEventListener("submit", handleAuthSubmit);
  if (authSwitchLink) authSwitchLink.addEventListener("click", switchAuthMode);
  if (authGuestBtn) authGuestBtn.addEventListener("click", handleGuest);
  if (logoutBtn) logoutBtn.addEventListener("click", handleLogout);

  // ---- 初始化 ----
  checkAuth();
})();

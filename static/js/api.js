/* =========================================================
 * API 辅助模块 - 封装后端接口调用
 * ========================================================= */
(function (global) {
  "use strict";

  const API = {
    /** 流式聊天（SSE）*/
    async *chatStream(message, devMode = false) {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, dev_mode: devMode }),
      });

      if (!resp.ok) throw new Error(`请求失败 (${resp.status})`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // 解析 SSE 数据
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();
            if (data && data !== "[DONE]") {
              try {
                yield JSON.parse(data);
              } catch (e) {
                // 忽略解析错误
              }
            }
          }
        }
      }
    },

    /** 获取 Agent 状态 */
    async getState() {
      const resp = await fetch("/api/state");
      return resp.json();
    },

    /** 获取对话历史（工作记忆）*/
    async getHistory() {
      const resp = await fetch("/api/history");
      return resp.json();
    },

    /** 重置 Agent */
    async reset() {
      const resp = await fetch("/api/reset", { method: "POST" });
      return resp.json();
    },

    /** 忘记我（清空长期记忆：情景记忆+用户画像）*/
    async forget() {
      const resp = await fetch("/api/forget", { method: "POST" });
      if (!resp.ok) throw new Error(`请求失败 (${resp.status})`);
      return resp.json();
    },

    /** 获取记忆列表（支持分页）*/
  async getMemories(limit = 20, offset = 0) {
    const resp = await fetch(`/api/memories?limit=${limit}&offset=${offset}`);
    return resp.json();
  },

  // ---- 认证相关 ----

  /** 注册 */
  async register(username, password) {
    const resp = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "注册失败");
    return data;
  },

  /** 登录（后端设置 HttpOnly Cookie）*/
  async login(username, password) {
    const resp = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "登录失败");
    return data;
  },

  /** 登出 */
  async logout() {
    const resp = await fetch("/api/auth/logout", { method: "POST" });
    return resp.json();
  },

  /** 获取当前登录用户（401 表示未登录）*/
  async getMe() {
    const resp = await fetch("/api/auth/me");
    if (resp.status === 401) return null;
    if (!resp.ok) throw new Error("获取用户信息失败");
    return resp.json();
  },
  };

  /** 转义 HTML */
  function escapeHtml(text) {
    if (text == null) return "";
    return String(text)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  API.escapeHtml = escapeHtml;

  /** 简易 Markdown */
  function renderMarkdown(md) {
    if (!md) return "";
    let html = escapeHtml(md);
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\n/g, "<br>");
    return html;
  }
  API.renderMarkdown = renderMarkdown;

  global.API = API;
})(window);


    // ─── State ───────────────────────────────────────────────────────────────────
    let sessionId = null;
    let conversations = [];
    let currentConversation = null;
    let isWaiting = false;

    // ─── Theme ───────────────────────────────────────────────────────────────────
    function toggleTheme() {
      const html = document.documentElement;
      const currentTheme = html.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
    }

    function loadTheme() {
      const savedTheme = localStorage.getItem('theme') || 'dark';
      document.documentElement.setAttribute('data-theme', savedTheme);
      document.getElementById('themeToggle').checked = savedTheme === 'light';
    }

    // ─── Sidebar ─────────────────────────────────────────────────────────────────
    function toggleSidebar() {
      const sidebar = document.getElementById('sidebar');
      if (window.innerWidth <= 900) {
        sidebar.classList.toggle('mobile-open');
      } else {
        sidebar.classList.toggle('collapsed');
      }
    }

    function toggleSettings() {
      document.getElementById('sidePanel').classList.toggle('open');
    }

    // ─── Conversations ───────────────────────────────────────────────────────────
    function loadConversations() {
      const saved = localStorage.getItem('conversations');
      if (saved) {
        conversations = JSON.parse(saved);
        renderConversationList();
      }
    }

    function saveConversations() {
      localStorage.setItem('conversations', JSON.stringify(conversations));
    }

    function renderConversationList() {
      const container = document.getElementById('conversationList');
      container.innerHTML = '<div class="sidebar-section-title">Recent</div>';

      conversations.slice().reverse().forEach((conv, idx) => {
        const realIdx = conversations.length - 1 - idx;
        const item = document.createElement('div');
        item.className = 'conversation-item' + (currentConversation === realIdx ? ' active' : '');
        item.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          <span class="conversation-item-text">${conv.title || 'New conversation'}</span>
          <span class="conversation-item-delete" onclick="event.stopPropagation(); deleteConversation(${realIdx})">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </span>
        `;
        item.onclick = () => loadConversation(realIdx);
        container.appendChild(item);
      });
    }

    async function startNewChat() {
      // Create new session
      try {
        const res = await fetch('/conversation/session/new', { method: 'POST' });
        if (!res.ok) {
          console.warn('Session creation failed:', res.status);
          sessionId = 'local_' + Date.now();
        } else {
          const data = await res.json();
          sessionId = data.session_id || ('local_' + Date.now());
        }
      } catch (e) {
        console.warn('Session creation error:', e);
        sessionId = 'local_' + Date.now();
      }

      // Create new conversation
      const newConv = {
        id: sessionId,
        title: 'New conversation',
        messages: [],
        currentSpec: null,
        createdAt: new Date().toISOString()
      };
      conversations.push(newConv);
      currentConversation = conversations.length - 1;
      saveConversations();
      renderConversationList();

      // Reset UI
      document.getElementById('welcomeScreen').style.display = 'flex';
      document.getElementById('messages').style.display = 'none';
      document.getElementById('messages').innerHTML = '';
      document.getElementById('userInput').value = '';

      // Close sidebar on mobile
      if (window.innerWidth <= 900) {
        document.getElementById('sidebar').classList.remove('mobile-open');
      }
    }

    function loadConversation(idx) {
      currentConversation = idx;
      const conv = conversations[idx];
      sessionId = conv.id;
      conv.currentSpec = conv.currentSpec || null;

      const messagesEl = document.getElementById('messages');
      const welcomeEl = document.getElementById('welcomeScreen');

      messagesEl.innerHTML = '';

      if (conv.messages.length === 0) {
        welcomeEl.style.display = 'flex';
        messagesEl.style.display = 'none';
      } else {
        welcomeEl.style.display = 'none';
        messagesEl.style.display = 'block';
        conv.messages.forEach(msg => {
          appendMessage(msg.role, msg.content, msg.timestamp, msg.blueprintUrl, false);
        });
      }

      renderConversationList();

      // Close sidebar on mobile
      if (window.innerWidth <= 900) {
        document.getElementById('sidebar').classList.remove('mobile-open');
      }
    }

    function deleteConversation(idx) {
      conversations.splice(idx, 1);
      saveConversations();

      if (currentConversation === idx) {
        if (conversations.length > 0) {
          loadConversation(conversations.length - 1);
        } else {
          startNewChat();
        }
      } else if (currentConversation > idx) {
        currentConversation--;
      }

      renderConversationList();
    }

    // ─── Messages ────────────────────────────────────────────────────────────────
    function formatTime(date) {
      return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function escapeHtml(text) {
      return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function renderMarkdown(content) {
      let html = escapeHtml(content || '');

      html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
      html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
      html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
      html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
      html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
      html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
      html = html.replace(/\n{2,}/g, '</p><p>');
      html = html.replace(/\n/g, '<br>');

      return `<p>${html}</p>`
        .replace(/<p><h/g, '<h')
        .replace(/<\/h([123])><\/p>/g, '</h$1>')
        .replace(/<p><ul>/g, '<ul>')
        .replace(/<\/ul><\/p>/g, '</ul>')
        .replace(/<br><\/li>/g, '</li>');
    }

    function appendMessage(role, content, timestamp, blueprintUrl, save = true) {
      const messagesEl = document.getElementById('messages');
      const welcomeEl = document.getElementById('welcomeScreen');

      welcomeEl.style.display = 'none';
      messagesEl.style.display = 'block';

      const isUser = role === 'user';
      const time = timestamp || new Date().toISOString();

      const msgRow = document.createElement('div');
      msgRow.className = `msg-row ${isUser ? 'user-row' : 'bot-row'}`;

      // Parse markdown for bot messages
      let htmlContent = content;
      if (!isUser) {
        htmlContent = renderMarkdown(content);
      } else {
        htmlContent = escapeHtml(content).replace(/\n/g, '<br>');
      }

      msgRow.innerHTML = `
        <div class="msg-container">
          <div class="msg-avatar ${isUser ? 'user-av' : 'bot-av'}">
            ${isUser ? 'U' : 'B'}
          </div>
          <div class="msg-body">
            <div class="msg-header">
              <span class="msg-name">${isUser ? 'You' : 'BlueprintGPT'}</span>
              <span class="msg-timestamp">${formatTime(time)}</span>
            </div>
            <div class="msg-text">${htmlContent}</div>
            ${blueprintUrl ? `
              <div class="blueprint-preview">
                <img src="${blueprintUrl}" alt="Floor Plan" onclick="window.open('${blueprintUrl}', '_blank')">
              </div>
            ` : ''}
            <div class="msg-actions">
              <button class="msg-action-btn" onclick="copyMessage(this, \`${content.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                Copy
              </button>
              ${!isUser ? `
                <button class="msg-action-btn" onclick="regenerateResponse()">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="1 4 1 10 7 10"/><polyline points="23 20 23 14 17 14"/><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
                  </svg>
                  Regenerate
                </button>
              ` : ''}
            </div>
          </div>
        </div>
      `;

      messagesEl.appendChild(msgRow);
      messagesEl.scrollTop = messagesEl.scrollHeight;

      // Save to conversation
      if (save && currentConversation !== null) {
        conversations[currentConversation].messages.push({
          role, content, timestamp: time, blueprintUrl
        });

        // Update title from first user message
        if (isUser && conversations[currentConversation].messages.length === 1) {
          conversations[currentConversation].title = content.substring(0, 40) + (content.length > 40 ? '...' : '');
          renderConversationList();
        }

        saveConversations();
      }
    }

    function showTypingIndicator() {
      const messagesEl = document.getElementById('messages');
      const indicator = document.createElement('div');
      indicator.className = 'typing-indicator';
      indicator.id = 'typingIndicator';
      indicator.innerHTML = `
        <div class="typing-container">
          <div class="msg-avatar bot-av">B</div>
          <div class="msg-body">
            <div class="typing-dots">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      `;
      messagesEl.appendChild(indicator);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function hideTypingIndicator() {
      const indicator = document.getElementById('typingIndicator');
      if (indicator) indicator.remove();
    }

    function copyMessage(btn, text) {
      navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        btn.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
          Copied!
        `;
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
            Copy
          `;
        }, 2000);
      });
    }

    async function regenerateResponse() {
      if (isWaiting || currentConversation === null) return;

      const conv = conversations[currentConversation];
      if (conv.messages.length < 2) return;

      // Remove last bot message
      conv.messages.pop();
      saveConversations();

      // Reload conversation
      loadConversation(currentConversation);

      // Get last user message and resend
      const lastUserMsg = conv.messages[conv.messages.length - 1];
      if (lastUserMsg && lastUserMsg.role === 'user') {
        await sendToBackend(lastUserMsg.content);
      }
    }

    // ─── Input Handling ──────────────────────────────────────────────────────────
    function autoResize(textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    function handleKeyDown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }

    function useSuggestion(text) {
      document.getElementById('userInput').value = text;
      sendMessage();
    }

    async function sendMessage() {
      const input = document.getElementById('userInput');
      const message = input.value.trim();

      if (!message || isWaiting) return;

      // Ensure we have a conversation
      if (currentConversation === null) {
        await startNewChat();
      }

      // Add user message
      appendMessage('user', message);
      input.value = '';
      input.style.height = 'auto';

      await sendToBackend(message);
    }

    async function sendToBackend(message) {
      isWaiting = true;
      document.getElementById('sendBtn').disabled = true;
      showTypingIndicator();

      try {
        const currentConv = currentConversation !== null ? conversations[currentConversation] : null;
        const priorHistory = currentConv
          ? currentConv.messages.slice(0, -1).slice(-20).map(msg => ({
              role: msg.role,
              content: msg.content,
              timestamp: msg.timestamp
            }))
          : [];

        const plotWidth = parseFloat(document.getElementById('plotWidth').value) || 12;
        const plotHeight = parseFloat(document.getElementById('plotHeight').value) || 15;
        const entranceX = parseFloat(document.getElementById('entranceX').value) || plotWidth / 2;
        const entranceY = parseFloat(document.getElementById('entranceY').value) || 0;

        const response = await fetch('/conversation/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: message,
            session_id: sessionId,
            history: priorHistory,
            client_spec: currentConv ? currentConv.currentSpec : null,
            boundary: { width: plotWidth, height: plotHeight },
            entrance_point: [entranceX, entranceY],
            generate: true
          })
        });

        // Parse response JSON
        const data = await response.json();

        hideTypingIndicator();

        // Check for HTTP errors
        if (!response.ok) {
          // Server returned an error - show the detail message
          const errorMessage = data.detail || `Server error (${response.status})`;
          appendMessage('assistant', `An error occurred: ${errorMessage}`);
          console.error('Server error:', response.status, data);
          isWaiting = false;
          document.getElementById('sendBtn').disabled = false;
          return;
        }

        // Check if the response has assistant_text (successful response)
        if (data.assistant_text) {
          // Extract blueprint URL if available
          let blueprintUrl = null;
          if (data.designs && data.designs.length > 0) {
            const design = data.designs[0];
            if (design.artifact_urls && design.artifact_urls.svg) {
              blueprintUrl = design.artifact_urls.svg;
            }
          }

          appendMessage('assistant', data.assistant_text, null, blueprintUrl);
        } else {
          // No assistant text in response - show a fallback
          appendMessage('assistant', 'I processed your request but have no response to display.');
        }

        // Update session ID if provided
        if (data.session_id) {
          sessionId = data.session_id;
          if (currentConversation !== null) {
            conversations[currentConversation].id = sessionId;
            if (data.current_spec) {
              conversations[currentConversation].currentSpec = data.current_spec;
            }
            saveConversations();
          }
        }

      } catch (error) {
        hideTypingIndicator();
        console.error('Network/Parse error:', error);
        // Network or JSON parsing error - show helpful message
        appendMessage('assistant', 'Unable to connect to the server. Please check that the server is running and try again.');
      }

      isWaiting = false;
      document.getElementById('sendBtn').disabled = false;
    }

    // ─── Initialization ──────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
      loadTheme();
      loadConversations();

      if (conversations.length === 0) {
        startNewChat();
      } else {
        loadConversation(conversations.length - 1);
      }
    });
  
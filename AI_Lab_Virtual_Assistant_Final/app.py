"""
Hybrid AI Virtual Assistant Studio (Flask Web App)
- Runs cleanly on port 5050 by default so old cached servers on port 5000 never interfere!
- Works out-of-the-box with ZERO API keys (Dictionary + DuckDuckGo + Smart Wikipedia + Local NLP).
- Optional Gemini 2.5 Flash activation box in the sidebar (auto-saves to `.env` when activated).
Run: python app.py
"""

from flask import Flask, render_template_string, request, jsonify
from assistant import HybridVirtualAssistant

app = Flask(__name__)
bot = HybridVirtualAssistant(use_voice_output=False)

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Hybrid AI Virtual Assistant (Final)</title>
  <style>
    :root {
      --bg: #090d16;
      --panel: #111827;
      --card: #1f2937;
      --border: #374151;
      --accent: #38bdf8;
      --green: #22c55e;
      --text: #f3f4f6;
      --muted: #9ca3af;
      --user-bg: linear-gradient(135deg, #2563eb, #4f46e5);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
    body {
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: grid;
      grid-template-columns: 295px 1fr;
      overflow: hidden;
    }
    .sidebar {
      background: var(--panel);
      border-right: 1px solid var(--border);
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--accent);
    }
    .dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 10px var(--green);
    }
    .section-title {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .skill-list { display: flex; flex-direction: column; gap: 6px; }
    .skill-btn {
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 12px;
      border-radius: 10px;
      text-align: left;
      font-size: 0.83rem;
      cursor: pointer;
      transition: 0.2s;
    }
    .skill-btn:hover { border-color: var(--accent); transform: translateX(3px); }
    .api-card {
      background: #0b1120;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .api-card input {
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 7px 10px;
      border-radius: 8px;
      font-size: 0.8rem;
      width: 100%;
    }
    .api-card button {
      background: var(--accent);
      color: #090d16;
      border: none;
      padding: 7px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.8rem;
      cursor: pointer;
    }
    .telemetry-box {
      margin-top: auto;
      background: #0b1120;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      font-size: 0.78rem;
      color: var(--muted);
      line-height: 1.65;
    }
    .telemetry-box span { color: var(--accent); font-weight: 600; }

    .main { display: flex; flex-direction: column; height: 100vh; }
    .topbar {
      padding: 14px 24px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .mode-pill {
      font-size: 0.75rem;
      background: rgba(56, 189, 248, 0.14);
      color: var(--accent);
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .controls { display: flex; gap: 8px; }
    .toggle-btn {
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.8rem;
      cursor: pointer;
    }
    .toggle-btn.active { border-color: var(--accent); color: var(--accent); }

    .chat {
      flex: 1;
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .msg-wrap { display: flex; flex-direction: column; max-width: 78%; }
    .msg-wrap.user { align-self: flex-end; align-items: flex-end; }
    .msg-wrap.bot { align-self: flex-start; align-items: flex-start; }
    .bubble {
      padding: 13px 17px;
      border-radius: 16px;
      font-size: 0.94rem;
      line-height: 1.55;
      white-space: pre-wrap;
      box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }
    .user .bubble { background: var(--user-bg); border-bottom-right-radius: 4px; }
    .bot .bubble { background: var(--card); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
    .meta { font-size: 0.72rem; color: var(--muted); margin-top: 5px; padding: 0 4px; }

    .input-bar {
      padding: 16px 24px;
      background: var(--panel);
      border-top: 1px solid var(--border);
      display: flex;
      gap: 10px;
    }
    .input-bar input {
      flex: 1;
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 12px 16px;
      border-radius: 12px;
      font-size: 0.95rem;
      outline: none;
    }
    .input-bar input:focus { border-color: var(--accent); }
    .icon-btn, .send-btn {
      border: none;
      border-radius: 12px;
      padding: 0 18px;
      font-weight: 600;
      cursor: pointer;
    }
    .icon-btn { background: var(--card); color: var(--text); border: 1px solid var(--border); }
    .icon-btn.recording { background: #dc2626; color: white; }
    .send-btn { background: var(--accent); color: #090d16; }

    @media (max-width: 800px) {
      body { grid-template-columns: 1fr; }
      .sidebar { display: none; }
    }
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="brand"><div class="dot"></div> Hybrid AI Assistant</div>

    <div>
      <div class="section-title">General, Dictionary & Everyday</div>
      <div class="skill-list">
        <button class="skill-btn" onclick="quickSend('Explain meaning of rigorous')">📖 Meaning of 'rigorous'</button>
        <button class="skill-btn" onclick="quickSend('Why is the sky blue?')">🌍 Why is the sky blue?</button>
        <button class="skill-btn" onclick="quickSend('How does photosynthesis work?')">🌱 How photosynthesis works</button>
        <button class="skill-btn" onclick="quickSend('Tell me more')">🔄 Follow-up: Tell me more</button>
      </div>
    </div>

    <div>
      <div class="section-title">Smart Utility Skills</div>
      <div class="skill-list">
        <button class="skill-btn" onclick="quickSend('Weather in Kota')">🌤️ Live Weather (Kota)</button>
        <button class="skill-btn" onclick="quickSend('Latest news')">📰 Top News Headlines</button>
        <button class="skill-btn" onclick="quickSend('Calculate sqrt(256) + sin(30) * 10')">🧮 Scientific Calculator</button>
        <button class="skill-btn" onclick="quickSend('Convert 42 km to miles')">📏 Unit Converter</button>
        <button class="skill-btn" onclick="quickSend('System status')">💻 System Diagnostics</button>
      </div>
    </div>

    <div class="api-card">
      <div class="section-title" style="margin:0;">Optional: Gemini 2.5 Flash</div>
      <input type="password" id="apiKeyInput" placeholder="Paste GEMINI_API_KEY..." />
      <button onclick="saveApiKey()">Activate & Save Key</button>
      <div id="keyStatus" style="font-size:0.74rem; color:var(--muted);">Works without key via Dictionary/Wiki/Web QA</div>
    </div>

    <div class="telemetry-box">
      <div>Brain: <span id="tMode">{{ brain_mode }}</span></div>
      <div>Last Route: <span id="tIntent">—</span></div>
      <div>Confidence: <span id="tConf">—</span></div>
      <div>Latency: <span id="tLat">—</span></div>
    </div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div style="display:flex; align-items:center; gap:10px;">
        <strong>AI Lab Final Project</strong>
        <span class="mode-pill" id="topModePill">{{ brain_mode }}</span>
      </div>
      <div class="controls">
        <button class="toggle-btn" id="ttsToggle" onclick="toggleTTS()">🔊 Voice Out: OFF</button>
        <button class="toggle-btn" onclick="clearChat()">Clear</button>
      </div>
    </div>

    <div class="chat" id="chat">
      <div class="msg-wrap bot">
        <div class="bubble">Hello! I am your final Hybrid Virtual Assistant. Ask me word definitions ("explain meaning of rigorous"), general questions ("why is the sky blue?"), or run utility commands (Weather, News, Scientific Math, Unit Converter, To-Do List, System Status)!</div>
        <div class="meta">Tier-1 Ensemble NLP + Tier-2 Dictionary / Web / Gemini Engine</div>
      </div>
    </div>

    <form class="input-bar" onsubmit="sendQuery(event)">
      <button type="button" class="icon-btn" id="micBtn" onclick="startVoice()" title="Voice Input">🎙️</button>
      <input type="text" id="promptInput" placeholder="Ask anything — e.g. 'Explain meaning of rigorous', 'Why is the sky blue?', 'Weather in Kota'..." autocomplete="off" required />
      <button type="submit" class="send-btn">Send</button>
    </form>
  </main>

  <script>
    const chat = document.getElementById('chat');
    const promptInput = document.getElementById('promptInput');
    let ttsEnabled = false;

    function toggleTTS() {
      ttsEnabled = !ttsEnabled;
      const btn = document.getElementById('ttsToggle');
      btn.textContent = ttsEnabled ? '🔊 Voice Out: ON' : '🔊 Voice Out: OFF';
      btn.classList.toggle('active', ttsEnabled);
    }

    function speakText(text) {
      if (!ttsEnabled || !('speechSynthesis' in window)) return;
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(utter);
    }

    function addBubble(text, role, metaText = '') {
      const wrap = document.createElement('div');
      wrap.className = 'msg-wrap ' + role;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;
      wrap.appendChild(bubble);
      if (metaText) {
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = metaText;
        wrap.appendChild(meta);
      }
      chat.appendChild(wrap);
      chat.scrollTop = chat.scrollHeight;
    }

    function clearChat() { chat.innerHTML = ''; }
    function quickSend(text) { promptInput.value = text; sendQuery(new Event('submit')); }

    async function saveApiKey() {
      const key = document.getElementById('apiKeyInput').value.trim();
      const statusEl = document.getElementById('keyStatus');
      if (!key) {
        statusEl.textContent = 'Please paste a key first.';
        return;
      }
      statusEl.textContent = 'Validating key...';
      const res = await fetch('/set_key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key })
      });
      const data = await res.json();
      statusEl.textContent = data.message;
      statusEl.style.color = data.ok ? '#22c55e' : '#f87171';
      if (data.ok) {
        document.getElementById('tMode').textContent = data.brain_mode;
        document.getElementById('topModePill').textContent = data.brain_mode;
      }
    }

    async function sendQuery(e) {
      e.preventDefault();
      const q = promptInput.value.trim();
      if (!q) return;
      addBubble(q, 'user');
      promptInput.value = '';

      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q })
      });
      const data = await res.json();

      const meta = `Route: ${data.intent} • Confidence: ${(data.confidence * 100).toFixed(1)}% • ${data.latency_ms}ms`;
      addBubble(data.response, 'bot', meta);
      speakText(data.response);

      document.getElementById('tIntent').textContent = data.intent;
      document.getElementById('tConf').textContent = (data.confidence * 100).toFixed(1) + '%';
      document.getElementById('tLat').textContent = data.latency_ms + ' ms';
    }

    function startVoice() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        alert('Browser Speech Recognition not supported. Please use Chrome or Edge.');
        return;
      }
      const recognition = new SpeechRecognition();
      const micBtn = document.getElementById('micBtn');
      recognition.lang = 'en-US';
      micBtn.classList.add('recording');
      recognition.onresult = (event) => {
        promptInput.value = event.results[0][0].transcript;
        sendQuery(new Event('submit'));
      };
      recognition.onend = () => micBtn.classList.remove('recording');
      recognition.start();
    }
  </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_UI, brain_mode=bot.get_brain_mode())

@app.route("/set_key", methods=["POST"])
def set_key():
    key = request.get_json().get("api_key", "")
    ok, msg = bot.set_api_key(key)
    return jsonify({"ok": ok, "message": msg, "brain_mode": bot.get_brain_mode()})

@app.route("/chat", methods=["POST"])
def chat():
    query = request.get_json().get("query", "")
    return jsonify(bot.process_query(query))

if __name__ == "__main__":
    print(f"Starting Hybrid AI Assistant Studio ({bot.get_brain_mode()}) at http://127.0.0.1:5050")
    app.run(debug=False, port=5050)

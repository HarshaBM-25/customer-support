// script.js — Interactive Client-side Logic for Amazon Support Chatbot

const chatLauncher = document.getElementById("chat-launcher");
const chatModal = document.getElementById("chat-modal");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");

function toggleChat() {
  const isHidden = chatModal.classList.contains("hidden");
  if (isHidden) {
    chatModal.classList.remove("hidden");
    chatLauncher.style.display = "none";
    chatInput.focus();
  } else {
    chatModal.classList.add("hidden");
    chatLauncher.style.display = "flex";
  }
}

function quickAsk(text) {
  if (chatModal.classList.contains("hidden")) {
    toggleChat();
  }
  chatInput.value = text;
  handleSend(new Event("submit"));
}

function clearChat() {
  chatMessages.innerHTML = `
    <div class="message-wrapper agent-msg">
      <div class="msg-avatar"><i class="fa-brands fa-amazon"></i></div>
      <div class="msg-bubble">
        <p>Chat session refreshed. How can I assist you today?</p>
        <div class="suggested-prompts">
          <button onclick="quickAsk('Where is my package? The tracking number is 9876543210')">Where is my order?</button>
          <button onclick="quickAsk('I received a broken item and need a replacement')">Return a damaged item</button>
          <button onclick="quickAsk('Why was I charged twice for Prime membership?')">Billing inquiry</button>
        </div>
      </div>
    </div>
  `;
}

function appendMessage(sender, text, meta = null) {
  const wrapper = document.createElement("div");
  wrapper.className = `message-wrapper ${sender === "user" ? "user-msg" : "agent-msg"}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.innerHTML = sender === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-brands fa-amazon"></i>';

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  const p = document.createElement("p");
  p.innerText = text;
  bubble.appendChild(p);

  // If agent returned diagnostic metadata (Intent, Routing, RAG cases)
  if (meta && sender === "agent") {
    const metaContainer = document.createElement("div");
    metaContainer.className = "meta-tags";

    const intentTag = document.createElement("span");
    intentTag.className = "tag tag-intent";
    intentTag.innerText = `Intent: ${meta.intent} (${meta.confidence * 100}%)`;
    metaContainer.appendChild(intentTag);

    const routeTag = document.createElement("span");
    routeTag.className = `tag ${meta.route === "AUTO" ? "tag-route-auto" : "tag-route-escalate"}`;
    routeTag.innerText = `Route: ${meta.route}`;
    metaContainer.appendChild(routeTag);

    if (meta.retrieved_cases && meta.retrieved_cases.length > 0) {
      const ragTag = document.createElement("span");
      ragTag.className = "tag tag-rag";
      ragTag.innerText = `RAG: ${meta.retrieved_cases.join(", ")}`;
      metaContainer.appendChild(ragTag);
    }

    bubble.appendChild(metaContainer);
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.id = "typing-indicator";
  wrapper.className = "message-wrapper agent-msg";
  wrapper.innerHTML = `
    <div class="msg-avatar"><i class="fa-brands fa-amazon"></i></div>
    <div class="msg-bubble">
      <div class="typing-dots">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    </div>
  `;
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
  const indicator = document.getElementById("typing-indicator");
  if (indicator) {
    indicator.remove();
  }
}

async function handleSend(e) {
  if (e) e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  // 1. Render User Message
  appendMessage("user", message);
  chatInput.value = "";
  chatInput.disabled = true;

  // 2. Show Typing Dots
  showTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message })
    });

    removeTypingIndicator();
    const data = await response.json();

    if (response.ok && data.status === "success") {
      appendMessage("agent", data.reply, {
        intent: data.intent,
        confidence: data.confidence,
        route: data.route,
        retrieved_cases: data.retrieved_cases
      });
    } else {
      appendMessage("agent", data.reply || "Sorry, an error occurred while processing your request.");
    }
  } catch (err) {
    removeTypingIndicator();
    appendMessage("agent", "Could not connect to the AI support backend. Please ensure the server is running on http://localhost:5000.");
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
  }
}

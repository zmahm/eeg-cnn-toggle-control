let selectedModel = null;
let currentServoIndex = 0;
let servo3State = false;
let ws = null;

const MODEL_API = '/select_model';
const MODEL_LIST_API = '/';
const WEBSOCKET_URL = 'ws://192.168.4.1:81';

window.onload = () => {
  fetchAvailableModels();
  setupEventListeners();
};

function fetchAvailableModels() {
  fetch(MODEL_LIST_API)
    .then(() => {
      const dropdown = document.getElementById('model-select');
      if (dropdown.options.length === 0) {
        logToConsole("No models found.");
      } else {
        document.getElementById('current-model').textContent = dropdown.options[0].text;
      }
    });
}

function setupEventListeners() {
  document.getElementById('load-model').addEventListener('click', () => {
    const modelName = document.getElementById('model-select').value;
    if (!modelName) return;

    fetch(MODEL_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName })
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) throw new Error(data.error);
        selectedModel = modelName;
        document.getElementById('current-model').textContent = modelName;
        connectWebSocket();
        logToConsole(`Model '${modelName}' loaded.`);
      })
      .catch(err => logToConsole('Error: ' + err.message));
  });

  document.querySelectorAll('button[data-action]').forEach(btn => {
    btn.addEventListener('click', () => handleAction(btn.dataset.action));
  });
}

function connectWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return;

  ws = new WebSocket(WEBSOCKET_URL);

  ws.onopen = () => {
    document.getElementById('ws-status').textContent = 'Connected';
    logToConsole('WebSocket connected.');
  };

  ws.onerror = (err) => {
    document.getElementById('ws-status').textContent = 'Connection Failed';
    logToConsole('WebSocket error. Ensure ESP8266 is reachable.');
  };

  ws.onclose = () => {
    document.getElementById('ws-status').textContent = 'Disconnected';
    logToConsole('WebSocket connection closed.');
  };

  ws.onmessage = (event) => {
    logToConsole('ESP8266: ' + event.data);
  };
}

function handleAction(action) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    logToConsole('WebSocket not connected.');
    return;
  }

  switch (action) {
    case 'leg_movement':
      currentServoIndex = (currentServoIndex + 1) % 3;
      document.getElementById('servo-index').textContent = currentServoIndex;
      logToConsole(`Switched to servo ${currentServoIndex}`);
      break;

    case 'left_hand':
      sendToggleCommand(currentServoIndex, "towards_min");
      break;

    case 'right_hand':
      sendToggleCommand(currentServoIndex, "towards_max");
      break;

    case 'both_hands':
      servo3State = !servo3State;
      const direction = servo3State ? "towards_max" : "towards_min";
      sendToggleCommand(3, direction);
      document.getElementById('servo3-state').textContent = servo3State ? 'max' : 'min';
      logToConsole(`Toggled Servo 3 to ${servo3State ? 'max' : 'min'}`);
      break;
  }
}

function sendToggleCommand(servoId, direction = "") {
  const payload = {
    action: "toggle_servo",
    servoId: servoId,
    speed: "medium"
  };

  if (direction) payload.direction = direction;

  try {
    ws.send(JSON.stringify(payload));
    logToConsole(`Sent to ESP8266: toggle_servo -> Servo ${servoId} ${direction ? `(${direction})` : ""}`);
  } catch (err) {
    logToConsole('Failed to send command: ' + err.message);
  }
}

function logToConsole(message) {
  const logArea = document.getElementById('log-area');
  const timestamp = new Date().toLocaleTimeString();
  logArea.textContent += `[${timestamp}] ${message}\n`;
  logArea.scrollTop = logArea.scrollHeight;
}

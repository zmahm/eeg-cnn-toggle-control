let selectedModel = null;
let ws = null;
let lastPrediction = "";

const STATUS_API = "/status";
const MODEL_API = "/select_model";
const WEBSOCKET_URL = "ws://192.168.4.1:81";
const PREDICTION_POLL_INTERVAL = 500; // ms

window.onload = () => {
  setupEventListeners();
  pollPrediction();
};

function setupEventListeners() {
  document.getElementById("load-model").addEventListener("click", () => {
    const modelName = document.getElementById("model-select").value;
    if (!modelName) return;

    fetch(MODEL_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName }),
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) throw new Error(data.error);
        selectedModel = modelName;
        document.getElementById("current-model").textContent = modelName;
        logToConsole(`Model '${modelName}' loaded successfully.`);
        connectWebSocket();
      })
      .catch(err => logToConsole("Error: " + err.message));
  });
}

function connectWebSocket() {
  ws = new WebSocket(WEBSOCKET_URL);

  ws.onopen = () => {
    document.getElementById("ws-status").textContent = "Connected";
    logToConsole("WebSocket connected.");
  };

  ws.onerror = () => {
    document.getElementById("ws-status").textContent = "Connection Failed";
    logToConsole("WebSocket error — continuing without connection.");
  };

  ws.onclose = () => {
    document.getElementById("ws-status").textContent = "Disconnected";
    logToConsole("WebSocket connection closed.");
  };

  ws.onmessage = (event) => {
    logToConsole("ESP8266: " + event.data);
  };
}

function pollPrediction() {
  setInterval(() => {
    fetch(STATUS_API)
      .then(res => res.json())
      .then(data => {
        const prediction = data.current_prediction || "none";
        document.getElementById("current-prediction").textContent = prediction;

        // Avoid spamming the same prediction
        if (prediction !== lastPrediction) {
          handlePrediction(prediction);
          lastPrediction = prediction;
        }
      })
      .catch(() => {
        logToConsole("Warning: Failed to fetch prediction.");
      });
  }, PREDICTION_POLL_INTERVAL);
}

function handlePrediction(prediction) {
  logToConsole(`Prediction: ${prediction}`);

  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const actionMap = {
    left_hand: { servoId: 1, direction: "towards_min" },
    right_hand: { servoId: 2, direction: "towards_max" },
    both_hands: { servoId: 3, direction: "towards_max" },
    leg_movement: { servoId: 0, direction: "towards_max" },
    rest: null, // no action for "rest"
  };

  const action = actionMap[prediction];
  if (!action) return;

  const payload = {
    action: "toggle_servo",
    speed: "medium",
    ...action,
  };

  try {
    ws.send(JSON.stringify(payload));
    logToConsole(`Sent command: Servo ${action.servoId} (${action.direction})`);
  } catch (err) {
    logToConsole("Error sending WebSocket command: " + err.message);
  }
}

function logToConsole(message) {
  const logArea = document.getElementById("log-area");
  const timestamp = new Date().toLocaleTimeString();
  logArea.textContent += `[${timestamp}] ${message}\n`;
  logArea.scrollTop = logArea.scrollHeight;
}

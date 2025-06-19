document.addEventListener("DOMContentLoaded", () => {
  const startButton = document.getElementById("start-button");
  const statusText = document.getElementById("status");
  const taskText = document.getElementById("task");
  const trialText = document.getElementById("trial");
  const countdownText = document.getElementById("countdown");

  let pollingInterval;

  startButton.addEventListener("click", () => {
    fetch("/start", {
      method: "POST"
    });
    startButton.disabled = true;
    statusText.textContent = "Starting...";

    pollingInterval = setInterval(() => {
      fetch("/status")
        .then((res) => res.json())
        .then((data) => {
          statusText.textContent = data.status;
          taskText.textContent = data.current_class || "N/A";
          trialText.textContent = data.trial;
          countdownText.textContent = data.countdown.toFixed(1);

          if (data.status === "completed") {
            clearInterval(pollingInterval);
            startButton.disabled = false;
            statusText.textContent = "Completed";

            // Send a reset request so user can restart sampling again
            fetch("/reset", {
              method: "POST"
            });
          }

        });
    }, 200);
  });
});
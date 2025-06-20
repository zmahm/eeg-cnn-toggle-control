import asyncio
import websockets
import json

# WebSocket connection handler for sending commands to the ESP8266
class ArmWebSocketClient:
    def __init__(self, host="ws://192.168.4.1:81"):
        self.uri = host
        self.connection = None

    async def connect(self):
        # Establish the WebSocket connection
        self.connection = await websockets.connect(self.uri)
        print(f"Connected to ESP8266 WebSocket at {self.uri}")

    async def send_command(self, command_dict):
        # Convert the command to JSON and send it to the ESP8266
        if self.connection is None:
            raise RuntimeError("WebSocket connection not established")

        message = json.dumps(command_dict)
        await self.connection.send(message)
        print(f"Sent: {message}")

    async def close(self):
        if self.connection is not None:
            await self.connection.close()
            print("WebSocket connection closed")


# High-level control wrapper for managing servo toggling state and sending mapped actions
class ArmController:
    def __init__(self, websocket_client):
        self.client = websocket_client
        self.current_servo = 0  # Start with servo 0
        self.max_servo_index = 3  # Only cycling through servo 0 to 3

    async def handle_prediction(self, prediction_class):
        if prediction_class == "leg_movement":
            self.current_servo = (self.current_servo + 1) % (self.max_servo_index + 1)
            print(f"Switched to controlling servo {self.current_servo}")

        elif prediction_class == "left_hand":
            await self.client.send_command({
                "action": "toggle_servo",
                "servoId": self.current_servo,
                "direction": "min",
                "speed": "slow"
            })

        elif prediction_class == "right_hand":
            await self.client.send_command({
                "action": "toggle_servo",
                "servoId": self.current_servo,
                "direction": "max",
                "speed": "slow"
            })

        elif prediction_class == "both_hands":
            await self.client.send_command({
                "action": "toggle_servo",
                "servoId": 4,
                "speed": "slow"
            })

        elif prediction_class == "rest":
            # Do nothing for rest
            pass

        else:
            print(f"Unknown prediction class: {prediction_class}")

from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
h, w = 480, 640
canvas = np.zeros((h, w, 3), dtype=np.uint8)
prev_x, prev_y = None, None
current_color = (0, 255, 0)  # Default color is green
writing_direction = "ltr"  # Default writing direction: left-to-right
is_writing = True  # Flag to control writing

def is_fist(hand_landmarks):
    # Check if fingertips are close to the palm
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    mcp_joints = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP,
    ]

    for tip, mcp in zip(tips, mcp_joints):
        tip_x = hand_landmarks.landmark[tip].x
        tip_y = hand_landmarks.landmark[tip].y
        mcp_x = hand_landmarks.landmark[mcp].x
        mcp_y = hand_landmarks.landmark[mcp].y

        # Calculate distance between tip and MCP joint
        distance = ((tip_x - mcp_x) ** 2 + (tip_y - mcp_y) ** 2) ** 0.5
        if distance > 0.1:  # Adjust threshold as needed
            return False
    return True

def generate_frames():
    global prev_x, prev_y, canvas, current_color, writing_direction, is_writing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Check if the hand is in a fist
                if is_fist(hand_landmarks):
                    is_writing = False
                else:
                    is_writing = True

                if is_writing:
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    if prev_x and prev_y:
                        if writing_direction == "rtl":
                            cv2.line(canvas, (x, y), (prev_x, prev_y), current_color, 5)
                        else:
                            cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, 5)

                    prev_x, prev_y = x, y

        combined_image = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)
        ret, buffer = cv2.imencode('.jpg', combined_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_canvas')
def clear_canvas():
    global canvas, prev_x, prev_y
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    prev_x, prev_y = None, None
    return "Canvas cleared"

@app.route('/set_color/<color>')
def set_color(color):
    global current_color
    if color == "red":
        current_color = (0, 0, 255)
    elif color == "green":
        current_color = (0, 255, 0)
    elif color == "blue":
        current_color = (255, 0, 0)
    return f"Color set to {color}"

@app.route('/set_direction/<direction>')
def set_direction(direction):
    global writing_direction
    if direction in ["ltr", "rtl"]:
        writing_direction = direction
        print(f"Writing direction set to {direction}")  # Debugging
        return f"Writing direction set to {direction}"
    return "Invalid direction"

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)
h, w = 480, 640  # Frame dimensions
canvas = np.zeros((h, w, 3), dtype=np.uint8)  # Drawing canvas
prev_x, prev_y = None, None  # Previous finger position
current_color = (0, 255, 0)  # Default color is green
writing_direction = "ltr"  # Default writing direction: left-to-right
is_writing = True  # Flag to control writing

# Define color palette areas (x, y, width, height)
color_palette = {
    "red": (50, 400, 50, 50),
    "green": (110, 400, 50, 50),
    "blue": (170, 400, 50, 50),
}

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

        # Define color values for the palette
        palette_colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
        }

        # Draw color palette
        for color_name, (x, y, width, height) in color_palette.items():
            color_value = palette_colors[color_name]
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 0), 2)  # Border
            cv2.rectangle(image, (x + 2, y + 2), (x + width - 2, y + height - 2), color_value, -1)  # Fill

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip position
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Check if finger is over the color palette
                for color_name, (palette_x, palette_y, width, height) in color_palette.items():
                    if palette_x <= x <= palette_x + width and palette_y <= y <= palette_y + height:
                        current_color = palette_colors[color_name]
                        break

                # Draw on the canvas
                if prev_x and prev_y:
                    if writing_direction == "rtl":
                        cv2.line(canvas, (x, y), (prev_x, prev_y), current_color, 5)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, 5)

                prev_x, prev_y = x, y

        # Combine the canvas and video frame
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

@app.route('/set_direction/<direction>')
def set_direction(direction):
    global writing_direction
    if direction in ["ltr", "rtl"]:
        writing_direction = direction
        return f"Writing direction set to {direction}"
    return "Invalid direction"

@app.route('/get_selected_color')
def get_selected_color():
    color_map = {
        (0, 0, 255): "red",
        (0, 255, 0): "green",
        (255, 0, 0): "blue",
    }
    return jsonify({"color": color_map.get(tuple(current_color), "green")})

if __name__ == '__main__':
    app.run(debug=True)
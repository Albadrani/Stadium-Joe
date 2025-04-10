from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
import threading
import base64
import numpy as np
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO('yolov8n.pt')
video_source = 0


def video_processing():
    cap = cv2.VideoCapture('output.avi')

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break


        results = model(frame, classes=[0])
        count = len(results[0].boxes)
        annotated_frame = results[0].plot()


        _, buffer = cv2.imencode('.jpg', cv2.resize(annotated_frame, (640, 480)))
        frame_base64 = base64.b64encode(buffer).decode('utf-8')


        gate_status = [
            {
                'text': 'مفتوح' if count  < 10 else 'ازدحام',
                'color': '#00FF00' if count  < 10 else '#FF0000'
            },
            {
                'text': 'تدفق عادي' if 1 > 0.5 else 'ازدحام',
                'color': '#00FF00' if 1 > 0.6 else '#FFA500'
            }
        ]

        # إرسال البيانات
        socketio.emit('update', {
            'count': count,
            'frame': f'data:image/jpeg;base64,{frame_base64}',
            'gate_status': gate_status
        })

    cap.release()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/control')
def control_panel():
    return render_template('index.html')
@app.route('/sport')
def sport_panel():
    return render_template('sport.html')
@app.route('/nofai')
def nofai_panel():
    return render_template('nofai.html')
@app.route('/enter')
def enter_panel():
    return render_template('enter.html')
@app.route('/health')
def health_panel():
    return render_template('health.html')


@socketio.on('connect')

def handle_connect():
    threading.Thread(target=video_processing).start()


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
from flask import Flask, Response
import cv2
import threading
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

frame_lock = threading.Lock()
global_frame = None
prev_frame = None

motion_sensitivity = 5000
record_duration = 60  # in seconds

save_folder = 'save'
nomove_folder = 'nomove'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(save_folder)
ensure_dir(nomove_folder)

def handle_video(video_writer, folder, start_time, temp_video_path, motion_detected):
    global prev_frame
    video_writer.release()
    if motion_detected:
        timestamp = start_time.strftime('%Y-%m-%d %H-%M-%S')
        new_path = os.path.join(folder, f"{timestamp}.mp4")
        os.rename(temp_video_path, new_path)
        print(f"Saved: {new_path}")
    else:
        os.remove(temp_video_path)
        print(f"Deleted: {temp_video_path}")
    prev_frame = None

def capture_frames():
    global global_frame, prev_frame
    temp_video_path = 'temp_video.mp4'
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (640, 480))
    start_time = datetime.now()
    motion_detected = False
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if prev_frame is not None:
            frame_delta = cv2.absdiff(prev_frame, gray_frame)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < motion_sensitivity:
                    continue
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if (datetime.now() - start_time).seconds >= record_duration:
                handle_video(video_writer, save_folder if motion_detected else nomove_folder, start_time, temp_video_path, motion_detected)
                video_writer = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (640, 480))
                start_time = datetime.now()
                motion_detected = False

        prev_frame = gray_frame
        video_writer.write(frame)

        with frame_lock:
            global_frame = frame

def generate_frames():
    while True:
        with frame_lock:
            if global_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', global_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=25565, debug=False)

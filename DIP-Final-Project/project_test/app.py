from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)
camera = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def get_camera():
    global camera
    if not camera:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
    return camera

def get_reference_frame():
    ret, reference_frame = get_camera().read()
    if not ret:
        print("Error: Failed to grab initial frame.")
        camera.release()
        raise RuntimeError('Failed to grab initial frame.')
    return reference_frame

def generate_frames():
    camera = get_camera()
    reference_frame = get_reference_frame()
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    ref_hsv = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2HSV)
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff_gray = cv2.absdiff(ref_gray, gray_frame)
        diff_sat = cv2.absdiff(ref_hsv[:, :, 1], hsv_frame[:, :, 1])
        _, thresh_gray = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        _, thresh_sat = cv2.threshold(diff_sat, 25, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_or(thresh_gray, thresh_sat)
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        foreground = cv2.bitwise_and(frame, frame, mask=combined_mask)

        # Concatenate original frame and foreground frame
        spacer = np.ones((frame.shape[0], 10, 3), dtype=np.uint8) * 255  # white spacer
        combined_frame = np.hstack((frame, spacer, foreground))

        # Adding titles below the frames
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_frame, 'Original Frame', (50, combined_frame.shape[0] - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_frame, 'Foreground Frame', (frame.shape[1] + 60, combined_frame.shape[0] - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', combined_frame)
        frame_data = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)

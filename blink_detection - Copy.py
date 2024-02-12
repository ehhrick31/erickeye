import cv2
import time
import serial

# Initialize the face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Serial communication setup
com_port = 'COM4'  # Adjust as needed
ser = serial.Serial(com_port, 9600)

# Blink detection settings
blink_counter = 0
minute_timer_start = time.time()

# Variables to manage blink detection state



eye_detected_in_previous_frame = False
eye_detection_reset_threshold = 3  # Frames to wait before considering another blink
eye_detection_reset_counter = 0

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            # No faces detected in this frame
            eye_detected_in_previous_frame = False
            eye_detection_reset_counter = 0
            continue  # Move to the next frame

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(eyes) == 0 and eye_detected_in_previous_frame:
                eye_detection_reset_counter += 1
                if eye_detection_reset_counter >= eye_detection_reset_threshold:
                    blink_counter += 1
                    eye_detected_in_previous_frame = False
                    eye_detection_reset_counter = 0
            elif len(eyes) > 0:
                eye_detected_in_previous_frame = True
                eye_detection_reset_counter = 0
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # Check if a minute has passed
        current_time = time.time()
        if current_time - minute_timer_start >= 60:
            if blink_counter < 6:
                ser.write(b'B')  # Signal for low blink rate
            blink_counter = 0  # Reset blink counter
            minute_timer_start = current_time  # Reset timer

        # Display the current blink count
        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    ser.close()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("mask_detector.h5")

def predict_on_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break
        orig = frame.copy()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))  
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)[0]
        mask_prob, no_mask_prob = prediction
        if mask_prob > no_mask_prob:
            label = f"Mask ({mask_prob*100:.2f}%)"
            color = (0, 255, 0)
        else:
            label = f"No Mask ({no_mask_prob*100:.2f}%)"
            color = (0, 0, 255)
        cv2.putText(orig, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Live Mask Detection", orig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_on_webcam()
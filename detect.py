import base64
import cv2
import numpy as np

def load_model(model_file):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_file)
    return recognizer

# if __name__ == "__main__":
def detect(image):
    model_file = "model.yml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    model = load_model(model_file)

    # Now you can use the face_cascade for face detection on new images
    # For example:
    image = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.putText(image, result, (x - 5, y - 5), 1, (255, 255, 200))

        roi_gray = gray[y:y + h, x:x + w]
        label, confidence = model.predict(roi_gray)
        print('label: ', label, 'confidence: ', confidence, '%')    

    # only use at local (not at server)
    # scale_factor = 0.35  # You can adjust this value as needed
    # height, width = image.shape[:2]
    # resized_image = cv2.resize(image, (int(width * scale_factor), int(height * (scale_factor-0.05))))
    # cv2.imshow("Detected Faces", resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    _, img_encoded = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    return label, confidence, img_base64

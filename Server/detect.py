import base64
import cv2
import numpy as np
import face_recognition

from Server import globalVariables

model = None

def load_model(model_file):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_file)
    return recognizer

def showOnScreen(image):
    height, width = image.shape[:2]
    scale_factor = 614/height  # You can adjust this value as needed

    resize_image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
    
    cv2.imshow("Detected Faces", resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect(image):
    global model
 
    if globalVariables.isModelChanged:    
        model = load_model(globalVariables.model_file)
        globalVariables.isModelChanged = False

    image = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray, gray)    
    # faces = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")    
    faces = face_recognition.face_locations(image) # model = hog by default

    if len(faces) > 0:        
        (left, top, right, bottom) = globalVariables.getMaxFace(faces)
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        # gray = cv2.resize(gray, (200, 200))
        gray = gray[top:bottom, left:right]
    else:
        gray = gray    
    
    label, distance = model.predict(gray)
    print('distance', distance)
    fontSize = 3*image.shape[1]/800
    color = (255, 255, 0)
    weight = 3

    pos = (left - 5, top - 7) if len(faces) > 0 else (30, 30)

    if distance < globalVariables.MAX_DISTANCE: 
        cv2.putText(image, f'Id: {str(label)}', pos, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, weight)
        confidence = (1 - distance/globalVariables.MAX_DISTANCE) * 100
        print('label: ', label, 'confidence: ', confidence, '%')    

        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        # only use at local (not at server)
        # showOnScreen(image)

        return True, {'label': label, 'confidence': confidence}, img_base64
    else:
        cv2.putText(image, f'Khong biet', pos, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, weight)

        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        # only use at local (not at server)
        # showOnScreen(image)

        return False, 'Không nhận diện được', img_base64


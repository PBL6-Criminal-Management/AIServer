import base64
import cv2
import numpy as np

from Server import globalVariables

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
    model = load_model(globalVariables.model_file)

    image = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray, gray)
    faces = globalVariables.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # detect all faces in image
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #     roi_gray = gray[y:y + h, x:x + w]
    #     label, confidence = model.predict(roi_gray)
    #     cv2.putText(image, f'Id: {str(label)}', (x - 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
    #     print('label: ', label, 'confidence: ', confidence, '%') 

    if len(faces) > 0:        
        (x, y, w, h) = faces[0]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # gray = cv2.resize(gray, (200, 200))
        gray = gray[y:y + h, x:x + w]
    else:
        gray = gray    
    
    label, distance = model.predict(gray)

    fontSize = 3*image.shape[1]/800
    color = (255, 255, 0)
    weight = 3

    pos = (x - 5, y - 7) if len(faces) > 0 else (30, 30)

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


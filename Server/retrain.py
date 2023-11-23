import os
import cv2
import numpy as np
from PIL import Image

from Server.download_file import download_folder

from Server import globalVariables

def load_images_and_labels_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        dir_path = os.path.join(folder,label)

        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                img = np.array(Image.open(img_path))
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)
                    labels.append(label)

    return images, labels

def create_model():
    model = cv2.face.LBPHFaceRecognizer_create()
    return model

def prepare_data(images, labels):
    faces = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray, gray)

        face_image = globalVariables.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        if len(face_image) > 0:
            (x, y, w, h) = face_image[0]
            face_image = gray[y:y + h, x:x + w]
        else:
            face_image = gray

        face_image = cv2.resize(face_image, (200, 200))

        faces.append(face_image)

    return np.asarray(faces), np.asarray(labels, dtype=np.int32)

def train_and_save_model(images, labels, output_model_file):
    faces, labels = prepare_data(images, labels)

    model = create_model()
    model.train(faces, labels)
    model.save(output_model_file)

def retrain():
    # not save files into system
    images, labels = download_folder(globalVariables.TrainedImages_folder_id, globalVariables.dataset_folder)

    # save files (Consumes a lot of memory)
    # count = download_folder(globalVariables.sample_folder_id, globalVariables.dataset_folder, True)
    # print(f'Download file counts: {count}')
    # if not os.path.exists(globalVariables.dataset_folder):
    #     return 0
    # images, labels = load_images_and_labels_from_folder(globalVariables.dataset_folder)

    if len(labels) > 0:
        train_and_save_model(images, labels, globalVariables.model_file)
        globalVariables.isTrain = True
        globalVariables.isModelChanged = True

    return len(labels)


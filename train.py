import os
import cv2
import numpy as np
from PIL import Image

from download_file import download_folder

import globalVariables

def load_images_and_labels_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        dir_path = os.path.join(folder,label)

        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                img = np.array(Image.open(img_path))
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
        gray = cv2.cvtColor(cv2.resize(image, (200, 200)), cv2.COLOR_BGR2GRAY)
        faces.append(gray)

    return np.asarray(faces), np.asarray(labels, dtype=np.int32)

def train_and_save_model(images, labels, output_model_file):
    faces, labels = prepare_data(images, labels)

    model = create_model()
    model.train(faces, labels)
    model.save(output_model_file)

# if __name__ == "__main__":
def retrain():
    dataset_folder = "TrainedImages"
    output_model_file = "model.yml"

    #id of TrainedImages folder: 1FcPN4UNVUZHO7JL5MPSfCgIxmhmY9LC5
    #id of Sample folder: 1PyzPF-1vIPPzWtXApsLsir8kCAt7Fpyb

    # not save files into system
    images, labels = download_folder('1PyzPF-1vIPPzWtXApsLsir8kCAt7Fpyb', dataset_folder)

    # save files (Consumes a lot of memory)
    # count = download_folder('1FcPN4UNVUZHO7JL5MPSfCgIxmhmY9LC5', dataset_folder, True)
    # print(f'Download file counts: {count}')
    # images, labels = load_images_and_labels_from_folder(dataset_folder)

    train_and_save_model(images, labels, output_model_file)
    globalVariables.isTrain = True

    return len(labels)


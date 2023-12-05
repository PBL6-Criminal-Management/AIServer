import os
import cv2

def init():
    global isTrain, isModelChanged, model_file, cascade_name, dataset_folder, TrainedImages_folder_id, sample_folder_id, Model_folder_id, ALLOWED_IMAGE_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_DISTANCE, face_cascade
    model_file = "Server/model.yml"
    cascade_name = "haarcascade_frontalface_default.xml"
    dataset_folder = "Server/TrainedImages"

    #id of TrainedImages folder: 1FcPN4UNVUZHO7JL5MPSfCgIxmhmY9LC5
    #id of Sample folder: 1PyzPF-1vIPPzWtXApsLsir8kCAt7Fpyb
    TrainedImages_folder_id = '1FcPN4UNVUZHO7JL5MPSfCgIxmhmY9LC5'
    sample_folder_id = '1PyzPF-1vIPPzWtXApsLsir8kCAt7Fpyb'
    Model_folder_id = '12oueKnaRRY4b-47WoKz04IehUhWPJnji'

    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_FILE_SIZE_MB = 50  # Set your maximum allowed file size in megabytes
    MAX_DISTANCE = 95

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)

    if os.path.exists(model_file):
        isTrain = True
    else:
        isTrain = False

    isModelChanged = True
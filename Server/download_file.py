import io
import os
import cv2
from google.oauth2 import service_account
from googleapiclient.discovery import build

from googleapiclient.http import MediaIoBaseDownload
import numpy as np

imageCounts = 0
images = []
labels = []

def download_file(file_id, file_name, drive_service): #download file from gg drive  
    global imageCounts

    request = drive_service.files().get_media(fileId=file_id)
    fh = open(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% file {file_name}.")

    imageCounts += 1


#Get file instead of save to folder
def get_file(file_id, folder_name, drive_service):
    global images
    global labels

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    image_bytes = np.asarray(bytearray(fh.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    images.append(image)
    labels.append(folder_name)

# Get all pages (cause pagination)
def download_all_pages_of_folder(folder_id, folder, drive_service, isSave = False):
    if not os.path.exists(folder) and isSave:
        os.mkdir(folder)

    page_token = None
    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()

        files = response.get('files', [])
        if not files:
            break

        for file in files:
            file_id = file['id']
            file_name = file['name']
            if not isSave:
                get_file(file_id, folder, drive_service)
            else:
                download_file(file_id, f'{folder}/{file_name}', drive_service)

        page_token = response.get('nextPageToken', None)
        if not page_token:
            break

def download_folder(folder_id, local_folder_path, isSave = False):
    global imageCounts, images, labels
    imageCounts = 0
    images = []
    labels = []

    credentials = service_account.Credentials.from_service_account_file(
        'Server/exalted-pattern-400909-3eaa10f4b2b4.json',
        scopes=['https://www.googleapis.com/auth/drive']
    )

    drive_service = build('drive', 'v3', credentials=credentials)

    if not os.path.exists(local_folder_path) and isSave:
        os.mkdir(local_folder_path)

    page_token = None
    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()

        folders = response.get('files', [])
        if not folders:
            break

        for folder in folders:
            mime_type = folder['mimeType']
            if mime_type == 'application/vnd.google-apps.folder': #only download if it is a subfolder
                folder_id = folder['id']
                folder_name = folder['name']
                if not isSave:
                    download_all_pages_of_folder(folder_id, folder_name, drive_service, isSave)
                else:
                    download_all_pages_of_folder(folder_id, os.path.join(local_folder_path, folder_name), drive_service, isSave)

        page_token = response.get('nextPageToken', None)
        if not page_token:
            break
    
    if not isSave:
        return images, labels
    else:
        return imageCounts
    

    
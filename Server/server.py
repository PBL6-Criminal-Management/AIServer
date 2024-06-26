from flask import Flask, request, jsonify

from Server.detect import detect
from Server.retrain import retrain

from Server import globalVariables

from Server.download_file import download_model

app = Flask(__name__)

globalVariables.init()          # Call only once

def is_image(file):
    # Define image file signatures (magic numbers)
    image_signatures = {
        b'\xFF\xD8\xFF': 'jpeg',
        b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A': 'png',
        b'\x47\x49\x46\x38\x37\x61': 'gif'
        # Add more signatures for other image formats as needed
    }

    # Read the first few bytes of the file
    file_signature = file.read(8)
    file.seek(0) #back to start

    # Check if the file signature matches any known image formats
    for signature, format_name in image_signatures.items():
        if file_signature.startswith(signature):
            return format_name
            
    return None

def allowed_file(file):
    return '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in globalVariables.ALLOWED_IMAGE_EXTENSIONS and is_image(file) in globalVariables.ALLOWED_IMAGE_EXTENSIONS

def is_file_size_allowed(file):
    # Check if the file size is within the allowed limit
    fileLength = len(file.read())
    # print(fileLength, flush=True)
    file.seek(0)
    return fileLength <= globalVariables.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes

@app.route('/detect', methods=['POST'])
def detect_face():
    if 'CriminalImage' not in request.files:
        return jsonify({"error": "Không có ảnh nào được cung cấp!"}), 400
    
    file = request.files['CriminalImage']
    if file.filename is None or not file.filename:
        return jsonify({"error": "Không có ảnh nào được cung cấp!"}), 400
    
    if not allowed_file(file):
        return jsonify({"error": "File bạn tải lên không phải là 1 ảnh hoặc phần đuôi mở rộng không hợp lệ!"}), 400
    
    if not is_file_size_allowed(file):
        return jsonify({"error": f"File bạn tải lên vượt quá kích thước cho phép ({globalVariables.MAX_FILE_SIZE_MB} MB)!"}), 400

    if not globalVariables.isTrain:
        if download_model('model.yml', globalVariables.Model_folder_id):
            globalVariables.isTrain = True
            globalVariables.isModelChanged = True
        else:
            return jsonify({"message": "Mô hình AI chưa được huấn luyện!"}), 200

    isPredictable, result, image = detect(file)

    if isPredictable:
        return jsonify({'isPredictable': isPredictable, 'label': result['label'], 'confidence': result['confidence'], 'image': image}), 200
    else:
        return jsonify({'isPredictable': isPredictable, 'result': result, 'image': image}), 200

@app.route('/retrain', methods=['POST'])
def retrain_model():
    count = retrain()
    if count == 0:
        return jsonify({"message": "Không tìm thấy ảnh nào để huấn luyện mô hình!"}), 200
    else:
        return jsonify({"message": f"Mô hình AI được huấn luyện thành công! Đã huấn luyện {count} ảnh!"}), 200

if __name__ == '__main__':
    app.run(debug=True)
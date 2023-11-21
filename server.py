from flask import Flask, request, jsonify

from detect import detect
from train import retrain

import globalVariables

app = Flask(__name__)

globalVariables.init()          # Call only once

@app.route('/detect', methods=['POST'])
def detect_face():
    if 'CriminalImages' not in request.files:
        return jsonify({"error": "Không có ảnh nào được cung cấp"}), 400

    if not globalVariables.isTrain:
        return jsonify({"message": "Mô hình huấn luyện chưa được train!"})

    image_file = request.files['CriminalImages']    
    label, confidence, image = detect(image_file)

    return jsonify({'label': label, 'confidence': confidence, 'image': image}), 400
    

@app.route('/retrain', methods=['POST'])
def retrain_model():
    return jsonify({"message": f"Đã train {retrain()} ảnh"}), 400

if __name__ == '__main__':
    app.run(debug=True)
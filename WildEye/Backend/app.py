from flask import Flask, request, jsonify
from flask_cors import CORS
import torch  # Make sure you have PyTorch installed
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the YOLO model from the local cloned repository
model = torch.hub.load('./yolov5', 'custom', path='../train/best.pt', source='local')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['image'].read()
    image = Image.open(io.BytesIO(file))

    # Perform inference
    results = model(image)

    # Process results (adjust this based on your needs)
    predictions = results.pandas().xyxy[0].to_dict(orient='records')  # Convert to dictionary format

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
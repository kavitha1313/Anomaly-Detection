# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from waitress import serve
# from inference import detect_video_anomaly


# app = Flask(__name__)
# CORS(app)

# LABELS_MAP = {
#     0: "Abuse",         
#     1: "Arrest",
#     2: "Arson",
#     3: "Assault",
#     4: "Burglary",
#     5: "Explosion",
#     6: "Fighting",
#     7: "NormalVideos",
#     8: "RoadAccidents",
#     9: "Robbery",
#     10: "Shooting",
#     11: "Shoplifting",
#     12: "Stealing",
#     13: "Vandalism"
# }

# @app.route('/predict', methods=['POST'])
# def predict():
#     print(request.files)
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     try:
#         file_extension = file.filename.split('.')[-1].lower()
        
#         if file_extension in ['mp4', 'avi', 'mov']:
#             results = detect_video_anomaly(file)
#             return jsonify({
#                 'overall_prediction': results['overall_prediction'],
#                 'frame_predictions': results['frame_predictions'],
#                 'confidence_scores': results['confidence_scores'],
#                 'file_type': 'video'
#             })
#         else:
#             return jsonify({'error': 'Unsupported file type. Please upload a video file (mp4, avi, or mov)'}), 400

#     except Exception as e:
#         app.logger.error(f"Error processing file: {str(e)}")
#         return jsonify({'error': f'Error processing file: {str(e)}'}), 500

# if __name__ == '__main__':
#     env = 'dev'
#     port = int(os.environ.get('PORT', 5000))
    
#     print(f"Starting server in {env} mode on port {port}")
#     print(f"Available classes: {list(LABELS_MAP.values())}")
    
#     if env == 'dev':
#         app.run(host='0.0.0.0', port=port, debug=True)
#     else:
#         serve(app, host='0.0.0.0', port=port)


import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from inference import detect_video_anomaly

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Allow up to 500MB uploads

LABELS_MAP = {
    0: "Abuse", 1: "Arrest", 2: "Arson", 3: "Assault", 4: "Burglary",
    5: "Explosion", 6: "Fighting", 7: "NormalVideos", 8: "RoadAccidents",
    9: "Robbery", 10: "Shooting", 11: "Shoplifting", 12: "Stealing", 13: "Vandalism"
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension in ['mp4', 'avi', 'mov']:
            results = detect_video_anomaly(file)
            return jsonify({
                'overall_prediction': results['overall_prediction'],
                 'frame_predictions': results['frame_predictions'],
                 'confidence_scores': results['confidence_scores'],
                 'file_type': 'video'
            })
        else:
            return jsonify({'error': 'Unsupported file type. Upload mp4, avi, or mov'}), 400

    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    serve(app, host='0.0.0.0', port=port)

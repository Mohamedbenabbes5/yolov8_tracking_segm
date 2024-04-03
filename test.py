from flask import Flask, request, jsonify
import requests
import os
app = Flask(__name__)
shared = r"C:\Users\abbes\Documents\developpement\BackEndProjectPFE\shared"

@app.route('/process_video', methods=['POST'])
def processVideo():
    try:
        # Récupérer le nom de la vidéo envoyée dans la requête POST
        video_name = request.json.get('video_name')
        print(video_name)
       
        
        # Retourner une réponse réussie
        return jsonify({'status': 'success', 'message': 'Video processing started'})
    
    except Exception as e:
        # Retourner une réponse d'erreur en cas de problème
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

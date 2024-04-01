from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/send_image', methods=['POST'])
def send_image():
    # Chemin vers l'image sur le disque
    image_path = r"C:\Users\abbes\Desktop\369ef49e63f6bc41bc87ad55eb8d479b.jpg"

    # Envoyer l'image au serveur Node.js
    url = 'http://localhost:3000/receive_image'
    files = {'image_file': open(image_path, 'rb')}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to send image to Node.js'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

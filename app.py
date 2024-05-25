from flask import Flask, render_template, jsonify
import subprocess
import threading

app = Flask(__name__)

def run_facial_detection():
    # This is where you call your facial detection script
    subprocess.run(['python', 'emotion.py'], check=True)

def run_visualization():

    subprocess.run(['python', 'visualizacao.py'], check=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute-script', methods=['POST'])
def execute_script():
    try:
        # Start the facial detection in a separate thread to avoid blocking
        threading.Thread(target=run_facial_detection).start()
        return jsonify({'status': 'success', 'message': 'Script started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
@app.route('/execute-vis', methods=['POST'])
def execute_vis():
    try:
        # Start the facial detection in a separate thread to avoid blocking
        threading.Thread(target=run_visualization).start()
        return jsonify({'status': 'success', 'message': 'Script started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
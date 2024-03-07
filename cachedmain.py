import random
import os
import magic
from flask import Flask, request, jsonify
from redis import Redis
from rq import Queue
import whisper
from resdis_connection import host
import torch

#torch.cuda.empty_cache()

AUDIO_SAVE_PATH = './audio'

app = Flask(__name__)
queue = Queue(connection=Redis(host), default_timeout=3600)
id_to_job = {}

# 'large-v2'
supported_models = ['large-v2']
models = {model_name: whisper.load_model(model_name) for model_name in supported_models}

def transcribe_job(job_id, audio_data, model_name, language):
    try:
        extension = 'mp3' if 'mpeg' in magic.from_buffer(audio_data).lower() else 'wav'
        filepath = os.path.join(AUDIO_SAVE_PATH, f'{job_id}.{extension}')
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        model = models[model_name]
        result = model.transcribe(filepath, language=language)
        os.remove(filepath)
        return {'ok': True, 'result': result}
    except Exception as e:
        return {'ok': False, 'error': e}

from main import transcribe_job
@app.route('/api/transcribe', methods=['POST'])
def transcribe_endpoint():
    job_id = str(0)
    while job_id in id_to_job:
        job_id = str(random.randint(0, 1000000000000))
    model_name = request.args.get('model') or 'large-v2'
    if model_name not in supported_models:
        return jsonify({'error': "Wrong model name"})
    language = request.args.get('language') or 'uk'
    audio_data = request.data
    if not audio_data:
        return jsonify({'error': "Request without audiofile"})
    j = queue.enqueue(transcribe_job, job_id, audio_data, model_name, language)
    id_to_job[job_id] = j
    return jsonify({'call-id': job_id})

@app.route('/api/status/<job_id>', methods=['GET'])
def status_endpoint(job_id):
    if not job_id in id_to_job:
        return jsonify({'error': 'Ivalid call id'})
    j = id_to_job[job_id]
    return jsonify({'finished': bool(j.return_value()), 'tasks': queue.count})
    
@app.route('/api/audio/<job_id>', methods=['GET'])
def audio_endpoint(job_id):
    if not job_id in id_to_job:
        return jsonify({'error': 'Ivalid call id'})
    result = id_to_job[job_id].return_value()
    if not result:
        return {'error': "Task is not finished"}
    if not result['ok']:
        return jsonify({'error': f"Error in worker: {result['error']}"})
    return jsonify(result['result'])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
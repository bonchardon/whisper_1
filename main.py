import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from redis import Redis
from rq import Queue
from resdis_connection import host
#import torch
from jobs import transcribe_job

#torch.cuda.empty_cache()

AUDIO_SAVE_PATH = './audio'

app = Flask(__name__)
CORS(app)
queue = Queue(connection=Redis(host), default_timeout=3600)
id_to_job = {}

supported_models = {'tiny','base','small', 'medium', 'large', 'large-v2'}


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
    channel = request.args.get('channel')
    if channel not in ['left', 'right', None]:
        return jsonify({'error': "Invalid channel name"})
    j = queue.enqueue(transcribe_job, job_id, audio_data, model_name, language, channel)
    j.execute_failure_callback(app.logger.error)
    id_to_job[job_id] = j
    return jsonify({'call-id': job_id})

@app.route('/api/status/<job_id>', methods=['GET'])
def status_endpoint(job_id):
    if not job_id in id_to_job:
        return jsonify({'error': 'Ivalid call id'})
    j = id_to_job[job_id]
    resp = {'finished': bool(j.return_value()), 'tasks': queue.count, 'failed': j.is_failed}
    if j.is_failed:
        resp['failed'] = True
    return jsonify(resp)
    
@app.route('/api/audio/<job_id>', methods=['GET'])
def audio_endpoint(job_id):
    if not job_id in id_to_job:
        return jsonify({'error': 'Ivalid call id'})
    result = id_to_job[job_id].return_value()
    if not result:
        return {'error': "Task is not finished"}
    if not result['ok']:
        return jsonify({'error': f"Error in worker: {result['error']}",
                        'filetype': str(result["filetype"])})
    return jsonify(result['result'])


if __name__ == '__main__':
    # port=5000
    app.run(host="0.0.0.0", port=5000)
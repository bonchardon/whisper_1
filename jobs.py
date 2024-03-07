import os
import whisper
import magic
import numpy as np
import ffmpeg

AUDIO_SAVE_PATH = './audio'


def load_audio_channel(file: str, channel: str, sr: int = 16000):
    try:
        channel = 'FL' if channel == 'left' else 'FR'
        out, _ = (
            ffmpeg.input(file, threads=0)
            .filter('pan', **{"mono|c0":channel})
            .output("-", format="s16le", acodec="pcm_s16le", ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def transcribe_job(job_id, audio_data, model_name, language, channel = None):
    try:
        filetype = magic.from_buffer(audio_data).lower()
        extension = 'mp3' if 'mpeg' in filetype else 'wav'
        filepath = os.path.join(AUDIO_SAVE_PATH, f'{job_id}.{extension}')
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        model = whisper.load_model(model_name)
        if not channel:
            audio = whisper.load_audio(filepath)
        else:
            audio = load_audio_channel(filepath, channel=channel)
        result = whisper.transcribe(model, audio, language=language, word_timestamps=True,
                                    fp16=True, beam_size=5, temperature=tuple(np.arange(0, 1.0 + 1e-6, 0.2)),
                                    best_of=5)
        # result = model.transcribe(filepath, language=language, word_timestamps=True,
        #                           best_of=5, beam_size=None, condition_on_previous_text=True,
        #                           temperature=(0.0,0.2,0.4,0.6,0.8,1.0,1.2))
        os.remove(filepath)
        return {'ok': True, 'result': result}
    except Exception as e:
        return {'ok': False, 'error': e, 'filetype': filetype}

# def transcribe_job(job_id, audio_data, model_name, language, channel = None):
#     try:
#         filetype = magic.from_buffer(audio_data).lower()
#         extension = 'mp3' if 'mpeg' in filetype else 'wav'
#         filepath = os.path.join(AUDIO_SAVE_PATH, f'{job_id}.{extension}')
#         N_SAMPLES = 480000
#
#         with open(filepath, 'wb') as f:
#             f.write(audio_data)
#         model = whisper.load_model(model_name)
#         if not channel:
#             audio = whisper.load_audio(filepath)
#         else:
#             audio = load_audio_channel(filepath, channel=channel)
#
#         # in case a file is less < 30 sec
#         if audio.size < 480000:
#             pad_widths = [(0, 0)] * audio.ndim
#             pad_widths = (0, N_SAMPLES - audio.size)
#             audio = np.pad(audio, pad_widths)
#
#         # make log-Mel spectrogram and move to the same device as the model
#         mel = whisper.log_mel_spectrogram(audio[-480000:]).to(model.device)
#         result = whisper.decode(model, mel, language=language, word_timestamps=True,
#                                     fp16=True, beam_size=5, temperature=tuple(np.arange(0, 1.0 + 1e-6, 0.2)),
#                                     best_of=5)
#         # result = whisper.transcribe(model, audio, language=language, word_timestamps=True,
#         #                           fp16=True, beam_size=5, temperature=tuple(np.arange(0, 1.0 + 1e-6, 0.2)),
#         #                            best_of=5)
#         # result = model.transcribe(filepath, language=language, word_timestamps=True,
#         #                           best_of=5, beam_size=None, condition_on_previous_text=True,
#         #                           temperature=(0.0,0.2,0.4,0.6,0.8,1.0,1.2))
#         os.remove(filepath)
#         return {'ok': True, 'result': result}
#     except Exception as e:
#         return {'ok': False, 'error': e, 'filetype': filetype}




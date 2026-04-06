import numpy as np
import audioop

def mu_to_pcm(audio_bytes):
    pcm16 = audioop.ulaw2lin(audio_bytes, 2)
    audio = np.frombuffer(pcm16, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    return audio

def pcm_to_mu(audio_bytes):
    audio_int = (audio_bytes * 32767).astype(np.int16)
    return audioop.lin2ulaw(audio_int.tobytes(), 2)
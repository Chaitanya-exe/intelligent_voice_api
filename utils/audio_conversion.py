import numpy as np
import audioop

def mu_to_pcm(audio_bytes):
    return audioop.ulaw2lin(audio_bytes, 2)
    
def pcm_to_mu(audio_bytes):

    if audio_bytes.dtype != np.int16:
        audio_int = (audio_bytes * 32767).astype(np.int16)
    else:
        audio_int = audio_bytes

    return audioop.lin2ulaw(audio_int.tobytes(), 2)
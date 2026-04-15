import numpy as np
import audioop

def mu_to_pcm(audio_bytes):
    pcm = audioop.ulaw2lin(audio_bytes, 2)
    audio_np = np.frombuffer(pcm, np.int16)
    audio_np = audio_np.astype(np.float32) / 32768.0
    return audio_np

    
def pcm_to_mu(audio):

    if hasattr(audio, "detach"):  
        audio = audio.detach().cpu().numpy()

    audio_int = (audio * 32767).astype(np.int16)
    return audioop.lin2ulaw(audio_int.tobytes(), 2)
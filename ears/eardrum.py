from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
from queue import Queue
from threading import Thread

class STTPipeline:
    def __init__(self):
        self.ears = WhisperModel("small", device="auto", compute_type="int8")
        self.mic_q = Queue()

    def _record_audio(self, seconds:int = 4, samplerate: int = 16000):

        print("Please start speaking...")

        audio = sd.rec(
            int(seconds * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='float32'
        )

        sd.wait()

        audio = np.squeeze(audio)

        return audio
    
    def get_transcription(self):
        audio = self._record_audio()

        segments, info = self.ears.transcribe(
            audio=audio,
            beam_size=1,
            language='en',
            condition_on_previous_text=False,
            vad_filter=True
        )

        text = " ".join(seg.text for seg in segments)
        print("Detected language: ", info.language)
        print("User: ", text)
        return text
    
    def start(self):
        pass
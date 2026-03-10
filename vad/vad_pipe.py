import sounddevice as sd
import numpy as np
from silero_vad import load_silero_vad
from silero_vad.utils_vad import VADIterator
from conversation.controller import ConversationController
from queue import Queue

class VadPipeline:
    def __init__(self, controller: ConversationController, speech_q: Queue, samplerate:int = 16000):
        self.samplerate = samplerate
        self.q = speech_q
        self.model = load_silero_vad()
        self.vad = VADIterator(model=self.model)
        self.buffer = []
        self.controller = controller

    def process_audio(self, audio):
        speech = self.vad(audio)

        if speech:
            self.controller.start_user()
            self.buffer.append(speech)
        else:
            if len(self.buffer) > 1:
                segment = np.concatenate(self.buffer)
                self.controller.stop_user()
                self.q.put(segment)
                self.buffer = []
    
    def start(self):
        def callback(indata, frames, time, status):
            audio = indata[:,0]
            self.process_audio(audio)
        
        stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=512,
            channels=1,
            callback=callback
        )

        stream.start()
        print("Listening...")

        while True:
            sd.sleep(1000)
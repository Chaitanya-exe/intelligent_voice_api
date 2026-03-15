import sounddevice as sd
import numpy as np
from silero_vad import load_silero_vad
from silero_vad.utils_vad import VADIterator
from conversation.controller import ConversationController
from queue import Queue

class VadPipeline:
    def __init__(self, controller: ConversationController, speech_q: Queue, samplerate=16000):

        self.controller = controller
        self.q = speech_q
        self.samplerate = samplerate

        self.model = load_silero_vad()
        self.vad = VADIterator(self.model)

        # rolling audio buffer
        self.audio_buffer = np.zeros(0, dtype=np.float32)

        # speech start position
        self.speech_start = None

    def process_audio(self, audio):

        audio = audio.astype(np.float32)

        # append to rolling buffer
        self.audio_buffer = np.concatenate((self.audio_buffer, audio))

        result = self.vad(audio)

        if result is None:
            return
        
        if self.controller.ai_speaking:
            self.audio_buffer = np.zeros(0, dtype=np.float32)
            return
        
        # speech started
        if "start" in result:

            if self.controller.ai_speaking:
                print("Interrupt detected...")
                self.controller.stop_ai()
                self.audio_buffer = np.zeros(0, dtype=np.float32)
                self.vad.reset_states()
                self.speech_start = None
                return 
                

            print("Speech started")
            self.controller.start_user()
            self.speech_start = result["start"]

        # speech ended
        if "end" in result and self.speech_start is not None:

            end = result["end"]

            if self.speech_start is None:
                return
            
            if end <= self.speech_start:
                return

            segment = self.audio_buffer[self.speech_start:end]

            print("Speech segment length:", segment.shape)

            self.controller.stop_user()

            if len(segment) > 4000:
                self.q.put(segment)

            self.speech_start = None
            self.vad.reset_states()

            if end < len(self.audio_buffer):
                self.audio_buffer = self.audio_buffer[end:]
            else:
                self.audio_buffer = np.zeros(0, dtype=np.float32)

    def start(self):

        def callback(indata, frames, time, status):

            audio = indata[:, 0]
            audio = np.squeeze(audio)

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
import sounddevice as sd
import numpy as np
from silero_vad import load_silero_vad
from silero_vad.utils_vad import VADIterator
from ai_pipeline.controller import ConversationController
from queue import Queue

class VadPipeline:
    def __init__(self, controller: ConversationController, input_q: Queue, output_q: Queue):

        self.controller = controller
        self.input_q = input_q
        self.output_q = output_q
        self.buffer = []
        self.temp = np.zeros(0, dtype=np.float32)
        self.model = load_silero_vad()
        self.vad = VADIterator(self.model)
        self.is_speaking = False

    def worker(self):
        while True:
            audio = self.input_q.get()
            audio = np.squeeze(audio)

            if audio is None:
                continue

            self.temp = np.concatenate((self.temp, audio))
            CHUNK_SIZE = 512

            while len(self.temp) >= CHUNK_SIZE: 

                chunk = self.temp[:CHUNK_SIZE]
                self.temp = self.temp[CHUNK_SIZE:]
                chunk = chunk.astype(np.float32)
                result = self.vad(chunk)

                if result is None:
                    continue

                if self.controller.ai_speaking and "start" in result:
                    print("Interrupt Detected")
                    self.controller.stop_ai()
                    self.buffer = []
                    self.vad.reset_states()
                    self.is_speaking = False
                    continue

                if "start" in result:
                    print("Speech started")
                    self.buffer = []
                    self.is_speaking = True
                    self.controller.start_user()
                
                if self.is_speaking:
                    self.buffer.append(audio)

                if "end" in result and self.is_speaking:
                    print("speech end")
                    segment = np.concatenate(self.buffer)
                    print(len(segment))

                    print("processing for transcription")
                    self.output_q.put(segment)
                    
                    self.buffer = []
                    self.is_speaking = False
                    self.controller.stop_user()
                    self.vad.reset_states()
                

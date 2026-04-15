from .brain_voice import BrainVoice
from .vad_pipe import VadPipeline
from .eardrum import EarDrum
from .controller import ConversationController
from queue import Queue
import threading

class VoiceSession:

    def __init__(self):
        self.text_q, self.speech_q, self.audio_q = Queue(), Queue(), Queue()
        self.sid = ""
        controller = ConversationController()

        self.vad = VadPipeline(
            speech_q=self.speech_q,
            controller=controller
        )

        self.ears = EarDrum(
            speech_q=self.speech_q,
            text_q=self.text_q
        )

        self.brain = BrainVoice(
            text_q=self.text_q,
            controller=controller,
            audio_q=self.audio_q
        )

    def set_sid(self, call_sid: str):
        self.sid = call_sid

    def start(self):
        threading.Thread(target=self.vad.start, daemon=True).start()
        threading.Thread(target=self.ears.worker, daemon=True).start()
        threading.Thread(target=self.brain.llm_worker, daemon=True).start()
        threading.Thread(target=self.brain.tts_worker, daemon=True).start()

from faster_whisper import WhisperModel
from queue import Queue

class EarDrum:
    def __init__(self, speech_q: Queue, text_q: Queue):
        self.model = WhisperModel("small", device="auto", compute_type="int8")
        self.speech_q = speech_q
        self.text_q = text_q

    def worker(self):

        while True:

            audio = self.speech_q.get()

            segments, _ = self.model.transcribe(
                audio,
                beam_size=1,
                language="en",
                condition_on_previous_text=False
            )

            text = " ".join(seg.text for seg in segments)

            self.text_q.put(text)

            self.speech_q.task_done()
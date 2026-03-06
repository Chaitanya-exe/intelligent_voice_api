from queue import Queue
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
import sounddevice as sd
from kokoro import KPipeline
import threading
from langchain_google_genai import ChatGoogleGenerativeAI
import time

class BrainVoice:
    def __init__(self):
        self.voice = KPipeline(lang_code='h')
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.q = Queue()
        self.speaker_stream = sd.OutputStream(
            channels=1,
            samplerate=24000,
            dtype="float32"
        )
        self.speaker_stream.start()
    
    def tokenise_sentences(self, conversation):
        sentence_buffer = ""

        for chunk in self.model.stream(conversation):
            
            token = chunk.content or ""

            print(token, end="", flush=True)
            
            sentence_buffer += token

            if sentence_buffer.endswith(("।", ".", "?", "!", "।")):
                
                clean = sentence_buffer.strip()
                if clean:
                    if self.q.qsize() > 3:
                        time.sleep(0.05)
                    self.q.put(clean)
                
                sentence_buffer = ""
        
        if sentence_buffer.strip():
            self.q.put(sentence_buffer.strip())

    def tts_worker(self):
        while True:
            text = self.q.get()

            if text is None:
                break

            generator = self.voice(text=text, voice='hf_alpha', speed=1.3)

            for _, _, audio in generator:
                self.speaker_stream.write(audio)
            
        self.q.task_done()
    
    def start(self):
        threading.Thread(target=self.tts_worker, daemon=True).start()
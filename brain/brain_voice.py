from queue import Queue
from dotenv import load_dotenv
load_dotenv()
import sounddevice as sd
import threading
from kokoro import KPipeline
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from conversation.controller import ConversationController

class BrainVoice:
    def __init__(self, text_q: Queue, controller: ConversationController):
        self.voice = KPipeline(lang_code='h')
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.system_prompt = """
You are a voice assisstant who talks in hindi, your task is make simple conversations based on user input. Your text output will be used in text-to-speech engine, so it is neccessary to produce correct hindi text with proper punctuations according to the conversation, for native english words that don't have a translation in hindi produce text, pronounced same as english when spoken
"""
        self.q = Queue()
        self.text_q = text_q
        self.controller = controller
        self.speaker_stream = sd.OutputStream(
            channels=1,
            samplerate=24000,
            dtype="float32"
        )
        self.speaker_stream.start()
    
    def llm_worker(self):
        while True:

            user = self.text_q.get()

            sentence_buffer = ""

            conversation = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user)
            ]

            print("Assistant:", end=" ")

            for chunk in self.model.stream(conversation):

                token = chunk.content or ""
                print(token, end="", flush=True)

                sentence_buffer += token

                if sentence_buffer.endswith(("।", ".", "?", "!")):

                    clean = sentence_buffer.strip()

                    if clean:

                        if self.q.qsize() > 3:
                            time.sleep(0.05)

                        self.q.put(clean)

                    sentence_buffer = ""

            if sentence_buffer.strip():
                self.q.put(sentence_buffer.strip())

            print()

            self.text_q.task_done()
        
    def tts_worker(self):
        while True:
            text = self.q.get()

            if text is None:
                break

            if not self.controller.ai_speaking:
                self.controller.start_ai()

            generator = self.voice(text=text, voice='hf_alpha', speed=1.3)

            for _, _, audio in generator:
                if self.controller.should_interrupt():
                    while not self.q.empty():
                        try:
                            self.q.get_nowait()
                        except:
                            break
                    break

                self.speaker_stream.write(audio)
            
            self.controller.stop_ai()
            self.q.task_done()
    
    def start(self):
        threading.Thread(target=self.llm_worker, daemon=True).start()
        threading.Thread(target=self.tts_worker, daemon=True).start()
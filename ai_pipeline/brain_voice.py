from queue import Queue
from dotenv import load_dotenv
load_dotenv()
import sounddevice as sd
from kokoro import KPipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import time
from ai_pipeline.controller import ConversationController

class BrainVoice:

    MAX_HISTORY = 20
    STRONG_BREAKS = ("।", ".", "?", "!", "|")
    SOFT_BREAKS = (" लेकिन ", " तो ", " फिर ", " क्योंकि ")
    MAX_CHARS = 60
    MIN_WORDS = 3
    MAX_LATENCY = 0.4

    def __init__(self, text_q: Queue, controller: ConversationController):
        self.voice = KPipeline(lang_code='h', repo_id='hexgrad/Kokoro-82M')
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.local_model = ChatOllama(model="gemma4:e2b", temperature=0)
        self.system_prompt = """
Your name is Anushka and you have a female persona.
You are a professional AI sales agent making cold calls to potential customers.
Your goal is to have a natural, human-like conversation and introduce a product or service in a polite and engaging way.
Strict rules:
- Speak only in Hindi.
- Use simple, conversational Hindi (natural spoken language).
- Keep sentences short and easy to speak.
- Use proper punctuation (। ? ! ,) to create natural pauses.
- Do not speak too fast or produce long paragraphs.
Conversation behavior:
- Start with a greeting and introduction.
- Ask if this is a good time to talk.
- If the user agrees, introduce the product clearly.
- Ask follow-up questions to understand interest.
- Be polite, not pushy.
- Handle hesitation naturally.
- Keep the tone friendly and confident.
Speech optimization:
- Avoid difficult words.
- Use pauses like "..." when needed.
- Make responses sound natural when spoken aloud.
- Convert English terms into Hindi pronunciation where needed.
Example tone:
"नमस्ते... क्या मैं आपसे दो मिनट बात कर सकता हूँ?"
Remember:
You are speaking over a call, not writing text.
"""
        self.q = Queue()
        self.history = []
        self.text_q = text_q
        self.controller = controller
        self.speaker_stream = sd.OutputStream(
            channels=1,
            samplerate=24000,
            dtype="float32"
        )
        self.speaker_stream.start()
    

    def should_flush(self, buffer: str, last_flush):
        now = time.time()

        if buffer.endswith(self.STRONG_BREAKS):
            return True
        
        if len(buffer) >= self.MAX_CHARS:
            return True
        
        if now - last_flush > self.MAX_LATENCY:
            return True
        
        return False
    
    def is_valid_chunk(self, chunk):
        return len(chunk.split(" ")) >= self.MIN_WORDS

    def llm_worker(self):
        while True:

            user = self.text_q.get()
            print("User: ", user)
            if not user or not user.strip():
                print("No input message...")
                continue

            sentence_buffer = ""

            print("Assistant:", end=" ")
            conversation = [
                SystemMessage(content=self.system_prompt)
            ]
            conversation.extend(self.history)
            conversation.append(HumanMessage(content=user))


            assistant_txt = ""
            last_flush = time.time()
            try:
                for chunk in self.local_model.stream(conversation):

                    token = chunk.content or ""
                    print(token, end="", flush=True)

                    sentence_buffer += token
                    assistant_txt += token
                    if self.should_flush(sentence_buffer, last_flush):

                        clean = sentence_buffer.strip()

                        if clean:

                            while self.q.qsize() > 3:
                                time.sleep(0.02)

                            self.q.put(clean)
                            last_flush = time.time()

                        sentence_buffer = ""
                
                final = sentence_buffer.strip()

                if final:
                    self.q.put(final)

                self.history.append(HumanMessage(content=user))
                self.history.append(AIMessage(content=assistant_txt.strip()))

                if len(self.history) > BrainVoice.MAX_HISTORY:
                    self.history = self.history[-BrainVoice.MAX_HISTORY:]

                self.text_q.task_done()
            except Exception as e:
                print("Error occured: ", str(e))
        
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
                    print("Interrupt was called...")
                    while not self.q.empty():
                        try:
                            self.q.get_nowait()
                        except:
                            break
                    break
                self.speaker_stream.write(audio)
            
            self.controller.stop_ai()
            self.q.task_done()
        
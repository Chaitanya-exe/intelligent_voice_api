from threading import Thread
from brain.brain_voice import BrainVoice
from ears.eardrum import EarDrum
from vad.vad_pipe import VadPipeline
from conversation.controller import ConversationController
from queue import Queue
import time

def main():
    print("initialising voice assisstant...")
    text_q, speech_q = Queue(), Queue()
    controller = ConversationController()
    vad = VadPipeline(
        speech_q=speech_q,
        controller=controller
    )
    ears = EarDrum(
        speech_q=speech_q,
        text_q=text_q
    )
    tts = BrainVoice(
        text_q=text_q,
        controller=controller
    )  
    Thread(target=vad.start).start()
    Thread(target=ears.worker).start()
    Thread(target=tts.llm_worker).start()
    Thread(target=tts.tts_worker).start()
    print("Voice assisstant started.")
    print("start speaking with your voice assisstant.")
    print("Listening...")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        
        print("\nUser exitted program")
    except Exception as e:
        import traceback
        traceback.print_exc()

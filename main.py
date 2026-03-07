from langchain_core.messages import HumanMessage, SystemMessage
from brain.brain_voice import BrainVoice
from ears.eardrum import STTPipeline

tts = BrainVoice()
stt = STTPipeline()
system_prompt = """
You are a voice assisstant who talks in hindi, your task is make simple conversations based on user input. Your text output will be used in text-to-speech engine, so it is neccessary to produce correct hindi text with proper punctuations according to the conversation, for native english words that don't have a translation in hindi produce text, pronounced same as english when spoken
"""

def main():
    tts.start()
    while True:

        input("Press enter to record your message: ")
        
        user = stt.get_transcription()
        
        conversation = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user)
        ]
        print("Assisstant: ")
        tts.tokenise_sentences(conversation)
        print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        tts.speaker_stream.stop()
        tts.speaker_stream.close()
        print("\nUser exitted program")
    except Exception as e:
        import traceback
        traceback.print_exc()
        tts.speaker_stream.stop()
        tts.speaker_stream.close()

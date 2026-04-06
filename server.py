from fastapi import FastAPI, WebSocket
from utils.audio_conversion import mu_to_pcm
import json
import base64

app = FastAPI()

@app.websocket('/twiml')
async def media_stream(ws: WebSocket):
    try:    
        await ws.accept()

        print("service connected")

        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)  

            event = data.get("event")

            if event == "start":
                print("stream started...")
            elif event == "media":
                audio = data['media']['payload']
                audio_bytes = base64.b64decode(audio)
                audio = mu_to_pcm(audio_bytes)
                
            elif event == "stop":
                print("stream ended.")
                break


    except Exception as e:
        print("Server error: ", str(e))
        import traceback
        traceback.print_exc()
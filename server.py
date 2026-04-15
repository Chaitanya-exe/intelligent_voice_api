from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, WebSocketException
from fastapi.responses import Response
from utils.audio_conversion import mu_to_pcm, pcm_to_mu
from twilio.rest import Client
from dotenv import load_dotenv
from ai_pipeline.session import VoiceSession
import json
import base64
import uvicorn
import os
import asyncio
import wave

load_dotenv()

DEV=os.getenv("DEV_ENV", None)
call_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
url = "https://6bd3-2409-40d6-1148-7a4f-7931-c83a-59a-e3d3.ngrok-free.app/twiml"

app = FastAPI()

async def sender(ws: WebSocket, session: VoiceSession, call_sid: str):
    try:
        while True:
            audio = await asyncio.to_thread(session.audio_q.get)
            mulaw = pcm_to_mu(audio)
            payload = base64.b64encode(mulaw).decode()

            await ws.send_text(json.dumps({
                "event":"media",
                "streamSid": call_sid,
                "media":{
                    "payload": payload
                }
            }))
    except asyncio.CancelledError:
        print("Sender task cancelled")
    except Exception as e:
        print("Error sending the audio: ", e)
        import traceback
        traceback.print_exc()

@app.post('/twiml')
async def twiml():
    xml = """
    <Response>
        <Connect>
            <Stream url="wss://6bd3-2409-40d6-1148-7a4f-7931-c83a-59a-e3d3.ngrok-free.app/twiml" />
        </Connect>
    </Response>
    """
    return Response(content=xml, media_type='application/xml')

@app.post("/init")
async def trigger_call():
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    try:
        call = call_client.calls.create(
            to="+919315390096",
            from_=from_number,
            url=url,
            method='POST',
            timeout=30
        )
        return { "msg": "call initiated", "sid": call.sid}
    
    except Exception as e:
        print("Error creating a call", e)
        raise HTTPException(status_code=500, detail="Some error occured")

@app.websocket('/twiml')
async def media_stream(ws: WebSocket):
    
    session = VoiceSession()
    session.start()
    
    await ws.accept()
    print("service connected")

    try:    
        sender_task = None
        while True:

            msg = await ws.receive()
            data = json.loads(msg["text"])  
            event = data.get("event")

            if event == "start":
                
                sid = data.get('streamSid')
                sender_task = asyncio.create_task(sender(session=session, ws=ws, call_sid=sid))
                session.set_sid(data.get('streamSid'))
                print(f"stream started... with stream SID: {data.get('streamSid')}")

            elif event == "media":
                
                audio = data['media']['payload']
                audio_bytes = base64.b64decode(audio)
                audio = mu_to_pcm(audio_bytes)
                session.input_stream.put(audio)

            elif event == "stop":
                print("stream ended.")
                break

    except WebSocketDisconnect:
        print("Web socket disconnected...")
    except WebSocketException as e:
        print("Error in the socket stream", e)
    except Exception as e:
        print("Server error: ", str(e))
        import traceback
        traceback.print_exc()
    finally:
        if sender_task:
            sender_task.cancel()

if __name__ == "__main__":
    uvicorn.run("server:app", port=8000, host="0.0.0.0", reload= True if DEV is not None else False)
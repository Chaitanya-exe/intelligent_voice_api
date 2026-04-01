# Real-Time Voice AI Assistant (Streaming + Interruptible)

A low-latency, real-time voice assistant built with a modular architecture using VAD + STT + LLM + TTS, supporting natural conversation flow, streaming responses, and user interruption (barge-in).

---

## Overview

This project implements a production-oriented voice interaction pipeline where:

- User speech is detected using Voice Activity Detection (VAD)
- Audio is transcribed into text using Speech-to-Text (STT)
- A Large Language Model (LLM) generates responses with conversational context
- Responses are streamed and converted into speech in real-time using Text-to-Speech (TTS)
- Users can interrupt the assistant mid-response

The system is designed to be:

- Non-blocking
- Streaming-first
- Modular and extensible
- Local-first (can run without cloud dependencies)

---

## Architecture

```

Microphone Input
↓
[VAD Pipeline]  → detects speech boundaries
↓
[STT Pipeline]  → converts speech → text
↓
[Conversation Controller]
↓
[LLM + Context Memory]
↓
[Sentence Chunking / Streaming]
↓
[TTS Pipeline]
↓
Speaker Output

```

---

## Core Components

### BrainVoice (LLM + TTS Engine)

Handles:

- LLM interaction (streaming)
- Sentence chunking
- TTS generation
- Context memory

Responsibilities:

- Streams tokens from LLM
- Buffers tokens into sentences
- Pushes sentences into TTS queue
- Maintains conversation history

Key Features:

- Streaming responses using model.stream()
- Sentence-aware TTS buffering
- Context-aware conversation
- Queue-based audio generation

---

### VadPipeline (Voice Activity Detection)

Uses Silero VAD to:

- Detect speech start and end
- Segment continuous audio stream

Responsibilities:

- Continuously listen to microphone
- Maintain rolling audio buffer
- Emit speech segments to STT queue

Key Features:

- Real-time detection (blocksize=512)
- Handles speech start/end events
- Prevents empty/ghost segments
- Supports interruption detection

---

### STTPipeline (Speech-to-Text)

Uses faster-whisper to:

- Convert audio segments into text

Responsibilities:

- Consume audio segments from queue
- Transcribe speech
- Push text into LLM queue

Key Features:

- Optimized inference (int8)
- Low-latency transcription
- Language auto-detection

---

### ConversationController

Central coordination unit.

Responsibilities:

- Track system state:
  - user_speaking
  - ai_speaking
  - interrupt_flag
- Enable or disable behaviors across modules

Key Features:

- AI interruption (barge-in)
- Prevents feedback loop (AI hearing itself)
- Synchronizes pipelines

---

## Orchestration Flow

Step-by-step execution:

1. Microphone stream starts
2. VadPipeline detects speech start
3. Audio buffered until speech ends
4. Audio segment sent to STTPipeline
5. Transcription sent to BrainVoice
6. LLM generates response (streaming)
7. Tokens are chunked into sentences and queued
8. TTS plays audio in real-time
9. If user speaks during TTS, interruption is triggered

---

## Key Features

### Real-Time Streaming

- LLM responses streamed token-by-token
- TTS starts before full response is generated

### Interruptible AI (Barge-In)

- User can interrupt AI mid-speech
- TTS stops immediately
- New input takes priority

### Context-Aware Conversations

- Maintains short-term history
- Enables multi-turn conversations

### Low Latency Pipeline

- Non-blocking queues between components
- Parallel processing using threads

### Sentence-Based Audio Generation

- Prevents broken speech output
- Improves naturalness

### Modular Design

- Each component is independent
- Easy to replace STT, TTS, or LLM

---

## Challenges Solved

### Empty Audio Segments

- Fixed VAD buffer misalignment
- Handled ghost end events

### AI Talking to Itself

- Caused by mic capturing speaker output
- Solved using controller-based gating

### Missing LLM Output

- Fixed incorrect streaming aggregation
- Ensured token accumulation

### Broken Sentence Streaming

- Improved punctuation-based segmentation
- Added fallback chunking strategy

### Thread Coordination Issues

- Introduced shared queues
- Centralized state using controller

---

## Tech Stack

- VAD: Silero VAD
- STT: faster-whisper
- LLM: Gemini / Ollama (Qwen)
- TTS: Kokoro (hexgrad/Kokoro-82M)
- Audio I/O: sounddevice
- Orchestration: Python threading + queues

---

## Project Structure

```

voice_api/
│
├── vad/
│   └── vad_pipe.py
│
├── stt/
│   └── stt_pipe.py
│
├── brain/
│   └── brain_voice.py
│
├── controller/
│   └── controller.py
│
├── main.py

```

---

## Current Progress

Completed:

- VAD-based speech segmentation
- Real-time STT pipeline
- Streaming LLM responses
- Sentence-based TTS streaming
- Context-aware conversations
- Interruptible AI (barge-in)
- Stable orchestration

In Progress:

- Latency optimization
- Partial STT (streaming transcription)
- Improved Hindi phonetic output

Planned:

- Noise-robust VAD tuning
- Partial streaming STT for faster response
- Long-term memory (database-backed)
- Expressive TTS (prosody control)
- Telephony integration
- Web UI / API layer

---

## How to Run

```

python main.py

```

Speak into the microphone and the assistant responds in real-time.

---

## Learning Outcomes

- Real-time audio processing systems
- Concurrency and thread coordination
- VAD + STT + TTS integration
- Streaming LLM pipelines
- Designing low-latency AI systems

---

## Goal

To build a production-grade, real-time conversational AI system that:

- Feels natural like human conversation
- Works locally with minimal resources
- Scales into a deployable product

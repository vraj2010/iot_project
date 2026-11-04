# Nexa: Edge AI & IoT Smart Speaker Voice Assistant

[![Platform](https://img.shields.io/badge/Platform-ESP32%20%7C%20Windows-blue.svg)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://python.org)
[![Database](https://img.shields.io/badge/Database-PostgreSQL%2017%20%28pgvector%29-336791.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Voice](https://img.shields.io/badge/Speech-faster--whisper%20%7C%20Edge--TTS-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![Interface](https://img.shields.io/badge/Dashboard-Cyber%20Orb%20UI-00f0ff.svg)](http://127.0.0.1:8777)

**Nexa** is an intelligent, low-latency IoT voice assistant that bridges edge microcontrollers with a high-performance local AI runtime. Combining dedicated I2S audio hardware on an ESP32 with local speech recognition, neural cognition, contextual long-term memory, and an interactive 3D Sci-Fi dashboard, Nexa provides an Iron Man JARVIS–style conversational experience.

---

## System Architecture

Nexa uses a split-computing IoT architecture: edge hardware handles digital audio playback and physical triggers, while the host workstation executes heavy AI workloads in real time.

```
                              ┌────────────────────────────────────────────────────────┐
                              │                    NEXA HOST RUNTIME                   │
                              │                                                        │
┌───────────────────────┐     │  ┌──────────────────┐      ┌────────────────────────┐  │
│      ESP32 NODE       │     │  │  VAD & Wake Spot │      │      Neural Core       │  │
│  (MAX98357A I2S DAC)  │◀──┐ │  │  (Silero + Tiny) │      │  (Intent & Reasoning)  │  │
└───────────────────────┘   │ │  └────────┬─────────┘      └───────────┬────────────┘  │
    ▲ (A2DP / Wi-Fi)        │ │           │ "Hey Nexa"                 │               │
    │                       │ │           ▼                            ▼               │
┌───┴───────────────────┐   │ │  ┌──────────────────┐      ┌────────────────────────┐  │
│   Host Mic / BOOT     ├───┼─┼─▶│  faster-whisper  ├─────▶│  PostgreSQL + pgvector │  │
│   Hardware Trigger    │   │ │  │   (Int8 / Float) │      │  (Long-Term Episodic)  │  │
└───────────────────────┘   │ │  └──────────────────┘      └───────────┬────────────┘  │
                            │ │                                        │               │
                            │ │  ┌──────────────────┐                  │               │
                            │ └──┤ Edge-TTS Stream  │◀─────────────────┘               │
                            │    │ (Neural Voice)   │                                  │
                            │    └──────────────────┘                                  │
                            └──────────────────────────────────────────────────────────┘
                                          │
                                          ▼ Telemetry & Live State
                            ┌───────────────────────────┐
                            │   Sci-Fi Orb Dashboard    │
                            │   (WebGL + Live Console)  │
                            └───────────────────────────┘
```

---

## Key Capabilities

- **Continuous Voice Wake ("Hey Nexa")**: Always-on acoustic speech spotter using Silero VAD and real-time speech spotting—no keyboard input or button pressing required.
- **Dual Hardware Configurations**:
  1. **ESP32 DevKit + MAX98357A I2S Amplifier (`nexa_speaker/`)**: Turns any standard ESP32 into a dedicated high-fidelity Bluetooth A2DP IoT smart speaker with hardware BOOT push-to-talk.
  2. **ESP32-S3 Wi-Fi Direct Node (`firmware/`)**: Direct Wi-Fi streaming node using ESP-IDF 5.5 and UDP discovery.
- **Hierarchical Intent Engine**:
  - **Zero-Latency Local Dispatch (<40 ms)**: Volume adjustment, media playback (play/pause/next/previous), time, and system metrics run locally without cloud delays.
  - **Deep Reasoning Engine**: Multi-turn dialogue, dynamic function calling (live weather, dual-engine web search, Cortex PDF retrieval).
- **Episodic Long-Term Memory**: Backed by PostgreSQL 17 with `pgvector` (`nexa-pg`). Automatically distills user facts, preferences, and conversations in the background (`AUTO_MEMORY`).
- **Interactive Cyberpunk Visualizer (`orb.py`)**: A 3D audio-reactive WebGL visualizer with real-time hardware status synchronization, latency waterfall breakdown, and integrated console.

---

## End-to-End Workflow

The following sequence diagram outlines how audio frames, triggers, neural processing, and telemetry travel through the system:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Mic as Microphone / BOOT
    participant Ears as Nexa Ears (VAD/Wake)
    participant Core as Nexa Server (server.py)
    participant DB as PostgreSQL (nexa-pg)
    participant TTS as Neural Voice (Edge-TTS)
    participant Spk as ESP32 (MAX98357A)
    participant UI as Cyber Orb UI (orb.py)

    Note over User,Ears: Phase 1: Wake Detection
    User->>Mic: "Hey Nexa" or BOOT Button
    Mic->>Ears: Audio Frame Stream (16 kHz PCM)
    Ears->>Core: Trigger Wake Event (Wake ID generated)
    Core->>UI: Status -> "LISTENING" (Orb Amber)
    Core->>Spk: Audio Chime (Acknowledge)

    Note over User,Core: Phase 2: Speech Capture & STT
    User->>Mic: "What is the weather today?"
    Mic->>Ears: Spoken Utterance
    Ears->>Core: Endpoint Reached (Silence detected)
    Core->>UI: Status -> "THINKING" (Orb Purple)
    Core->>Core: faster-whisper transcribes audio

    Note over Core,DB: Phase 3: Cognitive Processing & Memory
    Core->>DB: Query User Profile & Vector Episodic Memory
    DB-->>Core: Relevant Facts & Memory Context
    Core->>Core: Evaluate Tools (Local Intent / Weather / Search)
    Core->>DB: Persist Turn to `turns` Table

    Note over Core,Spk: Phase 4: Neural Voice Streaming & Telemetry
    Core->>TTS: Stream Reply Text
    TTS-->>Core: Synthesized Audio Chunks (MP3/PCM)
    Core->>UI: Status -> "SPEAKING" (Orb Green)
    Core->>Spk: Stream Audio to MAX98357A DAC
    Spk-->>User: High-fidelity voice playback
    Core->>UI: Telemetry (STT ms, Thinking ms, TTS ms)
    Core->>UI: Status -> "IDLE" (Orb Cyan)
```

---

## Hardware Assembly & Pinout

### ESP32 to MAX98357A Wiring

The standard build utilizes an ESP32 WROOM-32 connected to a MAX98357A I2S Class-D amplifier driving a 4Ω or 8Ω speaker:

| MAX98357A Pin | ESP32 GPIO | Description |
|:---|:---|:---|
| **BCLK** (Bit Clock) | **GPIO 26** | I2S Serial Clock |
| **LRC** (Left/Right Clock) | **GPIO 25** | Word Select / Frame Sync |
| **DIN** (Data In) | **GPIO 22** | Serial Audio Data |
| **SD** (Shutdown) | **GPIO 27** | Amp Enable (Pulled HIGH; floating = shutdown) |
| **GAIN** | **GND** | 12 dB Gain Configuration |
| **VIN** | **5V / 3V3** | System Power Supply |
| **GND** | **GND** | Common System Ground |
| **BOOT Button** | **GPIO 0** | Hardware Push-to-Talk (Optional manual trigger) |

> [!NOTE]
> Connect the speaker terminals directly to `SPK+` and `SPK-`. Never ground either speaker lead.

---

## Repository Structure

```
iot_project/
├── nexa_speaker/          # Arduino firmware for ESP32 Bluetooth / I2S audio sink
│   └── nexa_speaker.ino
├── server/                # Python Host Runtime & Neural Core
│   ├── server.py          # Primary HTTP/UDP server & orchestration
│   ├── speaker.py         # Bluetooth audio pipeline, mixer & hardware PTT bridge
│   ├── orb.py             # Cyberpunk Orb Web Server & real-time telemetry
│   ├── orb.html           # 3D WebGL Canvas visualizer & interactive console
│   ├── brain.py           # Cognitive engine & function execution
│   ├── gemini.py          # Neural Core communication & streaming
│   ├── stt.py             # faster-whisper local speech-to-text
│   ├── tts.py             # Edge-TTS / Piper neural speech synthesis
│   ├── memory.py          # PostgreSQL conversational memory interface
│   ├── intents.py         # Ultra-low latency local intent router
│   └── ears/              # Acoustic wake detection & VAD pipeline
│       ├── listener.py
│       ├── vad.py
│       └── wake.py
├── firmware/              # ESP32-S3 ESP-IDF native firmware (Wi-Fi streaming build)
├── db/                    # Dockerized PostgreSQL 17 + pgvector + Adminer
│   ├── docker-compose.yml
│   └── .env.example
├── cortex-web/            # Cortex document RAG web ingestion portal
└── docs/                  # Architecture specifications & latency benchmarks
```

---

## Quickstart Guide

### Prerequisites
- Python 3.11+ (or Python 3.14 on Windows)
- Docker Desktop (for persistent PostgreSQL memory)
- Arduino IDE with ESP32 Board Support (`v2.x` or `v3.x`)
- Git

---

### Step 1: Start PostgreSQL Memory (`db`)

```powershell
cd "c:\Data\B.Tech\Semester 7\Internet of Things\Project\iot_project\db"
copy .env.example .env
docker compose up -d
```
Verify the container is active:
- PostgreSQL is reachable at `127.0.0.1:5433` (mapped from container `5432`).
- Adminer Database Web GUI is accessible at `http://127.0.0.1:8081`.

---

### Step 2: Configure & Setup Server Environment

```powershell
cd "c:\Data\B.Tech\Semester 7\Internet of Things\Project\iot_project\server"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `server/.env` with your preferred credentials:
```env
GEMINI_API_KEY="your_api_key_here"
DATABASE_URL="postgresql://nexa:YOUR_PASSWORD@127.0.0.1:5433/nexa"
DEVICE_TOKEN="nexa_secure_token_123"
WAKE_ENABLED=1
WAKE_MODEL="nexa"
STREAM_REPLIES=1
```

---

### Step 3: Flash the ESP32 Speaker Firmware

1. Open `nexa_speaker/nexa_speaker.ino` in Arduino IDE.
2. Install the required libraries via Library Manager or zip:
   - `ESP32-A2DP` (by Phil Schatzmann)
   - `arduino-audio-tools` (by Phil Schatzmann)
3. Select Board: **ESP32 Dev Module**.
4. Connect the ESP32 via USB and click **Upload**.
5. Once flashed, open Windows **Bluetooth & Devices**, pair with **"Nexa"**, and set it as your default audio output device.

---

### Step 4: Run the Nexa Ecosystem

Launch each module in separate terminal windows:

#### Terminal 1: Core Server (The Brain)
```powershell
cd "c:\Data\B.Tech\Semester 7\Internet of Things\Project\iot_project\server"
.venv\Scripts\activate
python server.py
```
*Initializes faster-whisper, loads vector memory tables, warms the Neural Core, and starts listening on port 8765.*

#### Terminal 2: Bluetooth Speaker Client (Audio Bridge)
```powershell
cd "c:\Data\B.Tech\Semester 7\Internet of Things\Project\iot_project\server"
.venv\Scripts\activate
python speaker.py
```
*Bridges the host audio mixer and push-to-talk triggers directly to the ESP32 over Bluetooth.*

#### Terminal 3: Cyberpunk Orb Dashboard (Visualizer)
```powershell
cd "c:\Data\B.Tech\Semester 7\Internet of Things\Project\iot_project\server"
.venv\Scripts\activate
python orb.py
```
*Opens `http://127.0.0.1:8777/` with the interactive 3D orb visualizer, real-time telemetry, and console.*

---

## Operating Modes

### 1. Hands-Free Voice ("Hey Nexa")
Simply speak naturally near your microphone:
- *"Hey Nexa, what's on my schedule today?"*
- *"Nexa, remember that my project submission deadline is Friday."*
- *"What did I tell you about my project deadline?"*

### 2. Instant Local Commands (<40 ms response)
Commands answered instantly by the local router without cloud latency:
- *"Pause"* / *"Resume"* / *"Stop"*
- *"Volume 50"* / *"Louder"* / *"Softer"*
- *"What time is it?"* / *"What is today's date?"*

### 3. Hardware Push-to-Talk (BOOT Button)
Press the onboard **BOOT** button on the ESP32 DevKit to activate listening immediately without speaking the wake word.

---

## REST API & Telemetry Endpoints

The Nexa host runtime exposes a clean HTTP API on port `8765`:

| Endpoint | Method | Description |
|:---|:---|:---|
| `/health` | `GET` | Returns core health status, memory stats, and live ear state (`idle`, `listening`, `thinking`). |
| `/listen` | `GET` | Long-poll listener used by speaker clients to wait for wake events. |
| `/listen?wake=<id>` | `GET` | Retrieves recorded PCM audio for an active wake session. |
| `/listen/wake` | `POST` | Hardware trigger endpoint to signal manual push-to-talk. |
| `/say` | `POST` | Dispatches typed text directly into the neural pipeline (`{"text": "..."}`). |
| `/ask` | `POST` | Submits raw 16 kHz PCM audio data to run full STT-Reasoning-TTS pipeline. |
| `/music/<id>` | `GET` | Streams MP3 audio from the local indexed music library. |
| `/api/turns` | `GET` | Real-time endpoint on port `8777` providing conversation history and latency telemetry. |

---

## Diagnostic & Telemetry Log

Every turn records a comprehensive latency breakdown to `server/latency.log`:

```
kind     said                     stt_ms  gemini_ms  tool_ms  tts_ms  first_audio_ms
spoken   what time is it          210     0          1        38      249
spoken   who discovered penicillin 324    380        0        142     846
local    pause                    0       0          2        0       2
```

- **`stt_ms`**: Time spent in `faster-whisper` acoustic decoding.
- **`gemini_ms`**: Cognitive processing and tool invocation time.
- **`tts_ms`**: Time to synthesize first streaming audio chunk via Edge-TTS.
- **`first_audio_ms`**: Total latency from utterance end to first sound from speaker.

---

## Troubleshooting

- **Database Connection Error (`port 5433`)**: Ensure Docker is running and verify the port with `docker ps`. The container binds to host port `5433` to prevent collision with local PostgreSQL installations.
- **Bluetooth Audio Choppy / Silent**: Ensure your Windows audio output is set to **"Nexa"** and that the MAX98357A `SD` pin is pulled HIGH to `GPIO 27`.
- **Wake Word Sensitivity**: Adjust `WAKE_THRESHOLD` in `server/.env` (default is `0.5`). Lower values increase sensitivity, higher values decrease false activations.
- **Firewall Warnings**: If the ESP32 cannot find the server, add inbound firewall rules for TCP `8765` and UDP `5006` as detailed in [`server/README.md`](server/README.md).

---

## License

This project is open-source under the MIT License. Developed for academic demonstration, edge IoT research, and personal assistant automation.

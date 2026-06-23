import os, httpx

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"

# Cartesia voices:
# a0e99841-438c-4a64-b679-ae501e7d6091  — Calm female (recommended)
# 694f9389-aac1-45b6-b726-9d9369183238  — Professional male
VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"


async def synthesize_speech(text: str) -> bytes:
    """
    Convert text to speech via Cartesia and return raw WAV bytes.
    Model: sonic-2 (updated from sonic-english)
    API Version: 2024-11-13
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            CARTESIA_URL,
            headers={
                "X-API-Key": CARTESIA_API_KEY,
                "Cartesia-Version": "2024-11-13",
                "Content-Type": "application/json",
            },
            json={
                "model_id": "sonic-2",
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": VOICE_ID,
                },
                "output_format": {
                    "container": "wav",
                    "encoding": "pcm_f32le",
                    "sample_rate": 44100,
                },
            },
        )

        if not response.is_success:
            # Log the actual error body from Cartesia for easier debugging
            print(f"❌ Cartesia TTS error {response.status_code}: {response.text}")
            response.raise_for_status()

        return response.content

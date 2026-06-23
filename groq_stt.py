import os
from deepgram import DeepgramClient, PrerecordedOptions

client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))


def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    """
    Transcribe audio bytes using Deepgram Nova-2.
    Accepts webm/opus from MediaRecorder (browser default).
    """
    payload = {
        "buffer": audio_bytes,
        "mimetype": "audio/webm",
    }

    options = PrerecordedOptions(
        model="nova-2",
        language=language,
        smart_format=True,   # punctuation + casing
        utterances=False,
    )

    response = client.listen.prerecorded.v("1").transcribe_file(payload, options)

    try:
        transcript = response.results.channels[0].alternatives[0].transcript
        return transcript.strip()
    except (AttributeError, IndexError):
        return ""

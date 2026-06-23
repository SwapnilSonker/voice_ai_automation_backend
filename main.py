import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from database import create_db
from groq_stt import transcribe_audio
from groq_llm import chat_with_tools
from tools import execute_tool
from tts import synthesize_speech


# ── app setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db()
    print("✅ Database ready")
    yield

app = FastAPI(title="Mykare Voice AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket conversation ────────────────────────────────────────────────────

@app.websocket("/ws/conversation")
async def conversation_ws(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")

    # Get model query parameter (fallback to default)
    model_name = websocket.query_params.get("model", "llama-3.1-8b-instant")
    print(f"🤖 Connected with model: {model_name}")

    # Send init configuration to client
    await websocket.send_json({"type": "init", "model": model_name})

    messages: list[dict] = []   # full conversation history

    try:
        # Send greeting audio on connect
        greeting = "Hello! I'm Mia, your healthcare assistant. May I have your name and phone number to get started?"
        greeting_audio = await synthesize_speech(greeting)
        await websocket.send_json({"type": "ai_text", "text": greeting})
        await websocket.send_bytes(greeting_audio)
        messages.append({"role": "assistant", "content": greeting})

        while True:
            # ── receive audio from browser ────────────────────────────────────
            raw = await websocket.receive()

            if raw.get("type") == "websocket.disconnect":
                break

            audio_bytes = raw.get("bytes")
            if not audio_bytes:
                continue

            # ── STT ───────────────────────────────────────────────────────────
            await websocket.send_json({"type": "status", "text": "Transcribing..."})
            try:
                transcript = transcribe_audio(audio_bytes)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"STT failed: {e}"})
                continue

            if not transcript:
                continue

            await websocket.send_json({"type": "transcript", "text": transcript})
            messages.append({"role": "user", "content": transcript})

            # ── LLM + tool calling loop ───────────────────────────────────────
            await websocket.send_json({"type": "status", "text": "Thinking..."})
            conversation_ended = False

            while True:
                try:
                    result = chat_with_tools(messages, model=model_name)
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"LLM error: {e}"})
                    break

                # ── tool calls ────────────────────────────────────────────────
                if result["type"] == "tool_calls":
                    # Append assistant message once to history
                    messages.append(result["raw_assistant_msg"])

                    for tc in result["tool_calls"]:
                        tool_name = tc["name"]
                        tool_args = tc["args"]
                        tool_call_id = tc["id"]

                        # Notify UI: tool is being called
                        await websocket.send_json({
                            "type": "tool_calling",
                            "tool": tool_name,
                            "args": tool_args
                        })

                        # Execute
                        tool_result = execute_tool(tool_name, tool_args)

                        # Notify UI: tool completed
                        await websocket.send_json({
                            "type": "tool_done",
                            "tool": tool_name,
                            "result": tool_result
                        })

                        # Append tool response for this specific tool call ID
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(tool_result)
                        })

                        # End conversation tool → send summary and close
                        if tool_name == "end_conversation":
                            summary_text = tool_result.get("summary")
                            # If summary is missing or too short, generate a detailed fallback
                            if not summary_text or len(summary_text.strip()) < 10:
                                user_msgs = [m["content"] for m in messages if m["role"] == "user" and m.get("content")]
                                last_user_requests = user_msgs[-3:] if user_msgs else []
                                requests_str = ", ".join(f"'{req}'" for req in last_user_requests)
                                booked_ids = tool_result.get("appointments_booked", [])
                                if booked_ids:
                                    tool_result["summary"] = f"Patient discussed: {requests_str}. Successfully booked appointment ID(s): {', '.join(map(str, booked_ids))}."
                                else:
                                    tool_result["summary"] = f"Patient discussed: {requests_str}. The conversation was completed without new bookings."

                            await websocket.send_json({
                                "type": "summary",
                                "data": {
                                    **tool_result,
                                    "full_history": [
                                        m for m in messages
                                        if m["role"] in ("user", "assistant") and m.get("content")
                                    ]
                                }
                            })
                            conversation_ended = True

                    if conversation_ended:
                        break

                    # Continue loop — LLM may call another tool or respond with text
                    continue

                # ── text response ─────────────────────────────────────────────
                elif result["type"] == "text":
                    text_response = result["content"]
                    messages.append({"role": "assistant", "content": text_response})

                    await websocket.send_json({"type": "ai_text", "text": text_response})

                    # TTS
                    await websocket.send_json({"type": "status", "text": "Speaking..."})
                    try:
                        audio_bytes_out = await synthesize_speech(text_response)
                        await websocket.send_bytes(audio_bytes_out)
                    except Exception as e:
                        # Truncate error so WebSocket send never fails on long httpx traces
                        err_short = str(e)[:200]
                        print(f"❌ TTS error: {err_short}")
                        # Send tts_error so frontend resets to ready (not stuck at processing)
                        await websocket.send_json({
                            "type": "tts_error",
                            "message": f"Voice unavailable: {err_short}"
                        })

                    break

            if conversation_ended:
                break

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Mykare Voice AI Backend"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

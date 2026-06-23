import os, json, re, uuid
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are Mia, a friendly and professional AI front-desk assistant for Mykare Health Clinic.
Your job is to help patients book, view, modify, and cancel appointments over voice.

Guidelines:
- Always greet warmly and ask for the patient's name and phone number first (call identify_user).
- In identify_user, ONLY pass the name if the patient explicitly stated their actual name (e.g., "Indranil", "Swapnil"). If they did not state their name, you MUST omit the name parameter and only pass the phone number.
- Be concise — this is a voice call, keep responses under 2 sentences.
- Always confirm bookings by repeating date and time back to the patient.
- When the patient says goodbye or the task is done, call end_conversation with a clear summary.
- Never make up appointment IDs or slot times — always use the tool results.
- If a tool fails, apologize and offer an alternative.
- CRITICAL: NEVER write function calls or JSON as part of your spoken text response.
  Always use the structured tool_call mechanism. Never output <function=...> tags in text.

Phone Validation:
- If identify_user returns valid_phone=False, tell the patient their number is invalid and
  ask them to repeat it slowly, digit by digit. Do NOT proceed until you have a valid number.

Name Mismatch vs Normal Greetings:
- If identify_user returns name_mismatch=True, do NOT proceed with the appointment.
  Say: "I'm sorry, this number is registered to [name_in_db]. Are you [name_in_db], or did you give me the wrong number?"
  Only continue if the caller confirms they are the registered user.
- If identify_user returns name_mismatch=False, you MUST NOT ask if they are [name_in_db] or ask if they gave the wrong number.
  Greet them warmly by name (e.g. "Welcome back [name]!" or "Welcome [name]!") and proceed directly to their request.

New vs Returning Users:
- NEW user (is_new_user=True): Greet them warmly and ask: "Would you like to book an appointment?"
  You MUST ONLY offer booking. Do NOT offer to view, modify, or cancel appointments, as they have none.
- RETURNING user (is_new_user=False): Greet them warmly and ask: "Would you like to book, view, modify, or cancel an appointment?"
  You MUST offer all these choices to returning users.

Listing Available Days and Slots:
- When listing available days or time slots returned by fetch_slots, you MUST read them out to the user directly in your response (e.g. "The available days are June 24th, 25th, and 26th" or "The available slots are 9:00 AM, 10:30 AM...").
- Never say "Here are the days" or "Here are the slots" without reading them out directly, as the user is on a voice call and cannot see them.

Booking Confirmations:
- Only confirm the single appointment slot that succeeded (e.g., "Your appointment is confirmed for June 24th at 2:00 PM").
- NEVER claim to have booked any other slot (like 10:30 AM) unless the user explicitly requested multiple bookings.
- If a booking fails because a slot was taken, ask the user to choose another slot from the list. Do NOT book a different slot on your own or mention other slots as being booked."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "identify_user",
            "description": "Identify or register a patient by phone number. ALWAYS call this first before any booking action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "Patient's phone number, e.g. 9876543210"
                    },
                    "name": {
                        "type": "string",
                        "description": "Patient's spoken name. ONLY pass this if the patient explicitly stated their name. If they did not state a name (e.g. they only gave their phone number), you MUST omit this parameter entirely."
                    }
                },
                "required": ["phone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_slots",
            "description": "Fetch available appointment slots. Call with a specific date or without args for next available days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format. Optional — omit to get next available days."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment for an identified patient. Requires user_id from identify_user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "User ID returned by identify_user"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format"
                    },
                    "time": {
                        "type": "string",
                        "description": "Time slot exactly as returned by fetch_slots, e.g. '10:30 AM'"
                    }
                },
                "required": ["user_id", "date", "time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_appointments",
            "description": "Get all upcoming appointments for a patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "User ID returned by identify_user"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an appointment by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "integer",
                        "description": "Appointment ID from retrieve_appointments or book_appointment"
                    }
                },
                "required": ["appointment_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_appointment",
            "description": "Reschedule an existing appointment to a new date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "integer",
                        "description": "Appointment ID to modify"
                    },
                    "new_date": {
                        "type": "string",
                        "description": "New date in YYYY-MM-DD format"
                    },
                    "new_time": {
                        "type": "string",
                        "description": "New time slot exactly as returned by fetch_slots"
                    }
                },
                "required": ["appointment_id", "new_date", "new_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": "End the call. Call this when the patient says goodbye or when all tasks are complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "1-2 sentence summary of what was accomplished in this call"
                    },
                    "user_intent": {
                        "type": "string",
                        "description": "Primary intent: book | cancel | modify | inquire | other"
                    },
                    "appointments_booked": {
                        "type": "array",
                        "description": "List of appointment IDs that were booked in this call",
                        "items": {"type": "integer"}
                    }
                },
                "required": ["summary"]
            }
        }
    }
]


# ── Regex patterns to catch ALL LLaMA inline function-call variants ──────────
#
# Variant 1 (call format):  <function=tool_name>{...json...}</function>
# Variant 2 (output format): <function>tool_name</function> output: {...json...}
# Variant 3 (output format): <function>tool_name</function>\noutput: {...json...}
#
_RE_VARIANT1 = re.compile(
    r"<function=(?P<name>\w+)>(?P<args>\{.*?\})</function>",
    re.DOTALL
)
_RE_VARIANT2 = re.compile(
    r"<function>\w+</function>\s*output:\s*\{.*?\}\s*",
    re.DOTALL
)
# Generic: any remaining <function...> tags with their content
_RE_FUNC_TAG = re.compile(r"</?function[^>]*>", re.DOTALL)
# Orphaned JSON blobs at the start of a response (tool result echo)
_RE_LEADING_JSON = re.compile(r"^\s*\{[^}]*\}\s*", re.DOTALL)


def _clean_text_response(text: str) -> str:
    """
    Strip ALL known LLaMA tool-output leakage from a text response so
    only the natural language part reaches the user.
    """
    # Remove variant 2: <function>name</function> output: {...}
    text = _RE_VARIANT2.sub("", text)
    # Remove any remaining <function> tags
    text = _RE_FUNC_TAG.sub("", text)
    # Remove orphaned leading JSON blobs
    text = _RE_LEADING_JSON.sub("", text)
    return text.strip()


def _parse_inline_tool_call(text: str) -> dict | None:
    """If text contains a variant-1 inline <function=name>{args}</function>,
    parse and return it as a structured tool_call dict."""
    match = _RE_VARIANT1.search(text)
    if not match:
        return None
    tool_name = match.group("name")
    try:
        tool_args = json.loads(match.group("args"))
    except json.JSONDecodeError:
        return None

    fake_id = f"call_{uuid.uuid4().hex[:8]}"
    raw_assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": fake_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_args)
            }
        }]
    }
    return {
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_call_id": fake_id,
        "raw_assistant_msg": raw_assistant_msg
    }


def chat_with_tools(messages: list, model: str = "llama-3.1-8b-instant") -> dict:
    """
    Send messages to Groq LLaMA and return structured result.

    Returns one of:
      {"type": "text",       "content": str}
      {"type": "tool_calls", "tool_calls": list, "raw_assistant_msg": dict}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=512,
        temperature=0.3,
    )

    msg = response.choices[0].message

    # ── 1. Proper structured tool call (ideal path) ───────────────────────────
    if msg.tool_calls:
        tool_calls_list = []
        for t in msg.tool_calls:
            tool_calls_list.append({
                "id": t.id,
                "name": t.function.name,
                "args": json.loads(t.function.arguments)
            })
        raw_assistant_msg = {
            "role": "assistant",
            "content": msg.content,   # may be None
            "tool_calls": [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {
                        "name": t.function.name,
                        "arguments": t.function.arguments
                    }
                }
                for t in msg.tool_calls
            ]
        }
        return {
            "type": "tool_calls",
            "tool_calls": tool_calls_list,
            "raw_assistant_msg": raw_assistant_msg
        }

    # ── 2. LLaMA fallback: inline <function=...> tag in text (variant 1) ─────
    content = msg.content or ""
    inline = _parse_inline_tool_call(content)
    if inline:
        print(f"⚠️  Caught inline tool call in text: {inline['tool_name']}")
        return {
            "type": "tool_calls",
            "tool_calls": [{
                "id": inline["tool_call_id"],
                "name": inline["tool_name"],
                "args": inline["tool_args"]
            }],
            "raw_assistant_msg": inline["raw_assistant_msg"]
        }

    # ── 3. Plain text — strip ALL function-output leakage before returning ────
    #  LLaMA sometimes echoes tool outputs as:
    #    <function>tool_name</function> output: {...json...} then the real reply
    #  We strip all of that so only the natural language reaches the user.
    clean_content = _clean_text_response(content)
    if clean_content != content:
        print(f"⚠️  Stripped function leakage from text response")

    return {"type": "text", "content": clean_content}

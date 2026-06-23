import re
from database import User, Appointment, engine
from sqlmodel import Session, select
from datetime import datetime, timedelta


# ── Phone number sanitiser ───────────────────────────────────────────────────
def _clean_phone(raw: str) -> str:
    """Strip spaces, dashes, brackets, dots from a phone string.
    Optionally strips a leading country code of 1 (US/India +91 not stripped).
    Returns only the digit string.
    """
    digits = re.sub(r'\D', '', raw)
    # Strip leading 1 (US country code) only if result would be 11 digits
    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]
    return digits


# ── hardcoded available slots ────────────────────────────────────────────────
def _generate_slots():
    slots = {}
    base = datetime.now()
    for i in range(1, 8):
        day = base + timedelta(days=i)
        if day.weekday() < 5:  # weekdays only
            key = day.strftime("%Y-%m-%d")
            slots[key] = ["9:00 AM", "10:30 AM", "12:00 PM", "2:00 PM", "3:30 PM", "5:00 PM"]
    return slots

AVAILABLE_SLOTS = _generate_slots()


# ── Name helper ──────────────────────────────────────────────────────────────
def _clean_name(name: str) -> str | None:
    if not name:
        return None
    cleaned = name.strip()
    # Remove common prefixes case-insensitively
    lower_cleaned = cleaned.lower()
    for prefix in ["my name is ", "i am ", "myself ", "this is ", "name is "]:
        if lower_cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
            
    lower_cleaned = cleaned.lower()
    
    # Ignore if name contains signs of "name not known/provided" or sentence/phrases
    negatives = ["not", "no", "don't", "dont", "unknown", "unspecified", "aware", "filler", "n/a", "missing"]
    for neg in negatives:
        if neg in lower_cleaned:
            return None
            
    # Check if the remaining part is just filler/invalid
    fillers = {"and", "patient", "user", "none", "null", "unknown", "caller", "", "my name is", "i am", "myself", "this is", "name is"}
    if lower_cleaned in fillers or len(cleaned) < 2 or len(cleaned) > 30:
        return None
    return cleaned


# ── tool functions ────────────────────────────────────────────────────────────

def identify_user(phone: str, name: str = None) -> dict:
    """Create or fetch user by phone number."""
    name = _clean_name(name)

    # ── Validate phone ────────────────────────────────────────────────────────────
    clean = _clean_phone(phone)
    if len(clean) != 10:
        return {
            "error": True,
            "valid_phone": False,
            "message": (
                f"'{phone}' doesn't look like a valid 10-digit phone number. "
                f"Please ask the patient to repeat their number clearly, digit by digit."
            )
        }

    phone = clean   # use sanitised version going forward

    with Session(engine) as session:
        user = session.exec(select(User).where(User.phone == phone)).first()

        if not user:
            # Brand new patient
            user = User(phone=phone, name=name)
            session.add(user)
            session.commit()
            session.refresh(user)
            return {
                "user_id": user.id,
                "phone": user.phone,
                "name": user.name,
                "is_new_user": True,
                "name_mismatch": False,
                "message": f"Welcome! I've created a profile for you."
            }

        # ── Name mismatch check ───────────────────────────────────────────────
        # Caller gave a name that differs from what's stored in the DB.
        # Flag this so Mia can verify identity instead of silently accepting.
        name_mismatch = (
            name
            and user.name
            and name.strip().lower() != user.name.strip().lower()
        )

        if name_mismatch:
            return {
                "user_id": user.id,
                "phone": user.phone,
                "name_in_db": user.name,
                "name_provided": name,
                "is_new_user": False,
                "name_mismatch": True,
                "message": (
                    f"This phone number is already registered to '{user.name}', "
                    f"but the caller said their name is '{name}'. "
                    f"Please ask the caller to confirm their identity."
                )
            }

        # Update name if it was missing before
        if name and not user.name:
            user.name = name
            session.add(user)
            session.commit()
            session.refresh(user)

        return {
            "user_id": user.id,
            "phone": user.phone,
            "name": user.name,
            "is_new_user": False,
            "name_mismatch": False,
            "message": f"Welcome back{', ' + user.name if user.name else ''}!"
        }


def fetch_slots(date: str = None) -> dict:
    """Return available appointment slots."""
    if date and date in AVAILABLE_SLOTS:
        return {
            "date": date,
            "available_slots": AVAILABLE_SLOTS[date],
            "message": f"Available slots on {date}"
        }
    # Return next 3 available days
    preview = dict(list(AVAILABLE_SLOTS.items())[:3])
    return {
        "available_days": preview,
        "message": "Here are the next available days"
    }


def book_appointment(user_id: int, date: str, time: str) -> dict:
    """Book a slot, preventing double-booking."""
    with Session(engine) as session:
        # Check slot is in available list
        day_slots = AVAILABLE_SLOTS.get(date, [])
        if time not in day_slots:
            return {"success": False, "error": f"Slot {time} on {date} is not available."}

        # Prevent double booking
        existing = session.exec(
            select(Appointment).where(
                Appointment.date == date,
                Appointment.time == time,
                Appointment.status == "booked"
            )
        ).first()
        if existing:
            return {"success": False, "error": "That slot is already taken. Please choose another."}

        appt = Appointment(user_id=user_id, date=date, time=time)
        session.add(appt)
        session.commit()
        session.refresh(appt)
        return {
            "success": True,
            "appointment_id": appt.id,
            "date": date,
            "time": time,
            "message": f"Appointment confirmed for {date} at {time}."
        }


def retrieve_appointments(user_id: int) -> dict:
    """Get all active appointments for a user."""
    with Session(engine) as session:
        appts = session.exec(
            select(Appointment).where(
                Appointment.user_id == user_id,
                Appointment.status == "booked"
            )
        ).all()
        if not appts:
            return {"appointments": [], "message": "No upcoming appointments found."}
        return {
            "appointments": [
                {"id": a.id, "date": a.date, "time": a.time, "status": a.status}
                for a in appts
            ],
            "count": len(appts),
            "message": f"Found {len(appts)} upcoming appointment(s)."
        }


def cancel_appointment(appointment_id: int) -> dict:
    """Cancel an appointment by ID."""
    with Session(engine) as session:
        appt = session.get(Appointment, appointment_id)
        if not appt:
            return {"success": False, "error": "Appointment not found."}
        if appt.status == "cancelled":
            return {"success": False, "error": "Appointment is already cancelled."}
        appt.status = "cancelled"
        session.add(appt)
        session.commit()
        return {
            "success": True,
            "appointment_id": appointment_id,
            "message": f"Appointment on {appt.date} at {appt.time} has been cancelled."
        }


def modify_appointment(appointment_id: int, new_date: str, new_time: str) -> dict:
    """Reschedule an appointment."""
    with Session(engine) as session:
        appt = session.get(Appointment, appointment_id)
        if not appt:
            return {"success": False, "error": "Appointment not found."}

        # Check new slot availability
        day_slots = AVAILABLE_SLOTS.get(new_date, [])
        if new_time not in day_slots:
            return {"success": False, "error": f"Slot {new_time} on {new_date} is not available."}

        # Prevent double booking on new slot
        conflict = session.exec(
            select(Appointment).where(
                Appointment.date == new_date,
                Appointment.time == new_time,
                Appointment.status == "booked",
                Appointment.id != appointment_id
            )
        ).first()
        if conflict:
            return {"success": False, "error": "That new slot is already taken."}

        old_date, old_time = appt.date, appt.time
        appt.date = new_date
        appt.time = new_time
        session.add(appt)
        session.commit()
        return {
            "success": True,
            "appointment_id": appointment_id,
            "old_date": old_date,
            "old_time": old_time,
            "new_date": new_date,
            "new_time": new_time,
            "message": f"Appointment rescheduled from {old_date} {old_time} to {new_date} at {new_time}."
        }


def end_conversation(summary: str, user_intent: str = None, appointments_booked: list = None) -> dict:
    """Signal end of conversation and return summary."""
    return {
        "summary": summary,
        "user_intent": user_intent,
        "appointments_booked": appointments_booked or [],
        "timestamp": datetime.utcnow().isoformat()
    }


# ── dispatcher ────────────────────────────────────────────────────────────────
TOOL_MAP = {
    "identify_user": identify_user,
    "fetch_slots": fetch_slots,
    "book_appointment": book_appointment,
    "retrieve_appointments": retrieve_appointments,
    "cancel_appointment": cancel_appointment,
    "modify_appointment": modify_appointment,
    "end_conversation": end_conversation,
}


def execute_tool(name: str, args: dict) -> dict:
    fn = TOOL_MAP.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"Tool argument error: {str(e)}"}
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}

import os
import json
import re
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager, contextmanager

import httpx 
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from openai import AsyncOpenAI
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# -------------------------
# ENVIRONMENT
# -------------------------
load_dotenv()

WAZZUP_API_KEY = os.getenv("WAZZUP_API_KEY")
WAZZUP_CHANNEL_ID = os.getenv("WAZZUP_CHANNEL_ID")
TARGET_CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID", WAZZUP_CHANNEL_ID)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DEBOUNCE_SECONDS = int(os.getenv("DEBOUNCE_SECONDS", "20"))
ADMIN_PHONES_RAW = os.getenv("ADMIN_PHONES", "87779555889,77476688423")

db_pool: pool.ThreadedConnectionPool = None
ai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

INPUT_PRICE_PER_1M = 0.80
OUTPUT_PRICE_PER_1M = 3.20

UTC_PLUS_5 = timezone(timedelta(hours=5))


# -------------------------
# TIME HELPERS
# -------------------------
def now_utc_plus_5() -> datetime:
    return datetime.now(UTC_PLUS_5)

def is_bot_working_time() -> bool:
    hour = now_utc_plus_5().hour
    return hour >= 22 or hour < 7

def format_time_utc_plus_5() -> str:
    return now_utc_plus_5().strftime("%Y-%m-%d %H:%M:%S")


# -------------------------
# ADMIN CONFIG
# -------------------------
def parse_admin_phones(raw: str) -> set[str]:
    result = set()
    for x in (raw or "").split(","):
        x = re.sub(r"\D", "", x.strip())
        if x:
            result.add(x)
    return result

ADMIN_PHONES = parse_admin_phones(ADMIN_PHONES_RAW)


# -------------------------
# APP LIFECYCLE
# -------------------------
def init_db_pool():
    global db_pool
    db_pool = pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=20,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="prefer",
        connect_timeout=10,
        options="-c timezone=Etc/GMT-5"
    )
    log.info("DB pool initialized")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db_pool()
    log.info("App started")
    yield
    if db_pool:
        db_pool.closeall()
    log.info("DB pool closed")

app = FastAPI(lifespan=lifespan)


# -------------------------
# HELPERS
# -------------------------
def load_products_info() -> str:
    with open("products.txt", "r", encoding="utf-8") as f:
        return f.read()

@contextmanager
def get_db():
    conn = db_pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)

def normalize_phone(chat_id: str) -> str:
    return re.sub(r"\D", "", (chat_id or "").split("@")[0].strip())

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None

def normalize_language_code(lang: str | None) -> str | None:
    if not lang:
        return None
    lang = lang.lower().strip()

    if lang in ["kz", "kk", "kaz", "kazakh", "қазақ", "kazaksha", "kazakhsha"]:
        return "kz"

    if lang in ["ru", "rus", "russian", "орыс", "рус", "по-русски"]:
        return "ru"

    return None

def approx_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost = (prompt_tokens or 0) * INPUT_PRICE_PER_1M / 1_000_000
    output_cost = (completion_tokens or 0) * OUTPUT_PRICE_PER_1M / 1_000_000
    return input_cost + output_cost

def usage_to_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if not usage:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "approx_cost_usd": 0.0,
        }

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    approx_cost_usd = calculate_cost(prompt_tokens, completion_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "approx_cost_usd": approx_cost_usd,
    }

def looks_like_short_name_reply(text: str) -> bool:
    if not text:
        return False
    text = clean_text(text)
    if len(text.split()) > 2:
        return False
    return bool(re.fullmatch(r"[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһЁё\- ]{2,40}", text))

def summarize_students(students: list[dict]) -> list[dict]:
    result = []
    for s in students:
        result.append({
            "student_id": s["id"],
            "child_order": s.get("child_order"),
            "relation_label": s.get("relation_label"),
            "student_name": s.get("student_name"),
            "grade": s.get("grade"),
            "goal": s.get("goal"),
            "course_interest": s.get("course_interest"),
            "study_language": s.get("study_language"),
            "education_level": s.get("education_level"),
            "progress_notes": s.get("progress_notes"),
        })
    return result

def has_student_payload(extracted: dict) -> bool:
    keys = [
        "student_name",
        "grade",
        "study_format",
        "study_language",
        "education_level",
        "goal",
        "course_interest",
        "target_school",
        "progress_notes",
    ]
    return any(extracted.get(k) not in [None, ""] for k in keys)

def build_known_facts_summary(client_state: dict, active_student: dict | None, students_summary: list[dict]) -> dict:
    return {
        "known_parent_name": client_state.get("parent_name"),
        "known_preferred_language": client_state.get("preferred_language"),
        "known_wants_offline": client_state.get("wants_offline"),
        "known_call_time_preference": client_state.get("call_time_preference"),
        "active_student_id": active_student.get("id") if active_student else None,
        "active_student_name": active_student.get("student_name") if active_student else None,
        "active_student_grade": active_student.get("grade") if active_student else None,
        "active_student_goal": active_student.get("goal") if active_student else None,
        "active_student_course_interest": active_student.get("course_interest") if active_student else None,
        "active_student_progress_notes": active_student.get("progress_notes") if active_student else None,
        "active_student_education_level": active_student.get("education_level") if active_student else None,
        "active_student_study_language": active_student.get("study_language") if active_student else None,
        "students_count": len(students_summary),
    }

def build_missing_summary(client_state: dict, active_student: dict | None, student_count: int, ambiguous_child: bool) -> dict:
    return {
        "child_clarify": ambiguous_child or (student_count > 1 and active_student is None),
        "goal_missing": not bool(active_student and (active_student.get("goal") or active_student.get("course_interest"))),
        "grade_missing": not bool(active_student and active_student.get("grade")),
        "progress_missing": not bool(active_student and (active_student.get("education_level") or active_student.get("progress_notes"))),
        "parent_name_missing": not bool(client_state.get("parent_name")),
        "study_language_missing": not bool(active_student and active_student.get("study_language")),
        "student_count": student_count,
    }


# -------------------------
# CHAT DEBOUNCE
# -------------------------
class ChatState:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.task = None
        self.buffer = []

_chat_states = {}

def get_chat_state(chat_id: str) -> ChatState:
    if chat_id not in _chat_states:
        _chat_states[chat_id] = ChatState()
    return _chat_states[chat_id]

async def cancel_all_debounce_tasks():
    for _, state in _chat_states.items():
        async with state.lock:
            if state.task and not state.task.done():
                state.task.cancel()
            state.buffer = []
    log.warning("All debounce tasks cancelled")


# -------------------------
# BOT SETTINGS
# -------------------------
def is_global_bot_enabled() -> bool:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT global_bot_enabled
                FROM bot_settings
                WHERE id = 1
            """)
            row = cur.fetchone()
            if not row:
                return True
            return bool(row[0])

def set_global_bot_enabled(enabled: bool):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_settings (id, global_bot_enabled, updated_at)
                VALUES (1, %s, NOW())
                ON CONFLICT (id)
                DO UPDATE SET
                    global_bot_enabled = EXCLUDED.global_bot_enabled,
                    updated_at = NOW()
            """, (enabled,))
            conn.commit()

def set_client_ai_block(phone: str, blocked: bool):
    phone = re.sub(r"\D", "", phone or "")
    if not phone:
        return

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE clients
                SET ai_blocked = %s,
                    updated_at = NOW()
                WHERE phone = %s
            """, (blocked, phone))
            conn.commit()

def get_blockable_numbers(limit: int = 50) -> list[dict]:
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT phone, parent_name, ai_blocked, updated_at
                FROM clients
                ORDER BY updated_at DESC NULLS LAST, id DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

def is_admin_phone(phone: str) -> bool:
    return phone in ADMIN_PHONES

def build_admin_menu_text() -> str:
    global_enabled = is_global_bot_enabled()
    clients = get_blockable_numbers(limit=30)

    lines = []
    lines.append("ADMIN MENU")
    lines.append(f"Global bot: {'ON' if global_enabled else 'OFF'}")
    lines.append(f"Current time UTC+5: {format_time_utc_plus_5()}")
    lines.append("Working hours UTC+5: 22:00 - 07:00")
    lines.append("")
    lines.append("Commands:")
    lines.append("menu")
    lines.append("stop bot")
    lines.append("start bot")
    lines.append("block 77779555889")
    lines.append("unblock 77779555889")
    lines.append("")
    lines.append("Numbers:")

    if not clients:
        lines.append("No client numbers yet.")
    else:
        for idx, c in enumerate(clients, start=1):
            phone = c.get("phone") or ""
            name = c.get("parent_name") or "-"
            status = "BLOCKED" if c.get("ai_blocked") else "ACTIVE"
            lines.append(f"{idx}. {phone} | {name} | {status}")

    lines.append("")
    lines.append("How to block:")
    lines.append("Send: block 77779555889")
    lines.append("How to unblock:")
    lines.append("Send: unblock 77779555889")
    lines.append("phone format: digits only, for example 77779555889")

    return "\n".join(lines)

async def handle_admin_command(chat_id: str, phone: str, text: str) -> bool:
    command = clean_text(text).lower()

    if command == "menu":
        await send_whatsapp(chat_id, build_admin_menu_text())
        return True

    if command == "stop bot":
        set_global_bot_enabled(False)
        await cancel_all_debounce_tasks()
        await send_whatsapp(chat_id, "Bot fully stopped for all numbers.")
        return True

    if command == "start bot":
        set_global_bot_enabled(True)
        await send_whatsapp(chat_id, "Bot started again for all numbers.")
        return True

    block_match = re.fullmatch(r"block\s+(\d{11,15})", command)
    if block_match:
        target = block_match.group(1)
        set_client_ai_block(target, True)
        await send_whatsapp(chat_id, f"Blocked bot for {target}.")
        return True

    unblock_match = re.fullmatch(r"unblock\s+(\d{11,15})", command)
    if unblock_match:
        target = unblock_match.group(1)
        set_client_ai_block(target, False)
        await send_whatsapp(chat_id, f"Unblocked bot for {target}.")
        return True

    return False


# -------------------------
# DB: CLIENTS / STUDENTS
# -------------------------
def get_or_create_client(phone: str) -> int:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM clients WHERE phone=%s", (phone,))
            row = cur.fetchone()
            if row:
                return row[0]

            cur.execute(
                "INSERT INTO clients (phone, preferred_language) VALUES (%s, 'kz') RETURNING id",
                (phone,)
            )
            client_id = cur.fetchone()[0]
            conn.commit()
            return client_id

def get_client_state(client_id: int) -> dict:
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    id AS client_id,
                    phone,
                    parent_name,
                    preferred_language,
                    wants_offline,
                    call_time_preference,
                    ai_blocked
                FROM clients
                WHERE id = %s
            """, (client_id,))
            row = cur.fetchone()
            return dict(row) if row else {}

def get_students_for_client(client_id: int) -> list[dict]:
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    id,
                    client_id,
                    child_order,
                    relation_label,
                    student_name,
                    grade,
                    study_format,
                    study_language,
                    education_level,
                    goal,
                    course_interest,
                    target_school,
                    progress_notes,
                    created_at,
                    updated_at
                FROM students
                WHERE client_id = %s
                ORDER BY child_order NULLS LAST, id
            """, (client_id,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

def get_student_by_id(student_id: int | None) -> dict | None:
    if not student_id:
        return None

    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    id,
                    client_id,
                    child_order,
                    relation_label,
                    student_name,
                    grade,
                    study_format,
                    study_language,
                    education_level,
                    goal,
                    course_interest,
                    target_school,
                    progress_notes
                FROM students
                WHERE id = %s
            """, (student_id,))
            row = cur.fetchone()
            return dict(row) if row else None

def get_next_child_order(client_id: int) -> int:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(MAX(child_order), 0) + 1
                FROM students
                WHERE client_id = %s
            """, (client_id,))
            row = cur.fetchone()
            return row[0] if row else 1

def create_student_for_client(client_id: int, extracted: dict, relation_label: str | None = None) -> int:
    child_order = get_next_child_order(client_id)

    student_name = extracted.get("student_name")
    grade = safe_int(extracted.get("grade"))
    study_format = extracted.get("study_format")
    study_language = extracted.get("study_language")
    education_level = extracted.get("education_level")
    goal = extracted.get("goal")
    course_interest = extracted.get("course_interest")
    target_school = extracted.get("target_school")
    progress_notes = extracted.get("progress_notes")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO students (
                    client_id,
                    child_order,
                    relation_label,
                    student_name,
                    grade,
                    study_format,
                    study_language,
                    education_level,
                    goal,
                    course_interest,
                    target_school,
                    progress_notes
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                client_id,
                child_order,
                relation_label,
                student_name,
                grade,
                study_format,
                study_language,
                education_level,
                goal,
                course_interest,
                target_school,
                progress_notes
            ))
            student_id = cur.fetchone()[0]
            conn.commit()
            return student_id

def update_student_fields(student_id: int, extracted: dict):
    allowed = {
        "student_name": "student_name",
        "grade": "grade",
        "study_format": "study_format",
        "study_language": "study_language",
        "education_level": "education_level",
        "goal": "goal",
        "course_interest": "course_interest",
        "target_school": "target_school",
        "progress_notes": "progress_notes",
    }

    fields = []
    values = []

    for key, column in allowed.items():
        value = extracted.get(key)
        if value not in [None, ""]:
            if key == "grade":
                value = safe_int(value)
            fields.append(f"{column} = %s")
            values.append(value)

    if not fields:
        return

    fields.append("updated_at = NOW()")
    values.append(student_id)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE students SET {', '.join(fields)} WHERE id = %s",
                tuple(values)
            )
            conn.commit()

def update_clients_fields(client_id: int, extracted: dict):
    allowed = {
        "parent_name": "parent_name",
        "preferred_language": "preferred_language",
        "wants_offline": "wants_offline",
        "call_time_preference": "call_time_preference",
    }

    fields = []
    values = []

    for key, column in allowed.items():
        value = extracted.get(key)
        if value not in [None, ""]:
            if key == "preferred_language":
                value = normalize_language_code(value)
                if not value:
                    continue
            fields.append(f"{column} = %s")
            values.append(value)

    if not fields:
        return

    fields.append("updated_at = NOW()")
    values.append(client_id)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE clients SET {', '.join(fields)} WHERE id = %s",
                tuple(values)
            )
            conn.commit()


# -------------------------
# DB: CONVERSATIONS / MESSAGES
# -------------------------
def get_or_create_conversation(chat_id: str) -> int:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM conversations WHERE chat_id=%s", (chat_id,))
            row = cur.fetchone()
            if row:
                return row[0]

            cur.execute("""
                INSERT INTO conversations (chat_id, current_student_id)
                VALUES (%s, NULL)
                RETURNING id
            """, (chat_id,))
            conv_id = cur.fetchone()[0]
            conn.commit()
            return conv_id

def save_message(conversation_id: int, role: str, text: str):
    token_estimate = max(1, len(text or "") // 4)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (conversation_id, role, message_text, token_estimate)
                VALUES (%s, %s, %s, %s)
            """, (conversation_id, role, text, token_estimate))
            cur.execute(
                "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                (conversation_id,)
            )
            conn.commit()

def save_message_if_new(conversation_id: int, role: str, text: str):
    text = clean_text(text)
    if not text:
        return

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, message_text
                FROM messages
                WHERE conversation_id = %s
                ORDER BY id DESC
                LIMIT 1
            """, (conversation_id,))
            row = cur.fetchone()

            if row and row[0] == role and clean_text(row[1]) == text:
                return

            token_estimate = max(1, len(text) // 4)
            cur.execute("""
                INSERT INTO messages (conversation_id, role, message_text, token_estimate)
                VALUES (%s, %s, %s, %s)
            """, (conversation_id, role, text, token_estimate))
            cur.execute("""
                UPDATE conversations
                SET updated_at = NOW()
                WHERE id = %s
            """, (conversation_id,))
            conn.commit()

def get_recent_messages(conversation_id: int, limit: int = 12) -> list[dict]:
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, message_text
                FROM messages
                WHERE conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
            """, (conversation_id, limit))
            rows = cur.fetchall()

    rows = list(reversed(rows))
    return [{"role": r["role"], "content": r["message_text"]} for r in rows]

def get_recent_messages_for_extraction(conversation_id: int, limit: int = 20) -> list[dict]:
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, message_text
                FROM messages
                WHERE conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
            """, (conversation_id, limit))
            rows = cur.fetchall()

    rows = list(reversed(rows))
    return [{"role": r["role"], "content": r["message_text"]} for r in rows]

def has_assistant_replied(conversation_id: int) -> bool:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1
                FROM messages
                WHERE conversation_id = %s AND role = 'assistant'
                LIMIT 1
            """, (conversation_id,))
            return cur.fetchone() is not None

def get_last_assistant_message(conversation_id: int) -> str | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT message_text
                FROM messages
                WHERE conversation_id = %s AND role = 'assistant'
                ORDER BY id DESC
                LIMIT 1
            """, (conversation_id,))
            row = cur.fetchone()
            return row[0] if row else None

def get_current_student_id(conversation_id: int) -> int | None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT current_student_id
                FROM conversations
                WHERE id = %s
            """, (conversation_id,))
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else None

def set_current_student_id(conversation_id: int, student_id: int | None):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE conversations
                SET current_student_id = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (student_id, conversation_id))
            conn.commit()


# -------------------------
# RESOLVE ACTIVE STUDENT
# -------------------------
def find_student_by_name(students: list[dict], student_name: str | None) -> dict | None:
    if not student_name:
        return None

    target = clean_text(student_name).lower()
    for s in students:
        existing = clean_text(s.get("student_name") or "").lower()
        if existing and existing == target:
            return s

    return None

def find_student_by_child_reference(students: list[dict], child_reference: str | None) -> dict | None:
    if not child_reference or not students:
        return None

    ordered = sorted(students, key=lambda x: (x.get("child_order") or 999999, x["id"]))

    if child_reference == "first_child":
        return ordered[0]
    if child_reference == "second_child" and len(ordered) >= 2:
        return ordered[1]
    if child_reference == "third_child" and len(ordered) >= 3:
        return ordered[2]
    if child_reference == "older_child":
        return ordered[0]
    if child_reference == "younger_child":
        return ordered[-1]

    return None

def resolve_student_target(
    client_id: int,
    conversation_id: int,
    extracted: dict
) -> tuple[int | None, bool]:
    students = get_students_for_client(client_id)
    current_student_id = get_current_student_id(conversation_id)
    current_student = get_student_by_id(current_student_id) if current_student_id else None

    child_reference = extracted.get("child_reference")
    new_child_indicator = bool(extracted.get("new_child_indicator"))
    student_payload_present = has_student_payload(extracted)

    if not student_payload_present and not child_reference and not new_child_indicator:
        return (current_student_id, False)

    if new_child_indicator:
        relation_label = child_reference if child_reference not in [None, "", "current_child"] else "another_child"
        new_id = create_student_for_client(client_id, extracted, relation_label=relation_label)
        return (new_id, False)

    matched_by_name = find_student_by_name(students, extracted.get("student_name"))
    if matched_by_name:
        return (matched_by_name["id"], False)

    matched_by_ref = find_student_by_child_reference(students, child_reference)
    if matched_by_ref:
        return (matched_by_ref["id"], False)

    if current_student and student_payload_present and not new_child_indicator:
        return (current_student["id"], False)

    if len(students) == 1:
        return (students[0]["id"], False)

    if len(students) == 0 and student_payload_present:
        new_id = create_student_for_client(client_id, extracted, relation_label="first_child")
        return (new_id, False)

    if len(students) >= 1 and extracted.get("student_name") and not matched_by_name:
        new_id = create_student_for_client(client_id, extracted, relation_label="another_child")
        return (new_id, False)

    if len(students) > 1 and student_payload_present:
        return (None, True)

    return (current_student_id, False)

def get_active_student_for_reply(client_id: int, conversation_id: int) -> dict | None:
    current_student_id = get_current_student_id(conversation_id)
    if current_student_id:
        student = get_student_by_id(current_student_id)
        if student and student.get("client_id") == client_id:
            return student

    students = get_students_for_client(client_id)
    if len(students) == 1:
        return students[0]

    return None


# -------------------------
# AI EXTRACTOR
# -------------------------
async def extract_lead_fields(
    products_info: str,
    latest_message: str,
    recent_history: list[dict],
    client_state: dict,
    students_summary: list[dict],
    last_assistant_message: str | None,
    current_student_id: int | None
) -> tuple[dict, dict]:
    system_prompt = f"""
{products_info}

You are a strict extraction engine for a WhatsApp sales bot of an education company.

Main job:
- Extract ONLY facts clearly supported by the latest message plus recent conversation context.
- Use database state to understand what is already known.
- Never ask questions.
- Never generate reply text.
- Return ONLY valid JSON.

Critical rules:
- The parent is usually the person writing.
- Do not confuse parent_name with student_name.
- If parent_name is already known in DB, do not overwrite it with a random short name unless the message clearly gives a new parent name.
- If one child is already known and the new short name appears after discussion about a different child, treat it carefully.
- Prefer recent conversation context over isolated message interpretation.
- Prefer DB facts over guessing.
- If a field is unclear, return null.
- Do not invent anything.

Language rules:
- Detect whether the user's latest message is mainly Kazakh or Russian.
- language_code can only be "kz", "ru", or null.
- confidence_language must be a number from 0 to 1.

Multi-child rules:
- child_reference can be one of:
  "current_child", "first_child", "second_child", "third_child", "older_child", "younger_child", "another_child", or null.
- new_child_indicator should be true only if the user clearly introduces another child.

Grade rules:
- grade must be integer if clearly stated, else null.

JSON schema:
{
  "parent_name": string|null,
  "student_name": string|null,
  "grade": integer|null,
  "study_format": string|null,
  "study_language": string|null,
  "preferred_language": string|null,
  "education_level": string|null,
  "goal": string|null,
  "course_interest": string|null,
  "target_school": string|null,
  "progress_notes": string|null,
  "wants_offline": boolean|null,
  "asks_free_courses": boolean|null,
  "asks_price": boolean|null,
  "asks_direct_question": boolean|null,
  "ready_for_call": boolean|null,
  "call_time_preference": string|null,
  "language_code": string|null,
  "confidence_language": number|null,
  "new_child_indicator": boolean|null,
  "child_reference": string|null
}
""".strip()

    user_payload = {
        "latest_message": latest_message,
        "recent_history": recent_history,
        "current_client_state": client_state,
        "current_children": students_summary,
        "last_assistant_message": last_assistant_message,
        "current_student_id": current_student_id
    }

    response = await ai_client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        max_tokens=260,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ]
    )

    usage_info = usage_to_dict(response)
    raw = response.choices[0].message.content.strip()

    fallback = {
        "parent_name": None,
        "student_name": None,
        "grade": None,
        "study_format": None,
        "study_language": None,
        "preferred_language": None,
        "education_level": None,
        "goal": None,
        "course_interest": None,
        "target_school": None,
        "progress_notes": None,
        "wants_offline": None,
        "asks_free_courses": None,
        "asks_price": None,
        "asks_direct_question": None,
        "ready_for_call": None,
        "call_time_preference": None,
        "language_code": None,
        "confidence_language": None,
        "new_child_indicator": None,
        "child_reference": None,
    }

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = fallback.copy()
    except Exception:
        log.warning("Extractor invalid JSON: %s", raw)
        data = fallback.copy()

    for key, default_value in fallback.items():
        if key not in data:
            data[key] = default_value

    data["grade"] = safe_int(data.get("grade"))

    lang = normalize_language_code(data.get("language_code"))
    if lang:
        data["language_code"] = lang
        try:
            confidence = float(data.get("confidence_language")) if data.get("confidence_language") is not None else None
        except Exception:
            confidence = None
        data["confidence_language"] = confidence
        if confidence is not None and confidence >= 0.75:
            data["preferred_language"] = lang
    else:
        data["language_code"] = None
        data["confidence_language"] = None

    latest_clean = clean_text(latest_message)
    last_msg = clean_text(last_assistant_message or "").lower()

    if looks_like_short_name_reply(latest_clean):
        if ("ата-ана" in last_msg or "родител" in last_msg or "ваше имя" in last_msg or "аты-жөніңіз" in last_msg) and not data.get("parent_name"):
            data["parent_name"] = latest_clean
        elif ("бала" in last_msg or "ребен" in last_msg or "имя ребенка" in last_msg or "баланың аты" in last_msg) and not data.get("student_name"):
            data["student_name"] = latest_clean

    return data, usage_info


# -------------------------
# AI REPLY GENERATION
# -------------------------
async def generate_sales_reply(
    products_info: str,
    recent_messages: list[dict],
    client_state: dict,
    active_student: dict | None,
    students_summary: list[dict],
    extracted: dict,
    is_first_reply: bool,
    last_assistant_message: str | None,
    ambiguous_child: bool
) -> tuple[str, dict]:
    lang = client_state.get("preferred_language") or "kz"
    if lang not in ["kz", "ru"]:
        lang = "kz"

    known_facts = build_known_facts_summary(client_state, active_student, students_summary)
    missing_summary = build_missing_summary(
        client_state=client_state,
        active_student=active_student,
        student_count=len(students_summary),
        ambiguous_child=ambiguous_child
    )

    system_prompt = f"""
{products_info}

You are a professional, polite, natural WhatsApp sales manager of Zerdeli online direction.

Core behavior:
- Speak like a real human manager.
- Reply ONLY in {lang}.
- Keep the message short, usually 1-4 lines.
- Ask only one main question.
- First answer the client's direct question if there is one.
- Then move the conversation forward naturally.
- Never mention AI, extraction, database, saved fields, or internal logic.
- No markdown.

Very important:
- The person in chat is usually the parent.
- One parent may have multiple children.
- Do not confuse parent_name with student_name.
- If the parent's name is already known, do not ask it again.
- If the child's name is already known, do not ask it again.
- If grade is already known, do not ask it again.
- If goal or course_interest is already known, do not ask it again.
- If study_language is already known, do not ask it again.
- If several children exist and it is unclear which child the parent means, ask only to clarify which child.
- If parent_name is known but student_name is unknown, refer to the child as "балаңыз" or "ребенок".
- Use a child's actual name only when clearly known for that child.

Sales style:
- Warm, professional, concise.
- Do not sound like a form or checklist.
- Do not repeat already known facts back unnecessarily.
- If enough info is already known, softly move toward a short 5-10 minute call.
- If the client wants offline, politely say this is the online direction and their contact can be passed to the responsible manager.
- Sometimes add one short trust-building sentence about online format, company, olympiad direction, achievements, or results.
- Do not overload the message.
- Use missing_summary only as guidance for what is still unknown.
- Do not follow a rigid script.
- Ask the most natural next useful question based on the whole conversation.

Conversation facts:
- is_first_reply = {json.dumps(is_first_reply, ensure_ascii=False)}
- known_facts = {json.dumps(known_facts, ensure_ascii=False)}
- missing_summary = {json.dumps(missing_summary, ensure_ascii=False)}
- client_state = {json.dumps(client_state, ensure_ascii=False)}
- active_student = {json.dumps(active_student, ensure_ascii=False)}
- students_summary = {json.dumps(students_summary, ensure_ascii=False)}
- extracted = {json.dumps(extracted, ensure_ascii=False)}
- last_assistant_message = {json.dumps(last_assistant_message, ensure_ascii=False)}
- ambiguous_child = {json.dumps(ambiguous_child, ensure_ascii=False)}

Reply shape:
1. Short answer or short transition
2. Optional one short trust/value sentence
3. One best next question only

Never ask for anything that is already known in known_facts.
Never make the message feel like a form.
""".strip()

    response = await ai_client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        max_tokens=180,
        messages=[{"role": "system", "content": system_prompt}] + recent_messages[-10:]
    )

    usage_info = usage_to_dict(response)
    reply_text = response.choices[0].message.content.strip()
    return reply_text, usage_info


# -------------------------
# WHATSAPP SEND
# -------------------------
async def send_whatsapp(chat_id: str, text: str):
    url = "https://api.wazzup24.com/v3/message"
    headers = {
        "Authorization": f"Bearer {WAZZUP_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "channelId": WAZZUP_CHANNEL_ID,
        "chatId": normalize_phone(chat_id),
        "chatType": "whatsapp",
        "text": text
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            log.error("Wazzup send failed: %s", resp.text)


# -------------------------
# MAIN MESSAGE PROCESSING
# -------------------------
async def process_inbound(chat_id: str, merged_text: str, products_info: str):
    merged_text = clean_text(merged_text)
    if not merged_text:
        return

    if not is_global_bot_enabled():
        log.warning("GLOBAL BOT STOP active inside process_inbound, skipping chat_id=%s", chat_id)
        return

    if not is_bot_working_time():
        log.warning("Bot outside working hours inside process_inbound, skipping chat_id=%s time_utc_plus_5=%s", chat_id, format_time_utc_plus_5())
        return

    phone = normalize_phone(chat_id)
    cycle_approx_input_tokens = approx_text_tokens(products_info) + approx_text_tokens(merged_text)

    log.info("=== NEW INBOUND CYCLE START ===")
    log.info("Incoming phone=%s text=%r approx_input_tokens=%s time_utc_plus_5=%s", phone, merged_text, cycle_approx_input_tokens, format_time_utc_plus_5())

    client_id = get_or_create_client(phone)

    client_state_before = get_client_state(client_id)
    if client_state_before.get("ai_blocked"):
        log.info("AI blocked for phone=%s client_id=%s, skipping reply", phone, client_id)
        return

    conversation_id = get_or_create_conversation(chat_id)
    assistant_already_replied = has_assistant_replied(conversation_id)
    last_assistant_message = get_last_assistant_message(conversation_id)

    client_state_before = get_client_state(client_id)
    students_before = get_students_for_client(client_id)
    recent_history_for_extraction = get_recent_messages_for_extraction(conversation_id, limit=20)
    current_student_id_before = get_current_student_id(conversation_id)

    extracted, extractor_usage = await extract_lead_fields(
        products_info=products_info,
        latest_message=merged_text,
        recent_history=recent_history_for_extraction,
        client_state=client_state_before,
        students_summary=summarize_students(students_before),
        last_assistant_message=last_assistant_message,
        current_student_id=current_student_id_before
    )

    update_clients_fields(client_id, extracted)

    target_student_id, ambiguous_child = resolve_student_target(
        client_id=client_id,
        conversation_id=conversation_id,
        extracted=extracted
    )

    if target_student_id:
        update_student_fields(target_student_id, extracted)
        set_current_student_id(conversation_id, target_student_id)
    elif ambiguous_child:
        set_current_student_id(conversation_id, None)

    client_state_after = get_client_state(client_id)
    students_after = get_students_for_client(client_id)
    active_student = get_active_student_for_reply(client_id, conversation_id)
    students_summary = summarize_students(students_after)
    recent_messages = get_recent_messages(conversation_id, limit=12)

    reply_text, reply_usage = await generate_sales_reply(
        products_info=products_info,
        recent_messages=recent_messages,
        client_state=client_state_after,
        active_student=active_student,
        students_summary=students_summary,
        extracted=extracted,
        is_first_reply=not assistant_already_replied,
        last_assistant_message=last_assistant_message,
        ambiguous_child=ambiguous_child
    )

    reply_text = clean_text(reply_text)
    if not reply_text:
        log.warning("Empty assistant reply, skipping send")
        return

    await send_whatsapp(chat_id, reply_text)
    save_message(conversation_id, "assistant", reply_text)

    total_prompt_tokens = extractor_usage["prompt_tokens"] + reply_usage["prompt_tokens"]
    total_completion_tokens = extractor_usage["completion_tokens"] + reply_usage["completion_tokens"]
    total_tokens = extractor_usage["total_tokens"] + reply_usage["total_tokens"]
    total_cost = extractor_usage["approx_cost_usd"] + reply_usage["approx_cost_usd"]

    log.info(
        "Reply sent phone=%s active_student_id=%s total_tokens=%s approx_cost_usd=%.6f time_utc_plus_5=%s",
        phone,
        active_student["id"] if active_student else None,
        total_tokens,
        total_cost,
        format_time_utc_plus_5()
    )
    log.info(
        "Token details extractor=%s reply=%s total_prompt=%s total_completion=%s",
        extractor_usage,
        reply_usage,
        total_prompt_tokens,
        total_completion_tokens
    )
    log.info("=== INBOUND CYCLE END ===")


# -------------------------
# DEBOUNCE QUEUE
# -------------------------
async def _debounced_process(chat_id: str, products_info: str):
    state = get_chat_state(chat_id)

    await asyncio.sleep(DEBOUNCE_SECONDS)

    async with state.lock:
        merged_text = " ".join([clean_text(x) for x in state.buffer if clean_text(x)])
        state.buffer = []
        state.task = None

    if merged_text:
        log.info("Merged message(s) to AI chat_id=%s text=%r", chat_id, merged_text)
        await process_inbound(chat_id, merged_text, products_info)

async def enqueue_message(chat_id: str, text: str, products_info: str):
    state = get_chat_state(chat_id)

    async with state.lock:
        state.buffer.append(text)

        if state.task and not state.task.done():
            state.task.cancel()

        state.task = asyncio.create_task(_debounced_process(chat_id, products_info))


# -------------------------
# WEBHOOK
# -------------------------
@app.post("/wazzup")
async def webhook(request: Request):
    body = await request.json()
    products_info = load_products_info()

    messages = body.get("messages", [])
    log.info("Webhook received messages_count=%s", len(messages))

    for message in messages:
        channel_id = message.get("channelId")
        chat_id = message.get("chatId")
        text = clean_text(message.get("text") or "")
        status = message.get("status")
        is_echo = bool(message.get("isEcho"))

        if channel_id != TARGET_CHANNEL_ID:
            log.info("Webhook skipped other channel channel_id=%r target=%r", channel_id, TARGET_CHANNEL_ID)
            continue

        if not chat_id or not text:
            log.info("Webhook skipped incomplete message chat_id=%r text=%r", chat_id, text)
            continue

        phone = normalize_phone(chat_id)
        log.info(
            "Webhook message phone=%s channel_id=%s status=%s is_echo=%s text=%r time_utc_plus_5=%s",
            phone, channel_id, status, is_echo, text, format_time_utc_plus_5()
        )

        client_id = get_or_create_client(phone)
        conversation_id = get_or_create_conversation(chat_id)

        # Admin commands always work on inbound messages from admin phones.
        if status == "inbound" and not is_echo and is_admin_phone(phone):
            admin_handled = await handle_admin_command(chat_id, phone, text)
            if admin_handled:
                log.info("Admin command handled phone=%s text=%r", phone, text)
                continue

        # Manager message from phone / iframe, save 24/7.
        if is_echo:
            save_message_if_new(conversation_id, "manager", text)
            log.info("Saved MANAGER history phone=%s channel_id=%s text=%r", phone, channel_id, text)
            continue

        # Only inbound client messages below.
        if status != "inbound":
            continue

        # Save client history always.
        save_message_if_new(conversation_id, "user", text)
        log.info("Saved CLIENT history phone=%s channel_id=%s text=%r", phone, channel_id, text)

        # Global stop => history only.
        if not is_global_bot_enabled():
            log.warning("Global stop ON, history saved but AI skipped phone=%s", phone)
            continue

        client_state = get_client_state(client_id)
        if client_state.get("ai_blocked"):
            log.warning("Client blocked, history saved but AI skipped phone=%s client_id=%s", phone, client_id)
            continue

        # Daytime => history only.
        if not is_bot_working_time():
            log.info("Daytime mode: history saved, AI skipped phone=%s time_utc_plus_5=%s", phone, format_time_utc_plus_5())
            continue

        # Night => normal AI flow.
        await enqueue_message(chat_id, text, products_info)

    return {"status": "ok"}


@app.get("/wazzup/health")
def home():
    return {
        "status": "running",
        "admin_phones": sorted(list(ADMIN_PHONES)),
        "target_channel_id": TARGET_CHANNEL_ID,
        "send_channel_id": WAZZUP_CHANNEL_ID,
        "time_utc_plus_5": format_time_utc_plus_5(),
        "working_hours_utc_plus_5": "22:00-07:00",
        "global_bot_enabled": is_global_bot_enabled(),
        "is_bot_working_time_now": is_bot_working_time()
    }

"""Google Sheets and Calendar helper functions."""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import caldav
from icalendar import Alarm, Calendar, Event, vCalAddress, vText
from google.auth import default as google_default_credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import CONFIG

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/calendar"]

USERS_SHEET = "Users"
USERS_COLUMNS = [
    "user_id",
    "telegram_user_id",
    "telegram_username",
    "telegram_full_name",
    "display_name",
    "email",
    "calendar_email",
    "timezone",
    "role",
    "notify_calendar",
    "notify_telegram",
    "created_at",
    "last_seen_at",
    "is_active",
]

PERSONAL_NOTES_SHEET = "PersonalNotes"
PERSONAL_NOTES_COLUMNS = ["id", "user_id", "text", "created_at", "updated_at", "tags"]

PERSONAL_TASKS_SHEET = "PersonalTasks"
PERSONAL_TASKS_COLUMNS = [
    "id",
    "user_id",
    "title",
    "description",
    "status",
    "priority",
    "due_datetime",
    "tags",
    "calendar_event_id",
]

TEAM_TASKS_SHEET = "TeamTasks"
TEAM_TASKS_COLUMNS = [
    "id",
    "owner_user_id",
    "assignee_user_ids",
    "title",
    "description",
    "assignees",
    "status",
    "priority",
    "due_datetime",
    "calendar_event_id",
]

_sheets_service = None
_calendar_service = None
_yandex_client = None
_yandex_calendar = None
_users_cache: List[Dict[str, str]] | None = None


def _get_calendar_provider() -> str:
    return (CONFIG.calendar_provider or "google").strip().lower()


def _is_google_calendar() -> bool:
    provider = _get_calendar_provider()
    return provider in {"google", "gcal", "google_calendar"}


def _is_yandex_calendar() -> bool:
    provider = _get_calendar_provider()
    return provider in {"yandex", "ya", "yandex_calendar"}


def _get_credentials():
    creds_file = CONFIG.google_credentials_file
    if creds_file:
        creds_path = Path(creds_file)
        if not creds_path.exists():
            logger.error("Credentials file missing: %s", creds_path)
            raise RuntimeError("Не найден или некорректен файл сервисного аккаунта")
        try:
            data = json.loads(creds_path.read_text(encoding="utf-8"))
            if data.get("type") != "service_account":
                raise ValueError("Not a service account JSON")
            return service_account.Credentials.from_service_account_info(data, scopes=SCOPES)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to read credentials: %s", exc)
            raise RuntimeError("Не найден или некорректен файл сервисного аккаунта") from exc
    creds, _ = google_default_credentials(scopes=SCOPES)
    return creds


def _with_retries(func: Callable, *args, **kwargs):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except HttpError as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "resp", None)
            logger.warning("Google API error on attempt %s: %s", attempt + 1, exc)
            if attempt >= max_attempts - 1:
                raise
            time.sleep(1 + attempt)
        except OSError as exc:  # network issues
            logger.warning("Network error on attempt %s: %s", attempt + 1, exc)
            if attempt >= max_attempts - 1:
                raise
            time.sleep(1 + attempt)


def _with_retries_caldav(func: Callable, *args, **kwargs):
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CalDAV error on attempt %s: %s", attempt + 1, exc)
            if attempt >= 2:
                raise
            time.sleep(2 + attempt * 2)


def get_sheets_service():
    global _sheets_service
    if _sheets_service is None:
        _sheets_service = build("sheets", "v4", credentials=_get_credentials())
    return _sheets_service


def get_calendar_service():
    global _calendar_service
    if _calendar_service is None:
        _calendar_service = build("calendar", "v3", credentials=_get_credentials())
    return _calendar_service


def _sheet_exists(name: str) -> bool:
    sheets = _with_retries(
        get_sheets_service().spreadsheets().get(spreadsheetId=CONFIG.sheets_id).execute
    )
    return any(s.get("properties", {}).get("title") == name for s in sheets.get("sheets", []))


def _create_sheet_if_missing(name: str, headers: List[str]) -> None:
    if _sheet_exists(name):
        return
    body = {"requests": [{"addSheet": {"properties": {"title": name}}}]}
    _with_retries(get_sheets_service().spreadsheets().batchUpdate(spreadsheetId=CONFIG.sheets_id, body=body).execute)
    _with_retries(
        get_sheets_service().spreadsheets().values().update(
            spreadsheetId=CONFIG.sheets_id,
            range=f"{name}!A1:{chr(64 + len(headers))}1",
            valueInputOption="RAW",
            body={"values": [headers]},
        ).execute
    )


def ensure_structures() -> None:
    try:
        _create_sheet_if_missing(USERS_SHEET, USERS_COLUMNS)
        _create_sheet_if_missing(PERSONAL_NOTES_SHEET, PERSONAL_NOTES_COLUMNS)
        _create_sheet_if_missing(PERSONAL_TASKS_SHEET, PERSONAL_TASKS_COLUMNS)
        _create_sheet_if_missing(TEAM_TASKS_SHEET, TEAM_TASKS_COLUMNS)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to ensure Google structures: %s", exc)
        raise


# Helpers

def _read_values(sheet: str) -> List[List[str]]:
    try:
        result = _with_retries(
            get_sheets_service()
            .spreadsheets()
            .values()
            .get(spreadsheetId=CONFIG.sheets_id, range=f"{sheet}!A2:Z")
            .execute
        )
        return result.get("values", [])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to read sheet %s: %s", sheet, exc)
        return []


def _append_row(sheet: str, row: List[str]) -> None:
    try:
        _with_retries(
            get_sheets_service().spreadsheets().values().append(
                spreadsheetId=CONFIG.sheets_id,
                range=f"{sheet}!A2",
                valueInputOption="RAW",
                body={"values": [row]},
            ).execute
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to append row to %s: %s", sheet, exc)
        raise


def _update_row(sheet: str, row_index: int, row: List[str]) -> None:
    try:
        _with_retries(
            get_sheets_service().spreadsheets().values().update(
                spreadsheetId=CONFIG.sheets_id,
                range=f"{sheet}!A{row_index}:Z{row_index}",
                valueInputOption="RAW",
                body={"values": [row]},
            ).execute
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to update row in %s: %s", sheet, exc)
        raise


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat()


def _normalize_numeric_id(value: str | int | None, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        return str(int(text))
    except (TypeError, ValueError):
        logger.warning("Игнорирую %s: ожидается числовой идентификатор, получено %r", field_name, value)
        return None


def get_valid_chat_id(user: dict) -> int | None:
    raw_chat_id = user.get("telegram_chat_id") or user.get("telegram_user_id")
    normalized = _normalize_numeric_id(raw_chat_id, "telegram_chat_id")
    if normalized is None:
        logger.warning(
            "Пропускаю напоминание для user_id=%r: некорректный chat_id=%r",
            user.get("user_id"),
            raw_chat_id,
        )
        return None
    return int(normalized)


# Users

def _read_users() -> List[Dict[str, str]]:
    global _users_cache
    if _users_cache is not None:
        return _users_cache
    rows = _read_values(USERS_SHEET)
    users: List[Dict[str, str]] = []
    for r in rows:
        if not r:
            continue
        users.append(
            {
                "user_id": r[0],
                "telegram_user_id": r[1],
                "telegram_username": r[2] if len(r) > 2 else "",
                "telegram_full_name": r[3] if len(r) > 3 else "",
                "display_name": r[4] if len(r) > 4 else "",
                "email": r[5] if len(r) > 5 else "",
                "calendar_email": r[6] if len(r) > 6 else "",
                "timezone": r[7] if len(r) > 7 else "",
                "role": r[8] if len(r) > 8 else "",
                "notify_calendar": r[9] if len(r) > 9 else "TRUE",
                "notify_telegram": r[10] if len(r) > 10 else "TRUE",
                "created_at": r[11] if len(r) > 11 else "",
                "last_seen_at": r[12] if len(r) > 12 else "",
                "is_active": r[13] if len(r) > 13 else "TRUE",
            }
        )
    _users_cache = users
    return users


def _write_users(users: List[Dict[str, str]]) -> None:
    body = [
        [
            u.get("user_id", ""),
            u.get("telegram_user_id", ""),
            u.get("telegram_username", ""),
            u.get("telegram_full_name", ""),
            u.get("display_name", ""),
            u.get("email", ""),
            u.get("calendar_email", ""),
            u.get("timezone", ""),
            u.get("role", ""),
            str(u.get("notify_calendar", "TRUE")),
            str(u.get("notify_telegram", "TRUE")),
            u.get("created_at", ""),
            u.get("last_seen_at", ""),
            str(u.get("is_active", "TRUE")),
        ]
        for u in users
    ]
    _with_retries(
        get_sheets_service().spreadsheets().values().clear(
            spreadsheetId=CONFIG.sheets_id, range=f"{USERS_SHEET}!A2:Z"
        ).execute
    )
    if body:
        _with_retries(
            get_sheets_service().spreadsheets().values().append(
                spreadsheetId=CONFIG.sheets_id,
                range=f"{USERS_SHEET}!A2",
                valueInputOption="RAW",
                body={"values": body},
            ).execute
        )
    global _users_cache
    _users_cache = users


def get_user_profile(telegram_user_id: int) -> dict | None:
    for u in _read_users():
        if str(u.get("telegram_user_id")) == str(telegram_user_id):
            return u
    return None


def create_or_update_user_profile(profile: dict) -> dict:
    ensure_structures()
    users = _read_users()
    for idx, user in enumerate(users):
        if str(user.get("telegram_user_id")) == str(profile.get("telegram_user_id")):
            updated = {**user, **profile, "last_seen_at": _now_iso()}
            normalized_chat_id = _normalize_numeric_id(updated.get("telegram_user_id"), "telegram_user_id")
            if normalized_chat_id is not None:
                updated["telegram_user_id"] = normalized_chat_id
                updated["telegram_chat_id"] = normalized_chat_id
            else:
                updated["telegram_chat_id"] = ""
                updated["telegram_user_id"] = ""
            users[idx] = updated
            _write_users(users)
            return updated
    normalized_chat_id = _normalize_numeric_id(profile.get("telegram_user_id"), "telegram_user_id")
    if normalized_chat_id is None:
        normalized_chat_id = ""
    profile = {
        "user_id": profile.get("user_id") or normalized_chat_id or str(profile.get("telegram_user_id")),
        **profile,
        "telegram_user_id": normalized_chat_id,
        "telegram_chat_id": normalized_chat_id,
        "created_at": profile.get("created_at") or _now_iso(),
        "last_seen_at": profile.get("last_seen_at") or _now_iso(),
    }
    users.append(profile)
    _write_users(users)
    return profile


def update_last_seen(telegram_user_id: int) -> None:
    users = _read_users()
    updated = False
    for u in users:
        if str(u.get("telegram_user_id")) == str(telegram_user_id):
            u["last_seen_at"] = _now_iso()
            updated = True
            break
    if updated:
        _write_users(users)


# Notes

def create_personal_note(profile: dict, note_text: str, tags: Optional[List[str]] = None, **_: str) -> str:
    try:
        ensure_structures()
        note_id = str(uuid.uuid4())
        now = _now_iso()
        tags_str = ",".join(tags or [])
        _append_row(PERSONAL_NOTES_SHEET, [note_id, str(profile.get("user_id")), note_text, now, now, tags_str])
        return f"Заметка сохранена (id={note_id})."
    except Exception:  # noqa: BLE001
        return "Не удалось выполнить действие с Google (таблица/календарь). Я пробовал несколько раз. Попробуйте позже."


def read_personal_notes(profile: dict, limit: int = 5, **_: str) -> str:
    ensure_structures()
    notes = _read_values(PERSONAL_NOTES_SHEET)
    filtered = [n for n in notes if n and n[1] == str(profile.get("user_id"))]
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        logger.warning("Invalid limit for read_personal_notes: %r", limit)
        limit_value = 5
    filtered = list(reversed(filtered))[:limit_value]
    if not filtered:
        return "Заметок пока нет."
    lines = [f"• {n[2]} (теги: {n[5] if len(n)>5 else ''})" for n in filtered]
    return "\n".join(lines)


def search_personal_notes(profile: dict, query: str, limit: int = 5, **_: str) -> str:
    ensure_structures()
    notes = _read_values(PERSONAL_NOTES_SHEET)
    filtered = [n for n in notes if n and n[1] == str(profile.get("user_id")) and query.lower() in (n[2].lower())]
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        logger.warning("Invalid limit for search_personal_notes: %r", limit)
        limit_value = 5
    filtered = filtered[:limit_value]
    if not filtered:
        return "Ничего не найдено."
    return "\n".join(f"• {n[2]}" for n in filtered)


def update_personal_note(profile: dict, note_id: str, fields: Dict[str, str], **_: str) -> str:
    ensure_structures()
    rows = _read_values(PERSONAL_NOTES_SHEET)
    for idx, row in enumerate(rows, start=2):
        if row and row[0] == note_id and row[1] == str(profile.get("user_id")):
            text = fields.get("note_text", row[2])
            tags = fields.get("tags", row[5] if len(row) > 5 else "")
            updated = [row[0], row[1], text, row[3], _now_iso(), ",".join(tags) if isinstance(tags, list) else tags]
            _update_row(PERSONAL_NOTES_SHEET, idx, updated)
            return "Заметка обновлена."
    return "Заметка не найдена."


def delete_personal_note(profile: dict, note_id: str, **_: str) -> str:
    ensure_structures()
    rows = _read_values(PERSONAL_NOTES_SHEET)
    keep: List[List[str]] = []
    deleted = False
    for row in rows:
        if row and row[0] == note_id and row[1] == str(profile.get("user_id")):
            deleted = True
            continue
        keep.append(row)
    if deleted:
        _write_sheet_without_headers(PERSONAL_NOTES_SHEET, keep)
        return "Заметка удалена."
    return "Заметка не найдена."


def _write_sheet_without_headers(sheet: str, rows: List[List[str]]) -> None:
    _with_retries(
        get_sheets_service().spreadsheets().values().clear(spreadsheetId=CONFIG.sheets_id, range=f"{sheet}!A2:Z").execute
    )
    if rows:
        _with_retries(
            get_sheets_service().spreadsheets().values().append(
                spreadsheetId=CONFIG.sheets_id,
                range=f"{sheet}!A2",
                valueInputOption="RAW",
                body={"values": rows},
            ).execute
        )


# Tasks

def create_personal_task(profile: dict, **params) -> str:
    try:
        ensure_structures()
        task_id = str(uuid.uuid4())
        row = [
            task_id,
            str(profile.get("user_id")),
            params.get("title", ""),
            params.get("description", ""),
            params.get("status", "todo"),
            params.get("priority", "medium"),
            params.get("due_datetime", ""),
            ",".join(params.get("tags", [])) if isinstance(params.get("tags"), list) else params.get("tags", ""),
            params.get("calendar_event_id", ""),
        ]
        _append_row(PERSONAL_TASKS_SHEET, row)
        if _calendar_configured_for_creation() and params.get("due_datetime"):
            attendees = _collect_attendees([profile.get("user_id")])
            create_or_update_event(
                profile,
                title=params.get("title", ""),
                description=params.get("description", ""),
                start_datetime=params.get("due_datetime", ""),
                end_datetime=params.get("due_datetime", ""),
                attendees=attendees,
            )
        return f"Личная задача создана (id={task_id})."
    except Exception:  # noqa: BLE001
        return "Не удалось выполнить действие с Google (таблица/календарь). Я пробовал несколько раз. Попробуйте позже."


def update_personal_task(profile: dict, task_id: str, fields: Dict[str, str], **_: str) -> str:
    ensure_structures()
    rows = _read_values(PERSONAL_TASKS_SHEET)
    for idx, row in enumerate(rows, start=2):
        if row and row[0] == task_id and row[1] == str(profile.get("user_id")):
            new_row = list(row)
            mapping = {"title": 2, "description": 3, "status": 4, "priority": 5, "due_datetime": 6, "tags": 7}
            for key, pos in mapping.items():
                if key in fields:
                    val = fields[key]
                    if key == "tags" and isinstance(val, list):
                        val = ",".join(val)
                    new_row[pos] = val
            _update_row(PERSONAL_TASKS_SHEET, idx, new_row)
            return "Задача обновлена."
    return "Задача не найдена."


def list_personal_tasks(profile: dict, status: Optional[str] = None, **_: str) -> str:
    ensure_structures()
    rows = _read_values(PERSONAL_TASKS_SHEET)
    tasks = []
    for r in rows:
        if not r or r[1] != str(profile.get("user_id")):
            continue
        if status and r[4] != status:
            continue
        tasks.append(r)
    if not tasks:
        return "Личных задач нет."
    lines = [f"• {t[2]} [{t[4]}] до {t[6]}" for t in tasks]
    return "\n".join(lines)


def create_team_task(profile: dict, **params) -> str:
    try:
        ensure_structures()
        task_id = str(uuid.uuid4())
        assignees = params.get("assignees", []) or []
        if isinstance(assignees, str):
            assignees = [a.strip() for a in assignees.split(",") if a.strip()]
        assignee_ids = params.get("assignee_user_ids") or []
        row = [
            task_id,
            profile.get("user_id", ""),
            ",".join(assignee_ids),
            params.get("title", ""),
            params.get("description", ""),
            ",".join(assignees),
            params.get("status", "todo"),
            params.get("priority", "medium"),
            params.get("due_datetime", ""),
            params.get("calendar_event_id", ""),
        ]
        _append_row(TEAM_TASKS_SHEET, row)
        attendees = _collect_attendees(assignee_ids + [profile.get("user_id")])
        if _calendar_configured_for_creation() and params.get("due_datetime"):
            create_or_update_event(
                profile,
                title=params.get("title", ""),
                description=params.get("description", ""),
                start_datetime=params.get("due_datetime", ""),
                end_datetime=params.get("due_datetime", ""),
                attendees=attendees,
            )
        return f"Командная задача создана (id={task_id})."
    except Exception:  # noqa: BLE001
        return "Не удалось выполнить действие с Google (таблица/календарь). Я пробовал несколько раз. Попробуйте позже."


def update_team_task(profile: dict, task_id: str, fields: Dict[str, str], **_: str) -> str:
    ensure_structures()
    rows = _read_values(TEAM_TASKS_SHEET)
    for idx, row in enumerate(rows, start=2):
        if row and row[0] == task_id:
            new_row = list(row)
            mapping = {"title": 3, "description": 4, "assignees": 5, "status": 6, "priority": 7, "due_datetime": 8}
            for key, pos in mapping.items():
                if key in fields:
                    val = fields[key]
                    if key == "assignees" and isinstance(val, list):
                        val = ",".join(val)
                    new_row[pos] = val
            _update_row(TEAM_TASKS_SHEET, idx, new_row)
            return "Командная задача обновлена."
    return "Командная задача не найдена."


def list_team_tasks(profile: dict, status: Optional[str] = None, **_: str) -> str:
    ensure_structures()
    rows = _read_values(TEAM_TASKS_SHEET)
    tasks = []
    for r in rows:
        if not r:
            continue
        if status and r[6] != status:
            continue
        tasks.append(r)
    if not tasks:
        return "Командных задач нет."
    lines = [f"• {t[3]} [{t[6]}] до {t[8]} (исполнители: {t[5]})" for t in tasks]
    return "\n".join(lines)


# Calendar


def _calendar_configured_for_creation() -> bool:
    provider = _get_calendar_provider()
    if _is_google_calendar():
        return bool(CONFIG.calendar_id)
    if _is_yandex_calendar():
        return bool(CONFIG.yandex_calendar_login and CONFIG.yandex_calendar_password)
    logger.warning("Неизвестный календарный провайдер %r, создание событий отключено", provider)
    return False


def _collect_attendees(user_ids: List[str]) -> List[str]:
    emails: List[str] = []
    for uid in user_ids:
        for user in _read_users():
            if str(user.get("user_id")) != str(uid):
                continue
            if not _parse_bool(user.get("notify_calendar", "TRUE")):
                continue
            email = user.get("calendar_email") or user.get("email")
            if email:
                emails.append(email)
    return emails


def _get_timezone(tz_name: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name or "Europe/Moscow")
    except Exception:  # noqa: BLE001
        logger.warning("Unknown timezone %s, falling back to UTC", tz_name)
        return ZoneInfo("UTC")


def _parse_datetime_with_timezone(value: str, profile: dict) -> dt.datetime:
    tz = _get_timezone(profile.get("timezone") or "Europe/Moscow")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to parse datetime %r, using current time", value)
        return dt.datetime.now(tz)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def _format_datetime_for_output(value: dt.datetime | dt.date) -> str:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    return str(value)


def _get_yandex_calendar():
    global _yandex_client, _yandex_calendar
    if _yandex_calendar is not None:
        return _yandex_calendar
    if not (CONFIG.yandex_calendar_login and CONFIG.yandex_calendar_password):
        raise RuntimeError("Яндекс.Календарь не сконфигурирован.")

    try:
        client = caldav.DAVClient(
            url=CONFIG.yandex_caldav_url or "https://caldav.yandex.ru/",
            username=CONFIG.yandex_calendar_login,
            password=CONFIG.yandex_calendar_password,
        )
        principal = client.principal()
        calendars = _with_retries_caldav(principal.calendars)
        if not calendars:
            raise RuntimeError("Не удалось получить список календарей Яндекс.")

        selected = None
        if CONFIG.yandex_calendar_name:
            for calendar in calendars:
                try:
                    props = calendar.get_properties([caldav.elements.dav.DisplayName()])
                    display_name = props.get("{DAV:}displayname")
                    if display_name == CONFIG.yandex_calendar_name:
                        selected = calendar
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Не удалось прочитать название календаря: %s", exc)
        if selected is None:
            selected = calendars[0]

        _yandex_client = client
        _yandex_calendar = selected
        return selected
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to init Yandex Calendar: %s", exc)
        raise RuntimeError("Не удалось подключиться к Яндекс.Календарю.") from exc


def build_yandex_event_ics(
    title: str,
    description: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    attendees: Optional[List[str]] = None,
    uid: Optional[str] = None,
) -> bytes:
    cal = Calendar()
    cal.add("prodid", "-//AI Secretary//YA//")
    cal.add("version", "2.0")

    event = Event()
    event.add("uid", uid or str(uuid.uuid4()))
    event.add("summary", title)
    event.add("description", description)
    event.add("dtstart", start_dt)
    event.add("dtend", end_dt)
    event.add("dtstamp", dt.datetime.now(dt.UTC))

    organizer = vCalAddress(f"MAILTO:{CONFIG.yandex_calendar_login}")
    event.add("organizer", organizer)

    for attendee in attendees or []:
        attendee_address = vCalAddress(f"MAILTO:{attendee}")
        attendee_address.params["ROLE"] = vText("REQ-PARTICIPANT")
        attendee_address.params["PARTSTAT"] = vText("NEEDS-ACTION")
        attendee_address.params["RSVP"] = vText("TRUE")
        event.add("attendee", attendee_address)

    alarm = Alarm()
    alarm.add("action", "DISPLAY")
    alarm.add("description", "Напоминание")
    alarm.add("trigger", dt.timedelta(minutes=-15))
    event.add_component(alarm)

    cal.add_component(event)
    return cal.to_ical()


def _create_or_update_event_google(
    profile: dict,
    title: str,
    description: str,
    start_datetime: str,
    end_datetime: Optional[str],
    attendees: Optional[List[str]],
    link_task_id: Optional[str],
) -> str:
    if not CONFIG.calendar_id:
        return "Календарь не сконфигурирован."
    service = get_calendar_service()
    event_body = {
        "summary": title,
        "description": description if not link_task_id else f"{description}\nСвязано с задачей: {link_task_id}",
        "start": {"dateTime": start_datetime, "timeZone": profile.get("timezone", "UTC")},
        "end": {"dateTime": end_datetime or start_datetime, "timeZone": profile.get("timezone", "UTC")},
    }
    attendees = attendees or []
    if attendees:
        event_body["attendees"] = [{"email": e} for e in attendees]
    created = _with_retries(
        service.events().insert(calendarId=CONFIG.calendar_id, body=event_body, sendUpdates="all" if attendees else "none").execute
    )
    return f"Событие создано ({created.get('id')})."


def _create_or_update_event_yandex(
    profile: dict,
    title: str,
    description: str,
    start_datetime: str,
    end_datetime: Optional[str],
    attendees: Optional[List[str]],
    link_task_id: Optional[str],
) -> str:
    calendar = _get_yandex_calendar()
    start_dt = _parse_datetime_with_timezone(start_datetime, profile)
    end_dt = _parse_datetime_with_timezone(end_datetime, profile) if end_datetime else start_dt
    full_description = description if not link_task_id else f"{description}\nСвязано с задачей: {link_task_id}"
    event_ics = build_yandex_event_ics(
        title=title,
        description=full_description,
        start_dt=start_dt,
        end_dt=end_dt,
        attendees=attendees,
        uid=link_task_id,
    )
    _with_retries_caldav(calendar.add_event, event_ics)
    return "Событие создано в Яндекс.Календаре."


def create_or_update_event(
    profile: dict,
    title: str,
    description: str,
    start_datetime: str,
    end_datetime: Optional[str] = None,
    attendees: Optional[List[str]] = None,
    link_task_id: Optional[str] = None,
    **_: str,
) -> str:
    try:
        provider = _get_calendar_provider()

        if _is_google_calendar():
            return _create_or_update_event_google(
                profile,
                title,
                description,
                start_datetime,
                end_datetime,
                attendees,
                link_task_id,
            )
        if _is_yandex_calendar():
            return _create_or_update_event_yandex(
                profile,
                title,
                description,
                start_datetime,
                end_datetime,
                attendees,
                link_task_id,
            )
        if CONFIG.calendar_id:
            logger.warning(
                "Неизвестный провайдер %r, использую Google Calendar по умолчанию", provider
            )
            return _create_or_update_event_google(
                profile,
                title,
                description,
                start_datetime,
                end_datetime,
                attendees,
                link_task_id,
            )
        return "Календарный провайдер не настроен."
    except Exception as exc:  # noqa: BLE001
        logger.exception("Calendar operation failed: %s", exc)
        return "Не удалось выполнить действие с календарём. Я пробовал несколько раз. Попробуйте позже."


def _show_calendar_agenda_google(
    profile: dict, from_datetime: Optional[str] = None, to_datetime: Optional[str] = None
) -> str:
    if not CONFIG.calendar_id:
        return "Календарь не сконфигурирован."
    now_iso = dt.datetime.utcnow().isoformat() + "Z"
    params = {
        "calendarId": CONFIG.calendar_id,
        "timeMin": (from_datetime or now_iso),
        "maxResults": 10,
        "singleEvents": True,
        "orderBy": "startTime",
    }
    if to_datetime:
        params["timeMax"] = to_datetime
    events = _with_retries(get_calendar_service().events().list(**params).execute).get("items", [])
    if not events:
        return "Ближайших событий нет."
    lines = []
    for ev in events:
        start = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
        lines.append(f"• {ev.get('summary')} — {start}")
    return "\n".join(lines)


def _show_calendar_agenda_yandex(
    profile: dict, from_datetime: Optional[str] = None, to_datetime: Optional[str] = None
) -> str:
    calendar = _get_yandex_calendar()
    tz = _get_timezone(profile.get("timezone") or "Europe/Moscow")
    start = _parse_datetime_with_timezone(from_datetime, profile) if from_datetime else dt.datetime.now(tz)
    end = _parse_datetime_with_timezone(to_datetime, profile) if to_datetime else start + dt.timedelta(days=7)

    events = _with_retries_caldav(calendar.date_search, start, end)
    if not events:
        return "Ближайших событий нет."

    lines: List[str] = []
    for ev in events:
        try:
            ical_data = Calendar.from_ical(ev.data)
            for component in ical_data.walk("VEVENT"):
                summary = component.get("summary", "Без названия")
                dtstart = component.get("dtstart")
                if dtstart is None:
                    continue
                start_value = dtstart.dt
                if isinstance(start_value, dt.datetime):
                    if start_value.tzinfo is None:
                        start_value = start_value.replace(tzinfo=tz)
                    else:
                        start_value = start_value.astimezone(tz)
                lines.append(f"• {summary} — {_format_datetime_for_output(start_value)}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse Yandex event: %s", exc)
            continue

    if not lines:
        return "Ближайших событий нет."
    return "\n".join(lines)


def show_calendar_agenda(profile: dict, from_datetime: Optional[str] = None, to_datetime: Optional[str] = None, **_: str) -> str:
    try:
        if _is_google_calendar():
            return _show_calendar_agenda_google(profile, from_datetime, to_datetime)
        if _is_yandex_calendar():
            return _show_calendar_agenda_yandex(profile, from_datetime, to_datetime)
        return "Календарный провайдер не настроен."
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to read calendar: %s", exc)
        return "Не удалось выполнить действие с календарём. Я пробовал несколько раз. Попробуйте позже."


# Context for AI

def build_context_for_user(profile: dict) -> dict:
    personal = list_personal_tasks(profile)
    team = list_team_tasks(profile)
    summary = f"Личные задачи: {personal[:200]} | Командные задачи: {team[:200]}"
    return {"summary": summary}


def list_users() -> List[Dict[str, str]]:
    return list(_read_users())


def upcoming_tasks_for_user(user_id: str, within_hours: int = 24) -> List[Dict[str, str]]:
    now = dt.datetime.utcnow()
    soon = now + dt.timedelta(hours=within_hours)
    tasks: List[Dict[str, str]] = []
    for r in _read_values(PERSONAL_TASKS_SHEET):
        if not r or r[1] != str(user_id):
            continue
        due = r[6] if len(r) > 6 else ""
        if not due:
            continue
        try:
            due_dt = dt.datetime.fromisoformat(due)
        except Exception:  # noqa: BLE001
            continue
        if due_dt <= soon:
            tasks.append(
                {
                    "id": r[0],
                    "scope": "personal",
                    "title": r[2],
                    "description": r[3],
                    "status": r[4],
                    "priority": r[5],
                    "due_datetime": due,
                    "assignees": [],
                    "is_overdue": due_dt < now,
                    "is_soon": due_dt <= soon,
                }
            )
    for r in _read_values(TEAM_TASKS_SHEET):
        if not r:
            continue
        assignees_ids = (r[2] if len(r) > 2 else "").split(",")
        if str(user_id) not in assignees_ids:
            continue
        due = r[8] if len(r) > 8 else ""
        if not due:
            continue
        try:
            due_dt = dt.datetime.fromisoformat(due)
        except Exception:  # noqa: BLE001
            continue
        if due_dt <= soon:
            tasks.append(
                {
                    "id": r[0],
                    "scope": "team",
                    "title": r[3],
                    "description": r[4],
                    "status": r[6],
                    "priority": r[7],
                    "due_datetime": due,
                    "assignees": (r[5] if len(r) > 5 else "").split(","),
                    "is_overdue": due_dt < now,
                    "is_soon": due_dt <= soon,
                }
            )
    return tasks

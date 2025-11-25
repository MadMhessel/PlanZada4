"""Google Sheets and Calendar helper functions."""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Dict, List, Optional

from google.oauth2 import service_account
from google.auth import default as google_default_credentials
from googleapiclient.discovery import build

from config import CONFIG

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/calendar",
]

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
    "due_date",
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
    "due_date",
    "calendar_event_id",
]

_sheets_service = None
_calendar_service = None
_users_cache: List[Dict[str, str]] | None = None


def _get_credentials():
    if CONFIG.google_credentials_file and CONFIG.google_credentials_file.exists():
        logger.info("Using service account credentials from %s", CONFIG.google_credentials_file)
        return service_account.Credentials.from_service_account_file(
            str(CONFIG.google_credentials_file), scopes=SCOPES
        )
    creds, _ = google_default_credentials(scopes=SCOPES)
    return creds


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
    sheets = get_sheets_service().spreadsheets().get(spreadsheetId=CONFIG.sheets_id).execute()
    return any(s.get("properties", {}).get("title") == name for s in sheets.get("sheets", []))


def _create_sheet_if_missing(name: str, headers: List[str]) -> None:
    if _sheet_exists(name):
        return
    body = {"requests": [{"addSheet": {"properties": {"title": name}}}]}
    get_sheets_service().spreadsheets().batchUpdate(
        spreadsheetId=CONFIG.sheets_id, body=body
    ).execute()
    get_sheets_service().spreadsheets().values().update(
        spreadsheetId=CONFIG.sheets_id,
        range=f"{name}!A1:{chr(64 + len(headers))}1",
        valueInputOption="RAW",
        body={"values": [headers]},
    ).execute()
    logger.info("Created sheet %s with headers", name)


def ensure_structures() -> None:
    _create_sheet_if_missing(USERS_SHEET, USERS_COLUMNS)
    _create_sheet_if_missing(PERSONAL_NOTES_SHEET, PERSONAL_NOTES_COLUMNS)
    _create_sheet_if_missing(PERSONAL_TASKS_SHEET, PERSONAL_TASKS_COLUMNS)
    _create_sheet_if_missing(TEAM_TASKS_SHEET, TEAM_TASKS_COLUMNS)


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat()


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
                "notify_calendar": r[9] if len(r) > 9 else "FALSE",
                "notify_telegram": r[10] if len(r) > 10 else "FALSE",
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
            str(u.get("notify_calendar", "FALSE")),
            str(u.get("notify_telegram", "FALSE")),
            u.get("created_at", ""),
            u.get("last_seen_at", ""),
            str(u.get("is_active", "TRUE")),
        ]
        for u in users
    ]
    get_sheets_service().spreadsheets().values().clear(
        spreadsheetId=CONFIG.sheets_id, range=f"{USERS_SHEET}!A2:Z"
    ).execute()
    if body:
        get_sheets_service().spreadsheets().values().append(
            spreadsheetId=CONFIG.sheets_id,
            range=f"{USERS_SHEET}!A2",
            valueInputOption="RAW",
            body={"values": body},
        ).execute()
    global _users_cache
    _users_cache = users


def list_users() -> List[Dict[str, str]]:
    return list(_read_users())


def get_user_by_telegram_id(telegram_user_id: int) -> Dict[str, str] | None:
    return next((u for u in _read_users() if str(telegram_user_id) == str(u.get("telegram_user_id"))), None)


def create_user(profile: Dict[str, str]) -> Dict[str, str]:
    ensure_structures()
    users = _read_users()
    users.append(profile)
    _append_row(
        USERS_SHEET,
        [
            profile.get("user_id", ""),
            profile.get("telegram_user_id", ""),
            profile.get("telegram_username", ""),
            profile.get("telegram_full_name", ""),
            profile.get("display_name", ""),
            profile.get("email", ""),
            profile.get("calendar_email", profile.get("email", "")),
            profile.get("timezone", ""),
            profile.get("role", ""),
            str(profile.get("notify_calendar", "FALSE")),
            str(profile.get("notify_telegram", "FALSE")),
            profile.get("created_at", _now_iso()),
            profile.get("last_seen_at", _now_iso()),
            str(profile.get("is_active", "TRUE")),
        ],
    )
    _users_cache = None
    return profile


def update_user_last_seen(user_id: str) -> None:
    users = _read_users()
    changed = False
    for u in users:
        if u.get("user_id") == user_id:
            u["last_seen_at"] = _now_iso()
            changed = True
            break
    if changed:
        _write_users(users)


def update_user_fields_by_telegram(telegram_user_id: int, updates: Dict[str, str]) -> bool:
    users = _read_users()
    changed = False
    for u in users:
        if str(u.get("telegram_user_id")) == str(telegram_user_id):
            u.update(updates)
            changed = True
            break
    if changed:
        _write_users(users)
    return changed


def _read_values(sheet: str) -> List[List[str]]:
    result = (
        get_sheets_service()
        .spreadsheets()
        .values()
        .get(spreadsheetId=CONFIG.sheets_id, range=f"{sheet}!A2:Z")
        .execute()
    )
    return result.get("values", [])


def _append_row(sheet: str, row: List[str]) -> None:
    get_sheets_service().spreadsheets().values().append(
        spreadsheetId=CONFIG.sheets_id,
        range=f"{sheet}!A2",
        valueInputOption="RAW",
        body={"values": [row]},
    ).execute()


def _update_row(sheet: str, row_index: int, row: List[str]) -> None:
    get_sheets_service().spreadsheets().values().update(
        spreadsheetId=CONFIG.sheets_id,
        range=f"{sheet}!A{row_index}:Z{row_index}",
        valueInputOption="RAW",
        body={"values": [row]},
    ).execute()


def add_personal_note(user_id: int, text: str, tags: str) -> str:
    note_id = str(uuid.uuid4())
    now = dt.datetime.utcnow().isoformat()
    _append_row(PERSONAL_NOTES_SHEET, [note_id, str(user_id), text, now, now, tags])
    return note_id


def list_personal_notes(user_id: int, limit: int = 5) -> List[Dict[str, str]]:
    rows = _read_values(PERSONAL_NOTES_SHEET)
    notes = [
        {
            "id": r[0],
            "user_id": r[1],
            "text": r[2],
            "created_at": r[3],
            "updated_at": r[4],
            "tags": r[5] if len(r) > 5 else "",
        }
        for r in rows
        if r and r[1] == str(user_id)
    ]
    return list(reversed(notes))[:limit]


def search_personal_notes(user_id: int, query: str) -> List[Dict[str, str]]:
    return [n for n in list_personal_notes(user_id, limit=100) if query.lower() in n["text"].lower()]


def add_personal_task(user_id: int, data: Dict[str, str]) -> str:
    task_id = str(uuid.uuid4())
    row = [
        task_id,
        str(user_id),
        data.get("title", ""),
        data.get("description", ""),
        data.get("status", "todo"),
        data.get("priority", "medium"),
        data.get("due_date", ""),
        ",".join(data.get("tags", [])) if isinstance(data.get("tags"), list) else data.get("tags", ""),
        data.get("calendar_event_id", ""),
    ]
    _append_row(PERSONAL_TASKS_SHEET, row)
    return task_id


def _update_personal_task(task_id: str, updater) -> bool:
    rows = _read_values(PERSONAL_TASKS_SHEET)
    for idx, row in enumerate(rows, start=2):
        if row and row[0] == task_id:
            new_row = updater(list(row))
            _update_row(PERSONAL_TASKS_SHEET, idx, new_row)
            return True
    return False


def update_personal_task_status(task_id: str, status: str) -> bool:
    return _update_personal_task(task_id, lambda r: r[:4] + [status] + r[5:])


def update_personal_task_due(task_id: str, due_date: str) -> bool:
    return _update_personal_task(task_id, lambda r: r[:6] + [due_date] + r[7:])


def list_personal_tasks(user_id: int, status: Optional[str] = None, due_before: Optional[str] = None) -> List[Dict[str, str]]:
    rows = _read_values(PERSONAL_TASKS_SHEET)
    tasks: List[Dict[str, str]] = []
    for r in rows:
        if not r or r[1] != str(user_id):
            continue
        task = {
            "id": r[0],
            "user_id": r[1],
            "title": r[2],
            "description": r[3],
            "status": r[4],
            "priority": r[5],
            "due_date": r[6] if len(r) > 6 else "",
            "tags": r[7] if len(r) > 7 else "",
            "calendar_event_id": r[8] if len(r) > 8 else "",
        }
        if status and task["status"] != status:
            continue
        if due_before and task["due_date"] and task["due_date"] > due_before:
            continue
        tasks.append(task)
    return tasks


def add_team_task(data: Dict[str, str]) -> str:
    task_id = str(uuid.uuid4())
    row = [
        task_id,
        data.get("owner_user_id", ""),
        ",".join(data.get("assignee_user_ids", [])) if isinstance(data.get("assignee_user_ids", []), list) else data.get("assignee_user_ids", ""),
        data.get("title", ""),
        data.get("description", ""),
        data.get("assignees", ""),
        data.get("status", "todo"),
        data.get("priority", "medium"),
        data.get("due_date", ""),
        data.get("calendar_event_id", ""),
    ]
    _append_row(TEAM_TASKS_SHEET, row)
    return task_id


def _update_team_task(task_id: str, updater) -> bool:
    rows = _read_values(TEAM_TASKS_SHEET)
    for idx, row in enumerate(rows, start=2):
        if row and row[0] == task_id:
            new_row = updater(list(row))
            _update_row(TEAM_TASKS_SHEET, idx, new_row)
            return True
    return False


def update_team_task_status(task_id: str, status: str) -> bool:
    return _update_team_task(task_id, lambda r: r[:6] + [status] + r[7:])


def update_team_task_assignees(task_id: str, assignees: str) -> bool:
    return _update_team_task(task_id, lambda r: r[:5] + [assignees] + r[6:])


def list_team_tasks_by_assignee(name: str) -> List[Dict[str, str]]:
    rows = _read_values(TEAM_TASKS_SHEET)
    tasks = []
    for r in rows:
        if not r:
            continue
        if name.lower() not in (r[5] if len(r) > 5 else "").lower():
            continue
        tasks.append(
            {
                "id": r[0],
                "owner_user_id": r[1] if len(r) > 1 else "",
                "assignee_user_ids": r[2] if len(r) > 2 else "",
                "title": r[3] if len(r) > 3 else "",
                "description": r[4] if len(r) > 4 else "",
                "assignees": r[5] if len(r) > 5 else "",
                "status": r[6] if len(r) > 6 else "",
                "priority": r[7] if len(r) > 7 else "",
                "due_date": r[8] if len(r) > 8 else "",
                "calendar_event_id": r[9] if len(r) > 9 else "",
            }
        )
    return tasks


def list_team_tasks_for_user_id(user_id: str) -> List[Dict[str, str]]:
    rows = _read_values(TEAM_TASKS_SHEET)
    tasks = []
    for r in rows:
        if not r:
            continue
        ids = (r[2] if len(r) > 2 else "").split(",")
        if user_id not in ids:
            continue
        tasks.append(
            {
                "id": r[0],
                "owner_user_id": r[1] if len(r) > 1 else "",
                "assignee_user_ids": r[2] if len(r) > 2 else "",
                "title": r[3] if len(r) > 3 else "",
                "description": r[4] if len(r) > 4 else "",
                "assignees": r[5] if len(r) > 5 else "",
                "status": r[6] if len(r) > 6 else "",
                "priority": r[7] if len(r) > 7 else "",
                "due_date": r[8] if len(r) > 8 else "",
                "calendar_event_id": r[9] if len(r) > 9 else "",
            }
        )
    return tasks


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def find_users_by_names(names: List[str]) -> List[Dict[str, str]]:
    normalized = [n.strip().lower() for n in names if n]
    if not normalized:
        return []
    users = []
    for u in _read_users():
        display = (u.get("display_name") or "").lower()
        username = (u.get("telegram_username") or "").lower()
        if any(n in display or (username and n in username) for n in normalized):
            users.append(u)
    return users


def attendee_emails_for_users(user_ids: List[str]) -> List[str]:
    emails: List[str] = []
    for uid in user_ids:
        user = next((u for u in _read_users() if u.get("user_id") == uid), None)
        if not user:
            continue
        if not _parse_bool(user.get("notify_calendar", "FALSE")):
            continue
        email = user.get("calendar_email") or user.get("email")
        if email:
            emails.append(email)
    return emails


def create_or_update_calendar_event(
    summary: str,
    description: str,
    start_datetime: str,
    end_datetime: Optional[str] = None,
    event_id: Optional[str] = None,
    attendees: Optional[List[str]] = None,
    timezone: str = "UTC",
) -> Optional[str]:
    if not CONFIG.calendar_id or not start_datetime:
        return event_id
    service = get_calendar_service()
    event_body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_datetime, "timeZone": timezone},
        "end": {"dateTime": end_datetime or start_datetime, "timeZone": timezone},
    }
    if attendees:
        event_body["attendees"] = [{"email": email} for email in attendees]
    params = {"calendarId": CONFIG.calendar_id, "sendUpdates": "all" if attendees else "none"}
    if event_id:
        service.events().update(eventId=event_id, body=event_body, **params).execute()
        return event_id
    created = service.events().insert(body=event_body, **params).execute()
    return created.get("id")


def upcoming_tasks_for_user(user_id: str, within_hours: int = 6) -> List[Dict[str, str]]:
    now = dt.datetime.utcnow()
    soon = now + dt.timedelta(hours=within_hours)
    output: List[Dict[str, str]] = []
    try:
        personal = list_personal_tasks(int(user_id), None, None)
    except (TypeError, ValueError):
        personal = []
    for t in personal:
        if not t.get("due_date"):
            continue
        try:
            due = dt.datetime.fromisoformat(t["due_date"])
        except ValueError:
            continue
        if due <= soon:
            output.append({**t, "scope": "personal", "is_overdue": due < now, "is_soon": due <= soon})
    for t in list_team_tasks_for_user_id(user_id):
        if not t.get("due_date"):
            continue
        try:
            due = dt.datetime.fromisoformat(t["due_date"])
        except ValueError:
            continue
        if due <= soon:
            output.append({**t, "scope": "team", "is_overdue": due < now, "is_soon": due <= soon})
    return output

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
    _create_sheet_if_missing(PERSONAL_NOTES_SHEET, PERSONAL_NOTES_COLUMNS)
    _create_sheet_if_missing(PERSONAL_TASKS_SHEET, PERSONAL_TASKS_COLUMNS)
    _create_sheet_if_missing(TEAM_TASKS_SHEET, TEAM_TASKS_COLUMNS)


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
    return _update_team_task(task_id, lambda r: r[:4] + [status] + r[5:])


def update_team_task_assignees(task_id: str, assignees: str) -> bool:
    return _update_team_task(task_id, lambda r: r[:3] + [assignees] + r[4:])


def list_team_tasks_by_assignee(name: str) -> List[Dict[str, str]]:
    rows = _read_values(TEAM_TASKS_SHEET)
    tasks = []
    for r in rows:
        if not r:
            continue
        if name.lower() not in r[3].lower():
            continue
        tasks.append(
            {
                "id": r[0],
                "title": r[1],
                "description": r[2],
                "assignees": r[3],
                "status": r[4],
                "priority": r[5],
                "due_date": r[6] if len(r) > 6 else "",
                "calendar_event_id": r[7] if len(r) > 7 else "",
            }
        )
    return tasks


def create_or_update_calendar_event(task_title: str, description: str, due_date: str, event_id: Optional[str]) -> Optional[str]:
    if not CONFIG.calendar_id or not due_date:
        return event_id
    service = get_calendar_service()
    event_body = {
        "summary": task_title,
        "description": description,
        "start": {"dateTime": due_date, "timeZone": "UTC"},
        "end": {"dateTime": due_date, "timeZone": "UTC"},
    }
    if event_id:
        service.events().update(calendarId=CONFIG.calendar_id, eventId=event_id, body=event_body).execute()
        return event_id
    created = service.events().insert(calendarId=CONFIG.calendar_id, body=event_body).execute()
    return created.get("id")


def upcoming_tasks_for_user(user_id: int, within_hours: int = 6) -> List[Dict[str, str]]:
    tasks = list_personal_tasks(user_id)
    now = dt.datetime.utcnow()
    soon = now + dt.timedelta(hours=within_hours)
    output: List[Dict[str, str]] = []
    for t in tasks:
        if not t.get("due_date"):
            continue
        try:
            due = dt.datetime.fromisoformat(t["due_date"])
        except ValueError:
            continue
        if due <= soon:
            output.append(t)
    return output

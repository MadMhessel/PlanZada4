"""Entry point for Telegram AI Secretary bot."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton

import ai_service
import command_service
from config import CONFIG
import google_service
from debug_service import debug_service
from dialog_logger import dialog_logger
from ai_schemas import PlannedAction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class RegistrationStates(StatesGroup):
    display_name = State()
    calendar_email = State()
    timezone = State()
    notify_calendar = State()
    notify_telegram = State()


router = Router()
bot = Bot(
    token=CONFIG.telegram_token,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher(storage=MemoryStorage())
dp.include_router(router)


def _menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Личные задачи"), KeyboardButton(text="Командные задачи")],
            [KeyboardButton(text="Заметки"), KeyboardButton(text="Настройки")],
        ],
        resize_keyboard=True,
    )


def _yes_no_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Да"), KeyboardButton(text="Нет")]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


async def _handle_ai(message: Message, text: str) -> None:
    profile = google_service.get_user_by_telegram_id(message.from_user.id)
    profile_context = ai_service.build_profile_context(profile) if profile else None
    cls, action = ai_service.analyze_request(text, context_data=profile_context)
    debug_block = ""
    response: str
    if action.confidence < CONFIG.thresholds.low:
        response, debug_meta = command_service.process(
            cls, PlannedAction("clarify", action.confidence, action.params), message.from_user.id
        )
    elif action.confidence < CONFIG.thresholds.high:
        summary = ai_service.build_confirmation(action)
        result, debug_meta = command_service.process(cls, action, message.from_user.id)
        response = f"План действия (требует подтверждения):\n{summary}\n\n{result}\nЕсли неверно — уточните запрос."
    else:
        response, debug_meta = command_service.process(cls, action, message.from_user.id)
    if debug_service.is_debug(message.from_user.id):
        debug_block = "\n\n<code>" + json.dumps(debug_meta, ensure_ascii=False) + "</code>"
    dialog_logger.log(
        {
            "user_id": message.from_user.id,
            "text": text,
            "response": response,
            "timestamp": datetime.utcnow().isoformat(),
            "debug": debug_meta,
        }
    )
    await message.answer(response + debug_block)


def _parse_yes_no(text: str) -> bool | None:
    lower = text.strip().lower()
    if lower in {"да", "yes", "y", "true", "1"}:
        return True
    if lower in {"нет", "no", "n", "false", "0"}:
        return False
    return None


def _is_valid_email(text: str) -> bool:
    cleaned = text.strip()
    return "@" in cleaned and "." in cleaned and len(cleaned.split("@", maxsplit=1)) == 2


async def _start_registration(message: Message, state: FSMContext) -> None:
    google_service.ensure_structures()
    await state.clear()
    await state.set_state(RegistrationStates.display_name)
    await message.answer("Как к вам обращаться? (Имя или имя + фамилия)")


def _profile_summary(profile: Dict[str, str]) -> str:
    lines = [
        f"Имя: {profile.get('display_name', '')}",
        f"Telegram: @{profile.get('telegram_username', '')} ({profile.get('telegram_full_name', '')})",
        f"E-mail: {profile.get('email', '')}",
        f"E-mail для календаря: {profile.get('calendar_email', profile.get('email', ''))}",
        f"Часовой пояс: {profile.get('timezone', '')}",
        f"Роль: {profile.get('role', '')}",
        f"Уведомления в календарь: {profile.get('notify_calendar', 'FALSE')}",
        f"Уведомления в Telegram: {profile.get('notify_telegram', 'FALSE')}",
    ]
    return "\n".join(lines)


async def _ensure_profile(message: Message, state: FSMContext) -> bool:
    google_service.ensure_structures()
    profile = google_service.get_user_by_telegram_id(message.from_user.id)
    if profile:
        google_service.update_user_last_seen(profile.get("user_id", ""))
        return True
    if await state.get_state():
        return False
    await _start_registration(message, state)
    return False


async def _require_profile(message: Message, state: FSMContext) -> bool:
    return await _ensure_profile(message, state)


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext) -> None:
    profile = google_service.get_user_by_telegram_id(message.from_user.id)
    if profile:
        await message.answer(
            "Вы уже зарегистрированы. Ваш профиль:\n" + _profile_summary(profile),
            reply_markup=_menu_keyboard(),
        )
        return
    await message.answer(
        "Привет! Я AI-секретарь. Умею управлять личными и командными задачами, заметками и календарём."
        " Давайте настроим профиль.",
        reply_markup=_menu_keyboard(),
    )
    await _start_registration(message, state)


@router.message(Command("help"))
async def cmd_help(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    await message.answer(
        "Примеры запросов:\n"
        "- напомни завтра позвонить Ивану в 18:00\n"
        "- создай заметку про бюджет проекта\n"
        "- создай задачу для команды по дизайну до пятницы\n"
        "- покажи мои задачи со статусом todo"
    )


@router.message(Command("profile"))
async def cmd_profile(message: Message, state: FSMContext) -> None:
    profile = google_service.get_user_by_telegram_id(message.from_user.id)
    if not profile:
        await _start_registration(message, state)
        return
    await message.answer(_profile_summary(profile))


@router.message(Command("settings"))
async def cmd_settings(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    profile = google_service.get_user_by_telegram_id(message.from_user.id)
    await message.answer(
        _profile_summary(profile)
        + "\n\nИзменить данные: /set_email <email> [calendar_email]\n"
        "Установить часовой пояс: /set_timezone <Europe/Moscow>\n"
        "Уведомления: /set_notify <calendar yes/no> <telegram yes/no>"
    )


@router.message(Command("set_email"))
async def cmd_set_email(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    parts = (message.text or "").split()
    if len(parts) < 2:
        await message.answer("Используйте: /set_email <email> [calendar_email]")
        return
    updates = {"email": parts[1]}
    if len(parts) > 2:
        updates["calendar_email"] = parts[2]
    google_service.update_user_fields_by_telegram(message.from_user.id, updates)
    await message.answer("E-mail обновлён")


@router.message(Command("set_timezone"))
async def cmd_set_timezone(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Используйте: /set_timezone Europe/Moscow")
        return
    google_service.update_user_fields_by_telegram(message.from_user.id, {"timezone": parts[1].strip()})
    await message.answer("Часовой пояс обновлён")


@router.message(Command("set_notify"))
async def cmd_set_notify(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    parts = (message.text or "").split()
    if len(parts) < 3:
        await message.answer("Используйте: /set_notify <calendar yes/no> <telegram yes/no>")
        return
    calendar_flag = _parse_yes_no(parts[1])
    telegram_flag = _parse_yes_no(parts[2])
    updates = {}
    if calendar_flag is not None:
        updates["notify_calendar"] = str(calendar_flag)
    if telegram_flag is not None:
        updates["notify_telegram"] = str(telegram_flag)
    google_service.update_user_fields_by_telegram(message.from_user.id, updates)
    await message.answer("Уведомления обновлены")


@router.message(Command("login"))
async def cmd_login(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Укажите пароль: /login <пароль>")
        return
    await message.answer(command_service.login(message.from_user.id, parts[1]))


@router.message(Command("logout"))
async def cmd_logout(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    await message.answer(command_service.logout(message.from_user.id))


@router.message(Command("debug_on"))
async def cmd_debug_on(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    debug_service.set_debug(message.from_user.id, True)
    await message.answer("Debug режим включен.")


@router.message(Command("debug_off"))
async def cmd_debug_off(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    debug_service.set_debug(message.from_user.id, False)
    await message.answer("Debug режим выключен.")


@router.message(Command("debug_status"))
async def cmd_debug_status(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    await message.answer(f"Debug режим {debug_service.status_text(message.from_user.id)}.")


@router.message(Command("menu"))
async def cmd_menu(message: Message, state: FSMContext) -> None:
    if not await _require_profile(message, state):
        return
    await message.answer("Меню", reply_markup=_menu_keyboard())


@router.message()
async def message_handler(message: Message, state: FSMContext) -> None:
    if not await _ensure_profile(message, state):
        return
    await _handle_ai(message, message.text or "")


@router.message(RegistrationStates.display_name)
async def reg_display_name(message: Message, state: FSMContext) -> None:
    await state.update_data(display_name=message.text.strip())
    await state.set_state(RegistrationStates.calendar_email)
    await message.answer(
        "Укажите e-mail на Google, куда присылать приглашения в календарь (например, ivanov@gmail.com).",
    )


@router.message(RegistrationStates.calendar_email)
async def reg_calendar_email(message: Message, state: FSMContext) -> None:
    if not _is_valid_email(message.text or ""):
        await message.answer("Похоже, это не e-mail. Укажите адрес вида example@gmail.com")
        return
    email = message.text.strip()
    await state.update_data(calendar_email=email, email=email)
    await state.set_state(RegistrationStates.timezone)
    await message.answer(
        "Выберите часовой пояс (например, Europe/Berlin). Если не уверены — можно оставить по умолчанию (UTC)."
    )


@router.message(RegistrationStates.timezone)
async def reg_timezone(message: Message, state: FSMContext) -> None:
    timezone = message.text.strip() or "UTC"
    await state.update_data(timezone=timezone)
    await state.set_state(RegistrationStates.notify_calendar)
    await message.answer("Получать уведомления в календарь? (да/нет)", reply_markup=_yes_no_keyboard())


@router.message(RegistrationStates.notify_calendar)
async def reg_notify_calendar(message: Message, state: FSMContext) -> None:
    flag = _parse_yes_no(message.text)
    if flag is None:
        await message.answer("Ответьте да или нет", reply_markup=_yes_no_keyboard())
        return
    await state.update_data(notify_calendar=str(flag))
    await state.set_state(RegistrationStates.notify_telegram)
    await message.answer("Получать напоминания в Telegram? (да/нет)", reply_markup=_yes_no_keyboard())


@router.message(RegistrationStates.notify_telegram)
async def reg_notify_telegram(message: Message, state: FSMContext) -> None:
    flag = _parse_yes_no(message.text)
    if flag is None:
        await message.answer("Ответьте да или нет", reply_markup=_yes_no_keyboard())
        return
    await state.update_data(notify_telegram=str(flag))
    data = await state.get_data()
    profile = {
        "user_id": str(message.from_user.id),
        "telegram_user_id": str(message.from_user.id),
        "telegram_username": message.from_user.username or "",
        "telegram_full_name": message.from_user.full_name,
        "display_name": data.get("display_name", message.from_user.full_name),
        "email": data.get("calendar_email", ""),
        "calendar_email": data.get("calendar_email", data.get("email", "")),
        "timezone": data.get("timezone", "UTC"),
        "role": "",
        "notify_calendar": data.get("notify_calendar", "FALSE"),
        "notify_telegram": data.get("notify_telegram", "FALSE"),
        "created_at": datetime.utcnow().isoformat(),
        "last_seen_at": datetime.utcnow().isoformat(),
        "is_active": "TRUE",
    }
    google_service.create_user(profile)
    await state.clear()
    await message.answer(
        "Готово, профиль сохранён. Теперь можно создавать задачи и заметки.\n" + _profile_summary(profile),
        reply_markup=_menu_keyboard(),
    )


async def reminder_worker() -> None:
    google_service.ensure_structures()
    while True:
        await asyncio.sleep(300)
        for user in google_service.list_users():
            if str(user.get("notify_telegram", "")).lower() not in {"true", "1", "yes", "y"}:
                continue
            telegram_id = user.get("telegram_user_id")
            if not telegram_id:
                continue
            tasks = google_service.upcoming_tasks_for_user(user.get("user_id", ""))
            if not tasks:
                continue
            text = ai_service.build_reminder_text(tasks)
            try:
                await bot.send_message(int(telegram_id), text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send reminder: %s", exc)


async def main() -> None:
    asyncio.create_task(reminder_worker())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

"""Entry point for the Telegram bot."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties

import ai_service
import command_service
from config import CONFIG
import google_service
from dialog_history import append_message

LOG_PATH = Path(__file__).resolve().parent / "bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _safe_append_dialog(profile: dict, role: str, text: str) -> None:
    try:
        user_id = int(profile.get("telegram_user_id", 0))
    except (TypeError, ValueError):
        logger.debug("Invalid telegram_user_id for dialog append: %r", profile.get("telegram_user_id"))
        return
    try:
        append_message(user_id, role, text)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to append dialog message", exc_info=True)


class RegistrationStates(StatesGroup):
    display_name = State()
    email = State()
    timezone = State()


bot = Bot(token=CONFIG.telegram_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())


async def _start_registration(message: Message, state: FSMContext) -> None:
    await state.set_state(RegistrationStates.display_name)
    await message.answer("Как к вам обращаться? (Имя или имя + фамилия)")


async def _finalize_registration(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    now_iso = datetime.utcnow().isoformat()
    profile = {
        "user_id": str(message.from_user.id),
        "telegram_user_id": str(message.from_user.id),
        "telegram_username": message.from_user.username or "",
        "telegram_full_name": message.from_user.full_name,
        "display_name": data.get("display_name", message.from_user.full_name),
        "email": data.get("email", ""),
        "calendar_email": data.get("email", ""),
        "timezone": data.get("timezone", "Europe/Moscow"),
        "role": "",
        "notify_calendar": "TRUE",
        "notify_telegram": "TRUE",
        "created_at": now_iso,
        "last_seen_at": now_iso,
        "is_active": "TRUE",
    }
    google_service.create_or_update_user_profile(profile)
    await state.clear()
    await message.answer("Регистрация завершена, можно создавать задачи и заметки.")


async def _handle_registration(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state is None:
        await _start_registration(message, state)
        return
    if current_state == RegistrationStates.display_name.state:
        await state.update_data(display_name=message.text.strip())
        await state.set_state(RegistrationStates.email)
        await message.answer(
            "Укажите e-mail на Google, куда присылать приглашения в календарь (например, ivanov@gmail.com)."
        )
        return
    if current_state == RegistrationStates.email.state:
        await state.update_data(email=(message.text or "").strip())
        await state.set_state(RegistrationStates.timezone)
        await message.answer(
            "Укажите часовой пояс (например, Europe/Berlin). Если не уверены — ответьте 'по умолчанию'."
        )
        return
    if current_state == RegistrationStates.timezone.state:
        tz = (message.text or "").strip()
        if tz.lower() in {"по умолчанию", "default", ""}:
            tz = "Europe/Moscow"
        await state.update_data(timezone=tz)
        await _finalize_registration(message, state)
        return


@dp.message(Command("start"))
async def handle_start(message: Message, state: FSMContext) -> None:
    profile = google_service.get_user_profile(message.from_user.id)
    if profile:
        await message.answer("Вы уже зарегистрированы. Можно работать с задачами.")
        return
    await message.answer("Привет! Давайте настроим профиль для работы с задачами и календарём.")
    await _start_registration(message, state)


@dp.message(Command("help"))
async def handle_help(message: Message, state: FSMContext) -> None:
    profile = google_service.get_user_profile(message.from_user.id) or {
        "telegram_user_id": message.from_user.id,
        "display_name": message.from_user.full_name,
    }
    await state.clear()
    text = await command_service.execute_plan(profile, {"method": "show_help", "confidence": 1.0, "params": {}})
    await message.answer(text.user_visible_answer or "Подсказка недоступна сейчас.")


@dp.message(F.text)
async def handle_any_message(message: Message, state: FSMContext) -> None:
    google_service.ensure_structures()
    profile = google_service.get_user_profile(message.from_user.id)
    if not profile:
        await _handle_registration(message, state)
        return

    await state.clear()
    google_service.update_last_seen(message.from_user.id)
    user_text = message.text or ""
    try:
        plan = await ai_service.process_user_request(profile, user_text)
        result = await command_service.execute_plan(profile, plan)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process incoming message: %s", exc)
        fallback_text = "Произошла ошибка при обработке запроса. Я продолжаю работу."
        try:
            await message.answer(fallback_text)
        except Exception as send_exc:  # noqa: BLE001
            logger.warning("Failed to send fallback reply: %s", send_exc)
        _safe_append_dialog(profile, "user", user_text)
        _safe_append_dialog(profile, "assistant", fallback_text)
        return

    reply_text = result.user_visible_answer or "Запрос обработан."
    try:
        await message.answer(reply_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to send reply: %s", exc)

    _safe_append_dialog(profile, "user", user_text)
    _safe_append_dialog(profile, "assistant", reply_text)


async def reminder_worker() -> None:
    google_service.ensure_structures()
    while True:
        try:
            users = google_service.list_users()
            for user in users:
                try:
                    if str(user.get("notify_telegram", "")).lower() not in {"true", "1", "yes", "y"}:
                        continue

                    raw_chat_id = user.get("telegram_chat_id") or user.get("telegram_user_id")
                    try:
                        chat_id = int(raw_chat_id)
                    except (TypeError, ValueError):
                        logger.warning("Пропускаю напоминание: некорректный chat_id=%r", raw_chat_id)
                        continue

                    tasks = google_service.upcoming_tasks_for_user(user.get("user_id", ""))
                    if not tasks:
                        continue

                    text = await ai_service.build_reminder_text(tasks, user)
                    try:
                        await bot.send_message(chat_id=chat_id, text=text)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to send reminder for user %r: %s", user, exc)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to send reminder for user %r: %s", user, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error in reminder_worker: %s", exc)

        await asyncio.sleep(CONFIG.reminder_interval_seconds)


def handle_unhandled_exception(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    logger.error("Unhandled exception: %s", context)


async def main() -> None:
    google_service.ensure_structures()
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(handle_unhandled_exception)
    asyncio.create_task(reminder_worker())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

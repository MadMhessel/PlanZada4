"""Entry point for Telegram AI Secretary bot."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
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

router = Router()
bot = Bot(CONFIG.telegram_token, parse_mode=ParseMode.HTML)
dp = Dispatcher()
dp.include_router(router)


def _menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Личные задачи"), KeyboardButton(text="Командные задачи")],
            [KeyboardButton(text="Заметки"), KeyboardButton(text="Настройки")],
        ],
        resize_keyboard=True,
    )


async def _handle_ai(message: Message, text: str) -> None:
    cls, action = ai_service.analyze_request(text)
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


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я AI-секретарь. Умею управлять личными и командными задачами, заметками и календарём."
        " Установите пароль для шифрования командой /login <пароль>.",
        reply_markup=_menu_keyboard(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Примеры запросов:\n"
        "- напомни завтра позвонить Ивану в 18:00\n"
        "- создай заметку про бюджет проекта\n"
        "- создай задачу для команды по дизайну до пятницы\n"
        "- покажи мои задачи со статусом todo"
    )


@router.message(Command("login"))
async def cmd_login(message: Message) -> None:
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Укажите пароль: /login <пароль>")
        return
    await message.answer(command_service.login(message.from_user.id, parts[1]))


@router.message(Command("logout"))
async def cmd_logout(message: Message) -> None:
    await message.answer(command_service.logout(message.from_user.id))


@router.message(Command("debug_on"))
async def cmd_debug_on(message: Message) -> None:
    debug_service.set_debug(message.from_user.id, True)
    await message.answer("Debug режим включен.")


@router.message(Command("debug_off"))
async def cmd_debug_off(message: Message) -> None:
    debug_service.set_debug(message.from_user.id, False)
    await message.answer("Debug режим выключен.")


@router.message(Command("debug_status"))
async def cmd_debug_status(message: Message) -> None:
    await message.answer(f"Debug режим {debug_service.status_text(message.from_user.id)}.")


@router.message(Command("menu"))
async def cmd_menu(message: Message) -> None:
    await message.answer("Меню", reply_markup=_menu_keyboard())


@router.message()
async def message_handler(message: Message) -> None:
    await _handle_ai(message, message.text or "")


async def reminder_worker() -> None:
    while True:
        await asyncio.sleep(300)
        for user_id in list(command_service.user_sessions.keys()):
            tasks = google_service.upcoming_tasks_for_user(user_id)
            if not tasks:
                continue
            text = ai_service.build_reminder_text(tasks)
            try:
                await bot.send_message(user_id, text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send reminder: %s", exc)


async def main() -> None:
    asyncio.create_task(reminder_worker())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

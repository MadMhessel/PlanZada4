# Отчёт по полному анализу кода

## 1. Краткое резюме
- [ ] Общее состояние кода: требуется внимание к блокирующим операциям и обработке ошибок.
- [ ] Наиболее рискованные места:
  1. Синхронные обращения к Google API внутри асинхронных хэндлеров блокируют event loop и могут уронить отклик бота.
  2. Алгоритм ретраев для Google API фактически делает только одну повторную попытку, повышая вероятность сбоев.
  3. Обработка пользовательского ввода без нормализации/проверок в нескольких местах может приводить к ошибкам типов.

## 2. Критические проблемы (требуют исправления в первую очередь)

### 2.1. Блокировка event loop из-за синхронных вызовов Google API *(исправлено)*
**Файл:** main.py  
**Строки (примерно):** 133-206

**Описание:**  
В асинхронных хэндлерах (`handle_any_message`, `reminder_worker`, `main`) выполняются синхронные функции `google_service.ensure_structures`, `get_user_profile`, `update_last_seen`, `list_users`, `upcoming_tasks_for_user`. Эти функции читают/пишут Google Sheets и вызывают HTTP-запросы через Google API-клиент. В текущем виде они выполняются прямо в event loop aiogram, что блокирует обработку других апдейтов Telegram на время сетевых вызовов. При задержках сети или ответов Google бот перестанет реагировать, а фоновые напоминания будут конкурировать с основным поллингом за цикл событий.

**Фрагмент кода (до):**
```python
@dp.message(F.text)
async def handle_any_message(message: Message, state: FSMContext) -> None:
    google_service.ensure_structures()
    profile = google_service.get_user_profile(message.from_user.id)
    ...
    google_service.update_last_seen(message.from_user.id)
```

**Исправление (после):**
```python
@dp.message(F.text)
async def handle_any_message(message: Message, state: FSMContext) -> None:
    await asyncio.to_thread(google_service.ensure_structures)
    profile = await asyncio.to_thread(google_service.get_user_profile, message.from_user.id)
    ...
    await asyncio.to_thread(google_service.update_last_seen, message.from_user.id)
```
Аналогично в `reminder_worker` и `main` вынести обращения к Google API в `asyncio.to_thread` или использовать полноценный aiohttp-клиент с async-обёрткой.

**Комментарий:**
Перенос сетевых вызовов в отдельный поток снимает блокировку event loop и позволяет боту продолжать принимать и отправлять сообщения, даже если Google API отвечает медленно. Это повышает отзывчивость и устойчивость к временным сетевым проблемам.

---

### 2.2. Неполные ретраи при обращении к Google API *(исправлено)*
**Файл:** google_service.py  
**Строки (примерно):** 114-128

**Описание:**  
Хелпер `_with_retries` объявлен с циклом `for attempt in range(3)`, но внутри при любой ошибке HTTP или сети срабатывает `raise`, начиная уже со второй итерации (`if attempt >= 1: raise`). В результате фактически выполняется всего одна дополнительная попытка вместо заявленных трёх. При кратковременных сетевых сбоях (частых для Google API) это увеличивает вероятность падений и преждевременного выхода из операций, таких как создание листов или чтение таблиц.

**Фрагмент кода (до):**
```python
def _with_retries(func: Callable, *args, **kwargs):
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except HttpError as exc:
            ...
            if attempt >= 1:
                raise
            time.sleep(1 + attempt)
        except OSError as exc:
            ...
            if attempt >= 1:
                raise
            time.sleep(1 + attempt)
```

**Исправление (после):**
```python
def _with_retries(func: Callable, *args, **kwargs):
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except (HttpError, OSError) as exc:
            logger.warning("Google API error on attempt %s: %s", attempt + 1, exc)
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
```

**Комментарий:**
Чёткое условие выхода после третьей попытки обеспечивает заявленное количество ретраев и уменьшает вероятность сбоев из-за временных ошибок сети. Одновременная обработка HTTP и сетевых исключений упрощает логику.

---

## 3. Важные проблемы (средний приоритет)

### 3.1. Риск ошибок при отсутствии/некорректности пользовательских параметров *(исправлено)*
**Файл:** google_service.py  
**Строки (примерно):** 420-447

**Описание:**  
Функции поиска и чтения заметок (`search_personal_notes`, `read_personal_notes`) предполагают, что параметры `query` и `limit` валидны. При вызове без `query` или с нестроковым значением происходит `AttributeError` из-за `query.lower()`, что приведёт к падению выполнения плана бота. 

**Фрагмент кода (до):**
```python
def search_personal_notes(profile: dict, query: str, limit: int = 5, **_: str) -> str:
    ensure_structures()
    notes = _read_values(PERSONAL_NOTES_SHEET)
    filtered = [n for n in notes if n and n[1] == str(profile.get("user_id")) and query.lower() in (n[2].lower())]
```

**Исправление (после):**
```python
def search_personal_notes(profile: dict, query: Optional[str] = None, limit: int = 5, **_: str) -> str:
    ensure_structures()
    query_text = (query or "").strip().lower()
    notes = _read_values(PERSONAL_NOTES_SHEET)
    filtered = [n for n in notes if n and n[1] == str(profile.get("user_id")) and query_text in (n[2].lower())]
```

**Комментарий:**
Нормализация входа и значения по умолчанию предотвращают падение при пустом вводе и улучшают устойчивость бота к неполным планам от модели.

---

## 4. Минорные проблемы и улучшения

4.1. **Логирование чувствительных данных.** В некоторых сообщениях логгеру передаются целые профили пользователей (например, при ошибках напоминаний в `reminder_worker`), что может включать email/ID. Рекомендуется логировать только идентификаторы или хэши, чтобы снизить риск утечек. *(минимизировано в текущей итерации)*
4.2. **Планирование задач без таймаутов.** Вызовы Google API выполняются без явных таймаутов; при зависании сети поток будет ждать по умолчанию. Стоит задать таймауты на уровне HTTP-клиента или обёртки.  
4.3. **Кеш пользователей не инвалидируется глобально.** `_users_cache` заполняется один раз и обновляется только при create/update профиля. Если изменения в таблице происходят вручную, бот их не увидит. Добавьте TTL или флаг сброса.

## 5. Рекомендации по дальнейшему развитию

- Добавить async-обёртки или использовать aiohttp для работы с Google API и CalDAV, чтобы исключить блокировки event loop. 
- Ввести линтеры/форматтеры (ruff/black, mypy) для упрощения поддержки и раннего выявления ошибок типов. 
- Покрыть критические функции (обработка команд, интеграции с Google) интеграционными тестами с заглушками сервисов. 
- Внедрить централизованную валидацию входа (pydantic-схемы) для параметров, получаемых из модели и сообщений пользователя.

---

## 6. Ограничения для Codex

1. Не переписывать проект с нуля. Действовать точечно.  
2. Сохранять существующее поведение, кроме явных ошибок.  
3. Крупные рефакторинги оформлять как рекомендации, не выполнять автоматом.

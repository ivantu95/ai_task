"""
Модуль интеграции с AI (Groq API).
Обрабатывает сообщения на естественном языке (текст и голос), извлекая
структурированные данные (намерение, задачу, дату/время, приоритет),
а также генерирует ответы для RAG-поиска и еженедельной аналитики.
"""
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from groq import Groq

_client = None
_active_model = None

TIMEZONE_NAME = os.environ.get("TIMEZONE", "Europe/Moscow")
try:
    BOT_TZ = ZoneInfo(TIMEZONE_NAME)
except Exception:
    BOT_TZ = ZoneInfo("UTC")


def _get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please set it in your .env file.")
        _client = Groq(api_key=api_key)
    return _client


def _get_chat_model() -> str:
    """Определяет доступную модель чата: либо из GROQ_MODEL,
    либо автоматически выбирает лучшую доступную текстовую модель из аккаунта Groq."""
    global _active_model
    if _active_model:
        return _active_model

    custom_model = os.environ.get("GROQ_MODEL", "").strip()
    if custom_model:
        _active_model = custom_model
        return _active_model

    preferred_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "qwen-2.5-32b",
    ]

    try:
        client = _get_groq_client()
        models_data = client.models.list().data
        available_ids = {m.id for m in models_data}

        for model_id in preferred_models:
            if model_id in available_ids:
                _active_model = model_id
                print(f"[Groq] Выбрана модель: {_active_model}")
                return _active_model

        # Если ни одна из списка не подошла, берем любую текстовую (не Whisper/Guard)
        for m in models_data:
            if "whisper" not in m.id and "guard" not in m.id:
                _active_model = m.id
                print(f"[Groq] Выбрана доступная модель из списка: {_active_model}")
                return _active_model
    except Exception as e:
        print(f"[Groq] Предупреждение: не удалось получить список моделей: {e}")

    _active_model = "llama-3.3-70b-versatile"
    return _active_model


SYSTEM_PROMPT = """Ты анализируешь входящее сообщение пользователя личному боту-ассистенту для задач и заметок.
Текущие дата и время: {now}

Ответь ТОЛЬКО валидным JSON, без какого-либо дополнительного текста, без markdown-разметки (без ```json), строго в следующем формате:
{{"intent": "add_task" | "show_today" | "show_list" | "show_bydate" | "show_shopping" | "show_calendar" |
  "mark_done" | "mark_in_progress" | "snooze" | "show_recurring" | "stop_recurring" | "show_insights" |
  "search_task" | "search_history",
  "task": "краткое описание задачи или заметки",
  "due_time": "YYYY-MM-DD HH:MM" или null,
  "end_time": "YYYY-MM-DD HH:MM" или null,
  "target_date": "YYYY-MM-DD" или null,
  "category": "work" | "personal" | "shopping" | "ideas" | "general",
  "recurring": true или false, "recurring_time": "HH:MM" или null, "exclude_days": [],
  "priority": "high" или "normal", "task_number": integer или null,
  "snooze_minutes": integer или null, "search_query": "поисковый запрос" или null}}

Правила определения намерения (intent) — определи в ПЕРВУЮ очередь:
- "show_calendar": запрос на просмотр календаря или расписания (например: "покажи календарь", "календарь", "расписание", "планы на четверг", "что в календаре на завтра", "календарь на неделю").
  Если в запросе названа конкретная дата или день недели ("на завтра", "на 29 октября", "на пятницу"), вычисли точную дату и запиши в target_date (YYYY-MM-DD). Если дата не названа, поставь null.
- "show_shopping": запрос списка покупок (например: "что купить", "список покупок", "покажи покупки", "продукты", "что надо купить").
- "show_today": запрос на просмотр задач на сегодня (например: "что у меня на сегодня", "покажи дела на сегодня", "планы на день", "сегодняшние задачи").
- "show_list": запрос на просмотр ВСЕХ активных задач и заметок (например: "покажи все задачи", "список дел", "все заметки", "что у меня запланировано").
- "show_bydate": запрос на просмотр задач, сгруппированных по датам (например: "покажи задачи по датам", "дела по дням", "расписание по числам").
- "mark_done": сообщение о выполнении задачи:
  * Если назван номер (например: "выполнил задачу 3", "закрой 2", "сделал 1", "отметь третью"), извлеки его в task_number.
  * Если номер НЕ назван, но названа суть выполненного действия (например: "я купил молоко", "сдал отчет", "встреча завершилась", "закрыл созвон"), извлеки суть в поле task (например: "купить молоко" или "сдать отчет"), а task_number оставь null.
- "mark_in_progress": сообщение о том, что задача сейчас выполняется, в процессе или взята в работу (например: "я в процессе", "в процессе отчет", "сейчас делаю", "взял в работу", "делаю задачу 1", "в процессе созвон").
  * Если назван номер, извлеки в task_number.
  * Если номер не назван, извлеки название/суть задачи в task.
- "snooze": просьба отложить/перенести последнее напоминание (например: "отложи на 10 минут", "напомни через полчаса", "не сейчас, напомни позже", "отложи").
  Извлеки задержку в минутах в snooze_minutes (часы переведи в минуты). Если интервал не указан, поставь 10.
- "show_recurring": запрос списка регулярных ежедневных напоминаний/привычек (например: "покажи регулярные напоминания", "мои привычки", "ежедневные дела").
- "stop_recurring": отмена регулярного напоминания по номеру (например: "отмени регулярное 2", "останови привычку 1", "удали ежедневное напоминание 3").
  Извлеки номер в task_number.
- "show_insights": запрос сводки или статистики продуктивности (например: "как у меня дела на этой неделе", "покажи статистику", "отчет за неделю", "инсайты").
- "search_task": поиск конкретной ТЕКУЩЕЙ активной задачи или заметки (например: "найди задачу про магазин", "что у меня записано про отчет", "где заметка о паспорте").
  Извлеки поисковую фразу в search_query.
- "search_history": вопрос о ПРОШЛЫХ или завершенных делах (например: "что я делал на прошлой неделе по финансам", "делал ли я что-то по проекту", "история тренировок").
  Это вопрос о том, что уже выполнено или сработало ранее. Извлеки суть вопроса в search_query.
- "add_task": ВСЁ ОСТАЛЬНОЕ — любое добавление новой задачи, напоминания или заметки (например: "купить молоко", "напомни сдать отчет завтра в 18:00").
  Даже если упоминается "сегодня" (например: "напомни отправить письмо сегодня в 20:00"), это add_task, а не show_today, так как создается действие.

Правила для категории (category):
- "shopping": любые покупки, продукты, еда, бытовые товары, одежда, заказ товаров (например: "купить хлеб и молоко", "заказать корм коту", "купить лампочки").
- "work": задачи по работе, бизнесу, клиентам, проектам, отчетам, звонкам, коду.
- "ideas": идеи, мысли, заметки на будущее, почитать книгу, посмотреть фильм.
- "personal": здоровье, семья, дом, личные дела, тренировки, врачи.
- "general": если трудно отнести к конкретной категории.
Если в сообщении прямо указан хештег (#покупки, #работа, #идеи, #личное), строго используй соответствующую категорию.

Для намерений, отличных от "add_task", оставляй неиспользуемые поля по умолчанию (null, false, []).

Правила для РАЗОВЫХ задач и заметок (intent = add_task, recurring = false):
- Относительные даты ("завтра", "послезавтра", "в эту пятницу", "сегодня вечером") вычисляй строго от текущей даты выше.
- Для фраз вроде "через N минут" или "через N часов", вычисли ТОЧНОЕ время: прибавь N минут/часов к текущему времени. Выполняй вычисления точно.
- В поле task пиши чистую суть задачи, убирая слова вежливости и команды ("напомни мне", "запиши", "пожалуйста", "заметка").
- ЗАДАЧИ БЕЗ ТОЧНОГО ЧАСА, НО С УКАЗАНИЕМ НА СЕГОДНЯ («в течение дня», «в течении дня», «за сегодня», «сегодня днем», «сегодня»):
  * Такие задачи ОБЯЗАТЕЛЬНО должны получить дату на сегодня и контрольное время напоминания в due_time (а end_time = null):
    - Если текущее время до 14:00 ➔ установи due_time на 15:00 ("YYYY-MM-DD 15:00").
    - Если текущее время от 14:00 до 18:00 ➔ установи due_time на 18:00 ("YYYY-MM-DD 18:00").
    - Если текущее время после 18:00 ➔ установи due_time на 20:00 (или через 1 час, но не позже 21:00).
  * Если указано «завтра в течение дня» / «в течение завтрашнего дня» ➔ дата завтрашняя, due_time на 15:00 завтра, end_time = null.
  * Для общих указаний времени суток: «утром» ➔ 10:00, «днем» / «в обед» ➔ 14:00, «вечером» ➔ 19:00.
- ДИАПАЗОНЫ И ИНТЕРВАЛЫ ВРЕМЕНИ:
  Если указан интервал (например: "с 14:00 до 16:30", "завтра с 10 до 11:30", "18:00 - 19:30", "встреча 15:00-16:00"):
  * due_time — точные дата и время НАЧАЛА интервала ("YYYY-MM-DD HH:MM").
  * end_time — точные дата и время ОКОНЧАНИЯ интервала ("YYYY-MM-DD HH:MM").
  Если указано только одно время ("в 14:00") или нет времени, end_time = null.
- Слово «заметка» не отменяет время! Если в заметке есть указание «в течение дня», «сегодня», «завтра» или точное время — извлеки due_time.
- ТОЛЬКО если дата и время ВООБЩЕ не указаны (чистая бессрочная мысль, например: "купить хлеб", "идея: записать подкаст"), установи due_time в null и end_time в null.

Правила для РЕГУЛЯРНЫХ задач (intent = add_task, recurring = true):
- Применяется, если указано повторять регулярно / каждый день (например: "напоминай каждый день в 09:00 пить воду", "ежедневно бег в 07:30").
- Установи due_time в null, а recurring_time — время в 24-часовом формате "HH:MM".
- exclude_days: список 3-буквенных кодов дней недели на английском (mon, tue, wed, thu, fri, sat, sun), которые нужно ПРОПУСТИТЬ. Если исключений нет — [].
  Примеры: "кроме воскресенья" -> ["sun"], "по будням" (пропустить выходные) -> ["sat", "sun"], "кроме субботы и воскресенья" -> ["sat", "sun"].

Правила для приоритета (только для add_task):
- priority = "high", если в сообщении есть слова: "срочно", "важно", "urgent", "asap", "критично", "высокий приоритет". В остальных случаях "normal".
"""


def parse_task(message: str) -> dict:
    now = datetime.now(BOT_TZ).strftime("%Y-%m-%d %H:%M, %A")
    fallback = {
        "intent": "add_task",
        "task": message,
        "due_time": None,
        "end_time": None,
        "target_date": None,
        "category": "general",
        "recurring": False,
        "recurring_time": None,
        "exclude_days": [],
        "priority": "normal",
        "task_number": None,
        "snooze_minutes": None,
        "search_query": None,
    }

    try:
        model = _get_chat_model()
        response = _get_groq_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(now=now)},
                {"role": "user", "content": message},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed.setdefault("intent", "add_task")
            parsed.setdefault("task", message)
            parsed.setdefault("due_time", None)
            parsed.setdefault("end_time", None)
            parsed.setdefault("target_date", None)
            parsed.setdefault("category", "general")
            parsed.setdefault("recurring", False)
            parsed.setdefault("recurring_time", None)
            parsed.setdefault("exclude_days", [])
            parsed.setdefault("priority", "normal")
            parsed.setdefault("task_number", None)
            parsed.setdefault("snooze_minutes", None)
            parsed.setdefault("search_query", None)

            # Программная страховка для задач "в течение дня" / "сегодня"
            if parsed.get("intent") == "add_task" and not parsed.get("recurring") and not parsed.get("due_time"):
                msg_lower = message.lower()
                day_triggers = ("в течение дня", "в течении дня", "за сегодня", "сегодня днем", "в течение сегодняшнего", "в течении сегодняшнего")
                if any(trig in msg_lower for trig in day_triggers):
                    now_dt = datetime.now(BOT_TZ)
                    if "завтра" in msg_lower:
                        target_dt = now_dt + timedelta(days=1)
                        parsed["due_time"] = target_dt.strftime("%Y-%m-%d 15:00")
                    else:
                        if now_dt.hour < 14:
                            remind_time = "15:00"
                        elif now_dt.hour < 18:
                            remind_time = "18:00"
                        elif now_dt.hour < 20:
                            remind_time = "20:00"
                        else:
                            remind_time = (now_dt + timedelta(hours=1)).strftime("%H:%M") if now_dt.hour < 23 else "23:50"
                        parsed["due_time"] = f"{now_dt.strftime('%Y-%m-%d')} {remind_time}"

            return parsed
    except json.JSONDecodeError:
        return fallback
    except Exception as e:
        print(f"[Groq] Ошибка в parse_task ({type(e).__name__}): {e}")
        return fallback

    return fallback


def transcribe_voice(file_path: str) -> str:
    """Отправляет голосовое сообщение (файл .ogg) в модель Whisper от Groq
    и возвращает распознанный текст."""
    with open(file_path, "rb") as audio_file:
        response = _get_groq_client().audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
        )
    return response.text


def generate_history_answer(query: str, matches: list[dict]) -> str:
    """Генерация ответа RAG: преобразует найденные в истории задачи в связный
    ответ на естественном русском языке."""
    context_lines = []
    for m in matches:
        status = "выполнено" if m.get("status") == "done" else "сработало напоминание"
        when = m.get("completed_at") or m.get("created_at")
        context_lines.append(f"- {m['text']} [{status}, {when}]")
    context_text = "\n".join(context_lines)

    prompt = f"""Пользователь задал вопрос о своих прошлых задачах и делах: "{query}"

Вот список релевантных завершенных задач, найденных в его истории:
{context_text}

Напиши краткий, естественный ответ на русском языке (2-4 предложения), суммирующий найденную информацию.
Упоминай конкретные задачи и даты, если они есть.
Если найденные совпадения не отвечают на вопрос прямо, честно скажи об этом, не выдумывая факты."""

    try:
        model = _get_chat_model()
        response = _get_groq_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq] Ошибка в generate_history_answer ({type(e).__name__}): {e}")
        return f"Найдено в истории:\n{context_text}"


def generate_weekly_insight(stats: dict) -> str:
    """Преобразует числовую статистику за неделю в понятный, мотивирующий отчет на русском языке."""
    prompt = f"""Вот данные пользователя по выполнению задач за эту неделю по сравнению с прошлой:
- Разовых задач создано на этой неделе: {stats['created_this_week']}
- Разовых задач завершено на этой неделе: {stats['done_this_week']}
- Процент выполнения на этой неделе: {stats['completion_rate']}%
- Процент выполнения на прошлой неделе: {stats['prev_completion_rate']}%
- Задач с высоким приоритетом создано: {stats['high_priority_created']}
- Задач с высоким приоритетом завершено: {stats['high_priority_done']}
- Регулярных привычек/напоминаний сработало: {stats['recurring_fired_this_week']}
- Регулярных привычек выполнено: {stats['recurring_completed_this_week']}
- Процент выполнения регулярных привычек: {stats['recurring_completion_rate']}%

Напиши краткую, теплую и мотивирующую сводку на русском языке (3-5 предложений, дружелюбный тон).
Оцени отдельно выполнение разовых задач и регулярность привычек.
Сравни текущую неделю с прошлой.
Если показатели невысокие — не критикуй, а поддержи и вдохнови на следующую неделю."""

    try:
        model = _get_chat_model()
        response = _get_groq_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq] Ошибка в generate_weekly_insight ({type(e).__name__}): {e}")
        return (
            f"За неделю создано задач: {stats['created_this_week']}, "
            f"выполнено: {stats['done_this_week']} ({stats['completion_rate']}%). "
            f"Регулярных привычек: {stats['recurring_completed_this_week']} из {stats['recurring_fired_this_week']}."
        )


SMART_NOTE_PROMPT = """Ты — персональный AI-редактор заметок (в стиле AudioPen / Oasis).
Пользователь надиктовал свои мысли, идеи или поток сознания в голосовом сообщении.
Текущие дата и время: {now}

Твоя задача:
1. Очистить текст от слов-паразитов ("эээ", "ну", "короче", "в общем", "как бы", повторов, оговорок, сбивчивости).
2. Придумать емкий и понятный заголовок (title).
3. Сформулировать главную суть (summary) в 1-2 предложениях.
4. Выделить ключевые мысли списком буллетов (key_points).
5. Определить основную категорию (category): "work", "personal", "shopping", "ideas", "general".
6. Извлечь любые конкретные задачи или напоминания с дедлайнами (action_items), упомянутые в речи.
   Для каждого пункта action_items:
   - task: четкая формулировка действия без лишних слов
   - due_time: точный дедлайн или контрольное время напоминания (YYYY-MM-DD HH:MM), вычисленный строго от текущей даты выше:
     * Если названо точное время (например "в 14:00", "завтра в 11:30") — используй его.
     * Если сказано "в течение дня", "в течении дня", "сегодня", "за сегодня":
       - если текущее время до 14:00: поставь 15:00 на сегодня
       - если текущее время от 14:00 до 18:00: поставь 18:00 на сегодня
       - если текущее время после 18:00: поставь 20:00 на сегодня
     * Если сказано "утром" — 10:00, "днем/в обед" — 14:00, "вечером" — 19:00.
     * Только если даты и времени совсем нет (чистая мысль/идея) — установи null.
   - end_time: время окончания интервала (YYYY-MM-DD HH:MM), если указан диапазон (например "с 15:00 до 16:30"). Для задач "в течение дня" ставь null!
   - priority: "high" (если срочно/важно) или "normal"
   - category: "work" | "personal" | "shopping" | "ideas" | "general"

Ответь ТОЛЬКО валидным JSON, без какого-либо другого текста, без markdown-разметки (без ```json), строго в формате:
{{
  "title": "Емкий заголовок",
  "summary": "Краткая суть мысли",
  "key_points": ["Ключевой пункт 1", "Ключевой пункт 2"],
  "category": "work" | "personal" | "shopping" | "ideas" | "general",
  "action_items": [
    {{"task": "Суть конкретного действия", "due_time": "YYYY-MM-DD HH:MM" или null, "end_time": "YYYY-MM-DD HH:MM" или null, "priority": "high" или "normal", "category": "work" | "personal" | "shopping" | "ideas" | "general"}}
  ]
}}
"""


def structure_smart_note(text: str) -> dict:
    """Превращает сбивчивый текст голосового сообщения в красивый структурированный конспект
    с выделением сути, буллетов и задач с дедлайнами."""
    now = datetime.now(BOT_TZ).strftime("%Y-%m-%d %H:%M, %A")
    fallback = {
        "title": "Голосовая заметка",
        "summary": text,
        "key_points": [],
        "category": "general",
        "action_items": [],
    }

    try:
        model = _get_chat_model()
        response = _get_groq_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SMART_NOTE_PROMPT.format(now=now)},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed.setdefault("title", "Голосовая заметка")
            parsed.setdefault("summary", text)
            parsed.setdefault("key_points", [])
            parsed.setdefault("category", "general")
            parsed.setdefault("action_items", [])

            # Программная страховка для пунктов "в течение дня" / "сегодня"
            day_triggers = ("в течение дня", "в течении дня", "за сегодня", "сегодня днем", "в течение сегодняшнего", "в течении сегодняшнего")
            for item in parsed.get("action_items", []):
                if not item.get("due_time") and item.get("task"):
                    combined = f"{item['task']} {text}".lower()
                    if any(trig in combined for trig in day_triggers):
                        now_dt = datetime.now(BOT_TZ)
                        if "завтра" in combined:
                            target_dt = now_dt + timedelta(days=1)
                            item["due_time"] = target_dt.strftime("%Y-%m-%d 15:00")
                        else:
                            if now_dt.hour < 14:
                                remind_time = "15:00"
                            elif now_dt.hour < 18:
                                remind_time = "18:00"
                            elif now_dt.hour < 20:
                                remind_time = "20:00"
                            else:
                                remind_time = (now_dt + timedelta(hours=1)).strftime("%H:%M") if now_dt.hour < 23 else "23:50"
                            item["due_time"] = f"{now_dt.strftime('%Y-%m-%d')} {remind_time}"
                        item["end_time"] = None

            return parsed
    except Exception as e:
        print(f"[Groq] Ошибка в structure_smart_note ({type(e).__name__}): {e}")
        return fallback

    return fallback

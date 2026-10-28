"""
Основной модуль Telegram-бота.
Использует long polling (не требует webhook, белого IP или ngrok)
и JobQueue для надежной отправки запланированных напоминаний.
Поддерживает интерактивные кнопки, автопроверку выполнения (follow-up)
и учет просроченных задач.
"""
import os
import tempfile
from collections import defaultdict
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters,
)

from src.db import (
    init_db, add_task, list_pending_tasks, mark_done, register_user, list_all_chat_ids,
    add_recurring_task, list_recurring_tasks, delete_task, get_weekly_stats,
    log_recurring_fired, mark_recurring_done, get_recurring_weekly_stats,
    get_task_history, get_task_by_id, mark_task_reminded, update_task_due_time,
)
from src.llm_parser import parse_task, transcribe_voice, generate_weekly_insight, generate_history_answer
from src.semantic_search import semantic_search

WEEKDAY_NUMBERS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
RU_WEEKDAYS = {"mon": "пн", "tue": "вт", "wed": "ср", "thu": "чт", "fri": "пт", "sat": "сб", "sun": "вс"}

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REMINDER_TIME = os.environ.get("REMINDER_TIME", "08:00")
TIMEZONE_NAME = os.environ.get("TIMEZONE", "Europe/Moscow")
try:
    BOT_TZ = ZoneInfo(TIMEZONE_NAME)
except Exception:
    BOT_TZ = ZoneInfo("UTC")
IST = BOT_TZ  # алиас для обратной совместимости

_last_list_mapping: dict[int, dict[int, int]] = {}  # chat_id -> {номер в списке: id в базе данных}
_recurring_mapping: dict[int, dict[int, int]] = {}  # chat_id -> {номер в списке: id в базе данных}
_last_fired: dict[int, tuple[str, int]] = {}  # chat_id -> (текст, счетчик откладываний)
_last_fired_recurring: dict[int, int] = {}  # chat_id -> task_id последнего регулярного напоминания


def _parse_due_datetime(due_time_str: str) -> datetime | None:
    try:
        naive = datetime.strptime(due_time_str, "%Y-%m-%d %H:%M")
        return naive.replace(tzinfo=BOT_TZ)
    except (ValueError, TypeError):
        return None


def get_reminder_keyboard(task_id: int) -> InlineKeyboardMarkup:
    """Клавиатура с быстрыми действиями под сработавшим напоминанием."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Выполнено", callback_data=f"done:{task_id}"),
            InlineKeyboardButton("⏰ +15 мин", callback_data=f"snooze:{task_id}:15"),
            InlineKeyboardButton("⏰ +1 час", callback_data=f"snooze:{task_id}:60"),
        ],
        [
            InlineKeyboardButton("📅 На завтра", callback_data=f"snooze:{task_id}:tomorrow"),
            InlineKeyboardButton("🗑 Удалить", callback_data=f"del:{task_id}"),
        ],
    ])


def get_followup_keyboard(task_id: int) -> InlineKeyboardMarkup:
    """Клавиатура для контрольного переспроса."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да, сделано", callback_data=f"done:{task_id}"),
            InlineKeyboardButton("⏰ Напомнить через 30 мин", callback_data=f"snooze:{task_id}:30"),
        ],
        [
            InlineKeyboardButton("📅 Перенести на завтра", callback_data=f"snooze:{task_id}:tomorrow"),
        ],
    ])


def get_recurring_keyboard(task_id: int) -> InlineKeyboardMarkup:
    """Клавиатура для регулярной привычки."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Выполнено сегодня", callback_data=f"rec_done:{task_id}"),
            InlineKeyboardButton("⏰ Напомнить через 30 мин", callback_data=f"rec_snooze:{task_id}:30"),
        ]
    ])


async def send_task_reminder(context: ContextTypes.DEFAULT_TYPE):
    """Срабатывание разового напоминания.
    Задача помечается как 'reminded' (остается в списке активных до закрытия),
    отправляются кнопки и планируется контрольный переспрос."""
    chat_id = context.job.chat_id
    task_id, text, snooze_count = context.job.data
    mark_task_reminded(chat_id, task_id)
    _last_fired[chat_id] = (text, snooze_count)

    await context.bot.send_message(
        chat_id,
        f"⏰ <b>Напоминание:</b> {text}",
        parse_mode="HTML",
        reply_markup=get_reminder_keyboard(task_id),
    )

    # Планируем контрольный переспрос через 45 минут
    schedule_followup_check(context.job_queue, chat_id, task_id, text, delay_minutes=45)


def schedule_followup_check(job_queue, chat_id: int, task_id: int, text: str, delay_minutes: int = 45):
    """Планирует контрольный опрос, если задача не будет выполнена вовремя."""
    if not job_queue:
        return
    job_queue.run_once(
        send_followup_reminder,
        when=timedelta(minutes=delay_minutes),
        chat_id=chat_id,
        data=(task_id, text),
        name=f"followup_{chat_id}_{task_id}",
    )


async def send_followup_reminder(context: ContextTypes.DEFAULT_TYPE):
    """Контрольный переспрос: проверяет, была ли задача закрыта."""
    chat_id = context.job.chat_id
    task_id, text = context.job.data
    task = get_task_by_id(chat_id, task_id)
    if not task or task.get("status") == "done":
        return

    await context.bot.send_message(
        chat_id,
        f"🔔 <b>Контрольный вопрос:</b>\nЗадача «{text}» всё еще не отмечена выполненной.\nУдалось сделать?",
        parse_mode="HTML",
        reply_markup=get_followup_keyboard(task_id),
    )


def schedule_task_reminder(
    job_queue, chat_id: int, task_id: int, text: str, due_time_str: str | None, snooze_count: int = 0
):
    due_dt = _parse_due_datetime(due_time_str) if due_time_str else None
    if due_dt and due_dt > datetime.now(BOT_TZ):
        job_queue.run_once(
            send_task_reminder,
            when=due_dt,
            chat_id=chat_id,
            data=(task_id, text, snooze_count),
            name=f"task_{chat_id}_{task_id}",
        )


async def send_recurring_reminder(context: ContextTypes.DEFAULT_TYPE):
    """Срабатывание регулярного напоминания."""
    chat_id = context.job.chat_id
    task_id, text = context.job.data
    await context.bot.send_message(
        chat_id,
        f"🔁 <b>Регулярное дело:</b> {text}",
        parse_mode="HTML",
        reply_markup=get_recurring_keyboard(task_id),
    )
    _last_fired[chat_id] = (text, 0)
    _last_fired_recurring[chat_id] = task_id
    log_recurring_fired(chat_id, task_id)


def schedule_recurring_reminder(
    job_queue, chat_id: int, task_id: int, text: str, recurring_time_str: str, exclude_days: list[str]
):
    hour, minute = map(int, recurring_time_str.split(":"))
    all_days = {0, 1, 2, 3, 4, 5, 6}
    excluded = {WEEKDAY_NUMBERS[d] for d in exclude_days if d in WEEKDAY_NUMBERS}
    days = tuple(sorted(all_days - excluded))

    job_queue.run_daily(
        send_recurring_reminder,
        time=dtime(hour=hour, minute=minute, tzinfo=BOT_TZ),
        days=days,
        chat_id=chat_id,
        data=(task_id, text),
        name=f"recurring_{chat_id}_{task_id}",
    )


def _group_by_date(tasks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        date_part = t["due_time"].split(" ")[0] if t["due_time"] else "No due date"
        grouped[date_part].append(t)
    return grouped


def _format_task_due(t: dict) -> str:
    """Форматирует метку времени с подсветкой просроченных задач."""
    if not t.get("due_time"):
        return " (📝 заметка)"

    due_dt = _parse_due_datetime(t["due_time"])
    now_dt = datetime.now(BOT_TZ)
    if due_dt and due_dt < now_dt and t.get("status") != "done":
        return f" ⚠️ (просрочено: {t['due_time']})"
    return f" (⏰ {t['due_time']})"


def _build_list_reply(chat_id: int) -> str:
    tasks = list_pending_tasks(chat_id)
    if not tasks:
        return "У вас нет активных задач и заметок."
    _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(tasks)}
    lines = ["📋 Ваши активные задачи и заметки:"]
    for i, t in enumerate(tasks):
        mark = "🔴 " if t["priority"] == "high" else ""
        lines.append(f"{i + 1}. {mark}{t['text']}{_format_task_due(t)}")
    lines.append("\nЧтобы отметить выполненной: /done <номер>")
    return "\n".join(lines)


def _build_today_reply(chat_id: int) -> str:
    tasks = list_pending_tasks(chat_id)
    today_str = date.today().isoformat()
    now_dt = datetime.now(BOT_TZ)

    todays_tasks = [
        t for t in tasks
        if (t["due_time"] and t["due_time"].startswith(today_str))
        or (t.get("status") == "reminded")  # включаем сработавшие сегодня
    ]
    if not todays_tasks:
        return "На сегодня нет запланированных задач."

    _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(todays_tasks)}
    lines = ["📅 Задачи на сегодня:"]
    for i, t in enumerate(todays_tasks):
        mark = "🔴 " if t["priority"] == "high" else ""
        time_part = t["due_time"].split(" ")[1] if t.get("due_time") and " " in t["due_time"] else ""
        due_dt = _parse_due_datetime(t["due_time"]) if t.get("due_time") else None

        if due_dt and due_dt < now_dt:
            suffix = f" (⚠️ просрочено: {time_part})" if time_part else " ⚠️ (просрочено)"
        else:
            suffix = f" (в {time_part})" if time_part else ""

        lines.append(f"{i + 1}. {mark}{t['text']}{suffix}")
    lines.append("\nЧтобы отметить выполненной: /done <номер>")
    return "\n".join(lines)


def _build_search_reply(chat_id: int, query: str) -> str:
    if not query:
        return "Укажите, что нужно найти (например: /search продукты)."

    tasks = list_pending_tasks(chat_id)
    matches = semantic_search(query, tasks, text_key="text", top_k=5, threshold=0.3)
    if not matches:
        return f"По запросу «{query}» ничего не найдено среди активных задач."

    _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(matches)}
    lines = [f"🔍 Результаты поиска для «{query}» среди активных задач:"]
    for i, t in enumerate(matches):
        mark = "🔴 " if t["priority"] == "high" else ""
        lines.append(f"{i + 1}. {mark}{t['text']}{_format_task_due(t)}")
    lines.append("\nЧтобы отметить выполненной: /done <номер>")
    return "\n".join(lines)


def _build_history_reply(chat_id: int, query: str) -> str:
    if not query:
        return "Укажите, что найти в истории (например: /history финансы)."

    history = get_task_history(chat_id)
    if not history:
        return "В истории пока нет завершенных или архивных задач."

    matches = semantic_search(query, history, text_key="text", top_k=8, threshold=0.25)
    candidates_for_ai = matches if matches else history[:25]
    return generate_history_answer(query, candidates_for_ai)


async def search_history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    query = " ".join(context.args) if context.args else ""
    await update.message.reply_text(_build_history_reply(chat_id, query))


async def search_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    query = " ".join(context.args) if context.args else ""
    await update.message.reply_text(_build_search_reply(chat_id, query))


async def list_tasks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    await update.message.reply_text(_build_list_reply(chat_id))


def _build_bydate_reply(chat_id: int) -> str:
    tasks = list_pending_tasks(chat_id)
    if not tasks:
        return "У вас нет активных задач и заметок."

    grouped = _group_by_date(tasks)
    ordered_dates = sorted(grouped.keys(), key=lambda d: (d == "No due date", d))

    mapping = {}
    blocks = []
    counter = 1
    for date_str in ordered_dates:
        header = "📝 Без даты (заметки)" if date_str == "No due date" else f"📅 {date_str}"
        lines = [header]
        for t in grouped[date_str]:
            mapping[counter] = t["id"]
            mark = "🔴 " if t["priority"] == "high" else ""
            lines.append(f"  {counter}. {mark}{t['text']}{_format_task_due(t)}")
            counter += 1
        blocks.append("\n".join(lines))

    _last_list_mapping[chat_id] = mapping
    blocks.append("Чтобы отметить выполненной: /done <номер>")
    return "\n\n".join(blocks)


async def bydate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    await update.message.reply_text(_build_bydate_reply(chat_id))


async def today_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    await update.message.reply_text(_build_today_reply(chat_id))


async def done_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /done <номер из списка>")
        return

    shown_number = int(context.args[0])
    task_id = _last_list_mapping.get(chat_id, {}).get(shown_number)

    if task_id is None:
        await update.message.reply_text("Сначала посмотри номер задачи через /list или /today.")
        return

    if mark_done(chat_id, task_id):
        for job in context.job_queue.get_jobs_by_name(f"followup_{chat_id}_{task_id}"):
            job.schedule_removal()
        await update.message.reply_text(f"Задача #{shown_number} выполнена! ✅")
    else:
        await update.message.reply_text("Эта задача уже была завершена ранее.")


def _build_recurring_reply(chat_id: int) -> str:
    tasks = list_recurring_tasks(chat_id)
    if not tasks:
        return "Нет активных регулярных напоминаний."

    _recurring_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(tasks)}
    lines = ["🔁 Регулярные напоминания и привычки:"]
    for i, t in enumerate(tasks):
        if t["exclude_days"]:
            excluded = [RU_WEEKDAYS.get(d, d) for d in t["exclude_days"].split(",") if d]
            skip = f", кроме {', '.join(excluded)}"
        else:
            skip = ""
        lines.append(f"{i + 1}. {t['text']} — ежедневно в {t['recurring_time']}{skip}")
    lines.append("\nЧтобы остановить: /stoprecurring <номер>")
    return "\n".join(lines)


async def recurring_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    await update.message.reply_text(_build_recurring_reply(chat_id))


def _stop_recurring_by_number(chat_id: int, task_number: int | None, job_queue) -> str:
    if task_number is None:
        return "Не понял, какое напоминание остановить. Сначала проверь номер через /recurring."

    task_id = _recurring_mapping.get(chat_id, {}).get(task_number)
    if task_id is None:
        return "Напоминание с таким номером не найдено. Сначала проверь список через /recurring."

    for job in job_queue.get_jobs_by_name(f"recurring_{chat_id}_{task_id}"):
        job.schedule_removal()
    delete_task(chat_id, task_id)
    return f"Регулярное напоминание #{task_number} отключено."


async def stop_recurring_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /stoprecurring <номер из /recurring>")
        return

    shown_number = int(context.args[0])
    await update.message.reply_text(_stop_recurring_by_number(chat_id, shown_number, context.job_queue))


async def _process_parsed_task(chat_id: int, parsed: dict, job_queue) -> str:
    """Общая логика сохранения задач и заметок (для текста и голоса)."""
    if parsed.get("recurring"):
        task_id = add_recurring_task(chat_id, parsed["task"], parsed["recurring_time"], parsed["exclude_days"])
        schedule_recurring_reminder(
            job_queue, chat_id, task_id, parsed["task"], parsed["recurring_time"], parsed["exclude_days"]
        )
        if parsed["exclude_days"]:
            excluded = [RU_WEEKDAYS.get(d, d) for d in parsed["exclude_days"] if d]
            skip = f", кроме {', '.join(excluded)}"
        else:
            skip = ""
        return f"Регулярное напоминание создано: {parsed['task']} — ежедневно в {parsed['recurring_time']}{skip}"

    priority = parsed.get("priority", "normal")
    task_id = add_task(chat_id, parsed["task"], parsed.get("due_time"), priority, snooze_count=0)
    schedule_task_reminder(job_queue, chat_id, task_id, parsed["task"], parsed.get("due_time"), snooze_count=0)
    due = f" (⏰ срок: {parsed['due_time']})" if parsed.get("due_time") else " (📝 заметка)"
    mark = "🔴 " if priority == "high" else ""
    return f"Добавлено [#{task_id}]: {mark}{parsed['task']}{due}"


def _mark_done_by_number(chat_id: int, task_number: int | None) -> str:
    if task_number is None:
        recurring_task_id = _last_fired_recurring.get(chat_id)
        if recurring_task_id is not None:
            mark_recurring_done(chat_id, recurring_task_id)
            return "Отлично! Регулярная привычка отмечена выполненной за сегодня. 🔥"
        return "Не понял, какую задачу отметить. Открой /list и укажи номер (например: /done 1)."

    task_id = _last_list_mapping.get(chat_id, {}).get(task_number)
    if task_id is None:
        return "Сначала посмотри номер задачи через /list или /today, затем вызови /done <номер>."

    if mark_done(chat_id, task_id):
        return f"Задача #{task_number} выполнена! ✅"
    return "Эта задача уже была завершена ранее."


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Любое сообщение, не являющееся /командой, анализируется на намерение (intent)."""
    chat_id = update.effective_chat.id
    register_user(chat_id)
    text = update.message.text.strip()
    parsed = parse_task(text)

    intent = parsed.get("intent")

    if intent == "show_today":
        await update.message.reply_text(_build_today_reply(chat_id))
        return
    if intent == "show_list":
        await update.message.reply_text(_build_list_reply(chat_id))
        return
    if intent == "show_bydate":
        await update.message.reply_text(_build_bydate_reply(chat_id))
        return
    if intent == "mark_done":
        await update.message.reply_text(_mark_done_by_number(chat_id, parsed.get("task_number")))
        return
    if intent == "snooze":
        minutes = parsed.get("snooze_minutes") or 10
        for msg in await _snooze_last_fired(chat_id, minutes, context.job_queue):
            await update.message.reply_text(msg)
        return
    if intent == "show_recurring":
        await update.message.reply_text(_build_recurring_reply(chat_id))
        return
    if intent == "stop_recurring":
        await update.message.reply_text(
            _stop_recurring_by_number(chat_id, parsed.get("task_number"), context.job_queue)
        )
        return
    if intent == "show_insights":
        await update.message.reply_text(f"📊 Еженедельная сводка:\n{_build_insights_reply(chat_id)}")
        return
    if intent == "search_task":
        await update.message.reply_text(_build_search_reply(chat_id, parsed.get("search_query")))
        return
    if intent == "search_history":
        await update.message.reply_text(_build_history_reply(chat_id, parsed.get("search_query")))
        return

    reply = await _process_parsed_task(chat_id, parsed, context.job_queue)
    await update.message.reply_text(reply)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Скачивает голосовую заметку, транскрибирует через Groq Whisper
    и направляет результат в общий пайплайн распознавания намерений."""
    chat_id = update.effective_chat.id
    register_user(chat_id)

    voice_file = await context.bot.get_file(update.message.voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
    await voice_file.download_to_drive(tmp_path)

    try:
        text = transcribe_voice(tmp_path)
    finally:
        os.remove(tmp_path)

    parsed = parse_task(text)
    intent = parsed.get("intent")

    if intent == "show_today":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{_build_today_reply(chat_id)}")
        return
    if intent == "show_list":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{_build_list_reply(chat_id)}")
        return
    if intent == "show_bydate":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{_build_bydate_reply(chat_id)}")
        return
    if intent == "mark_done":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{_mark_done_by_number(chat_id, parsed.get('task_number'))}")
        return
    if intent == "snooze":
        minutes = parsed.get("snooze_minutes") or 10
        replies = await _snooze_last_fired(chat_id, minutes, context.job_queue)
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n" + "\n".join(replies))
        return
    if intent == "show_recurring":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{_build_recurring_reply(chat_id)}")
        return
    if intent == "stop_recurring":
        result = _stop_recurring_by_number(chat_id, parsed.get("task_number"), context.job_queue)
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{result}")
        return
    if intent == "show_insights":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n📊 Еженедельная сводка:\n{_build_insights_reply(chat_id)}")
        return
    if intent == "search_task":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{_build_search_reply(chat_id, parsed.get('search_query'))}")
        return
    if intent == "search_history":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{_build_history_reply(chat_id, parsed.get('search_query'))}")
        return

    reply = await _process_parsed_task(chat_id, parsed, context.job_queue)
    await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{reply}")


async def _snooze_last_fired(chat_id: int, minutes: int, job_queue) -> list[str]:
    """Логика откладывания последнего напоминания на N минут."""
    fired = _last_fired.get(chat_id)
    if fired is None:
        return ["Нет недавних сработавших напоминаний, которые можно отложить."]

    text, prev_snooze_count = fired
    new_snooze_count = prev_snooze_count + 1

    new_due_dt = datetime.now(BOT_TZ) + timedelta(minutes=minutes)
    new_due = new_due_dt.strftime("%Y-%m-%d %H:%M")

    tasks = list_pending_tasks(chat_id)
    matching = [t for t in tasks if t["text"] == text]
    if matching:
        task_id = matching[0]["id"]
        update_task_due_time(chat_id, task_id, new_due, new_snooze_count)
    else:
        task_id = add_task(chat_id, text, new_due, snooze_count=new_snooze_count)

    for job in job_queue.get_jobs_by_name(f"task_{chat_id}_{task_id}"):
        job.schedule_removal()
    for job in job_queue.get_jobs_by_name(f"followup_{chat_id}_{task_id}"):
        job.schedule_removal()

    schedule_task_reminder(job_queue, chat_id, task_id, text, new_due, snooze_count=new_snooze_count)
    _last_fired[chat_id] = (text, new_snooze_count)

    messages = [f"Хорошо, напомню через {minutes} мин.: {text}"]
    if new_snooze_count >= 3:
        messages.append(
            f"💡 Заметил, что ты откладываешь «{text}» уже {new_snooze_count} раз(а). "
            f"Возможно, лучше перенести задачу на определенное время, например, на завтра? "
            f"Просто напиши новое время, и я обновлю напоминание."
        )
    return messages


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий на инлайн-кнопки под напоминаниями."""
    query = update.callback_query
    await query.answer()

    chat_id = update.effective_chat.id
    data = query.data or ""
    parts = data.split(":")
    action = parts[0]

    if action == "done":
        task_id = int(parts[1])
        task = get_task_by_id(chat_id, task_id)
        task_text = task.get("text", "") if task else ""
        if mark_done(chat_id, task_id):
            for job in context.job_queue.get_jobs_by_name(f"followup_{chat_id}_{task_id}"):
                job.schedule_removal()
            await query.edit_message_text(f"✅ Выполнено: «{task_text}»\nОтличная работа!")
        else:
            await query.edit_message_text("Задача уже отмечена выполненной.")

    elif action == "snooze":
        task_id = int(parts[1])
        snooze_arg = parts[2]
        task = get_task_by_id(chat_id, task_id)
        if not task:
            await query.edit_message_text("Задача не найдена.")
            return

        text = task.get("text", "")
        prev_snooze_count = task.get("snooze_count", 0)
        new_snooze_count = prev_snooze_count + 1

        if snooze_arg == "tomorrow":
            new_due_dt = (datetime.now(BOT_TZ) + timedelta(days=1)).replace(hour=9, minute=0)
            time_label = "завтра в 09:00"
        else:
            mins = int(snooze_arg)
            new_due_dt = datetime.now(BOT_TZ) + timedelta(minutes=mins)
            time_label = f"через {mins} мин."

        new_due = new_due_dt.strftime("%Y-%m-%d %H:%M")
        update_task_due_time(chat_id, task_id, new_due, new_snooze_count)

        for job in context.job_queue.get_jobs_by_name(f"task_{chat_id}_{task_id}"):
            job.schedule_removal()
        for job in context.job_queue.get_jobs_by_name(f"followup_{chat_id}_{task_id}"):
            job.schedule_removal()

        schedule_task_reminder(context.job_queue, chat_id, task_id, text, new_due, snooze_count=new_snooze_count)
        _last_fired[chat_id] = (text, new_snooze_count)

        nudge = ""
        if new_snooze_count >= 3:
            nudge = f"\n💡 Задача переносится уже {new_snooze_count}-й раз. Рекомендую выбрать удобное окно в расписании."

        await query.edit_message_text(f"⏰ Напоминание отложено ({time_label}): «{text}»{nudge}")

    elif action == "del":
        task_id = int(parts[1])
        for job in context.job_queue.get_jobs_by_name(f"task_{chat_id}_{task_id}"):
            job.schedule_removal()
        for job in context.job_queue.get_jobs_by_name(f"followup_{chat_id}_{task_id}"):
            job.schedule_removal()
        delete_task(chat_id, task_id)
        await query.edit_message_text("🗑 Задача удалена.")

    elif action == "rec_done":
        task_id = int(parts[1])
        mark_recurring_done(chat_id, task_id)
        task = get_task_by_id(chat_id, task_id)
        text = task.get("text", "") if task else ""
        await query.edit_message_text(f"🔥 Привычка выполнена за сегодня: «{text}»\nТак держать!")

    elif action == "rec_snooze":
        task_id = int(parts[1])
        mins = int(parts[2])
        task = get_task_by_id(chat_id, task_id)
        text = task.get("text", "привычка") if task else "привычка"
        context.job_queue.run_once(
            send_recurring_reminder,
            when=timedelta(minutes=mins),
            chat_id=chat_id,
            data=(task_id, text),
            name=f"rec_snooze_{chat_id}_{task_id}",
        )
        await query.edit_message_text(f"⏰ Напомню о привычке «{text}» через {mins} мин.")


async def snooze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /snooze <минуты>")
        return

    minutes = int(context.args[0])
    for msg in await _snooze_last_fired(chat_id, minutes, context.job_queue):
        await update.message.reply_text(msg)


def _build_insights_reply(chat_id: int) -> str:
    stats = {**get_weekly_stats(chat_id), **get_recurring_weekly_stats(chat_id)}
    if stats["created_this_week"] == 0 and stats["recurring_fired_this_week"] == 0:
        return "За эту неделю пока нет данных для статистики."
    return generate_weekly_insight(stats)


async def insights_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    await update.message.reply_text(f"📊 Еженедельная сводка:\n{_build_insights_reply(chat_id)}")


async def send_weekly_insight(context: ContextTypes.DEFAULT_TYPE):
    for chat_id in list_all_chat_ids():
        stats = {**get_weekly_stats(chat_id), **get_recurring_weekly_stats(chat_id)}
        if stats["created_this_week"] == 0 and stats["recurring_fired_this_week"] == 0:
            continue
        insight = generate_weekly_insight(stats)
        await context.bot.send_message(chat_id, f"📊 Еженедельный отчет по продуктивности:\n{insight}")


async def send_daily_reminder(context: ContextTypes.DEFAULT_TYPE):
    for chat_id in list_all_chat_ids():
        tasks = list_pending_tasks(chat_id)

        if not tasks:
            await context.bot.send_message(chat_id, "Сегодня нет незавершенных дел. Отличного дня! ✨")
            continue

        lines = ["☀️ Ваши активные дела и заметки:"]
        for i, t in enumerate(tasks):
            lines.append(f"{i + 1}. {t['text']}{_format_task_due(t)}")
        lines.append("\nЧтобы отметить задачу выполненной, используй /done <номер>.")

        _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(tasks)}
        await context.bot.send_message(chat_id, "\n".join(lines))


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    register_user(update.effective_chat.id)
    await update.message.reply_text(
        "👋 Привет! Я твой персональный AI-помощник для заметок и напоминаний.\n\n"
        "Ты можешь отправлять мне текст или голосовые сообщения:\n"
        "• «Купить кофе и молоко» — запишется как заметка\n"
        "• «Напомни позвонить врачу завтра в 15:30» — разовое напоминание\n"
        "• «Напоминай каждый день в 09:00 пить воду кроме воскресенья» — регулярная привычка\n"
        "• «Что у меня на сегодня?» или «Покажи список дел»\n"
        "• «Что я делал на прошлой неделе по налогам?» — умный поиск в истории дел\n\n"
        "💡 В напоминаниях доступны удобные кнопки в 1 клик:\n"
        "[✅ Выполнено], [⏰ +15 мин], [⏰ +1 час], [📅 На завтра].\n"
        "Если задача не выполнена, бот пришлет контрольный переспрос!\n\n"
        "Основные команды:\n"
        "/list — все задачи и заметки\n"
        "/today — дела на сегодня\n"
        "/bydate — группировка по датам\n"
        "/done <номер> — отметить выполненной\n"
        "/recurring — регулярные напоминания\n"
        "/stoprecurring <номер> — отключить регулярное напоминание\n"
        "/snooze <минуты> — отложить последнее напоминание\n"
        "/search <запрос> — быстрый поиск среди дел\n"
        "/history <вопрос> — поиск в истории выполненных задач\n"
        "/insights — сводка продуктивности"
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок Telegram."""
    print(f"[Bot] Ошибка при обработке запроса: {context.error}")
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Не удалось обработать запрос. Пожалуйста, попробуйте еще раз."
            )
        except Exception:
            pass


def main():
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан. Укажите его в файле .env.")
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_error_handler(on_error)

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("list", list_tasks_cmd))
    app.add_handler(CommandHandler("bydate", bydate_cmd))
    app.add_handler(CommandHandler("today", today_cmd))
    app.add_handler(CommandHandler("done", done_cmd))
    app.add_handler(CommandHandler("recurring", recurring_cmd))
    app.add_handler(CommandHandler("stoprecurring", stop_recurring_cmd))
    app.add_handler(CommandHandler("snooze", snooze_cmd))
    app.add_handler(CommandHandler("insights", insights_cmd))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("history", search_history_cmd))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    hour, minute = map(int, REMINDER_TIME.split(":"))
    app.job_queue.run_daily(send_daily_reminder, time=dtime(hour=hour, minute=minute, tzinfo=BOT_TZ))
    app.job_queue.run_daily(
        send_weekly_insight, time=dtime(hour=hour, minute=minute, tzinfo=BOT_TZ), days=(6,)
    )

    for chat_id in list_all_chat_ids():
        for t in list_pending_tasks(chat_id):
            schedule_task_reminder(
                app.job_queue, chat_id, t["id"], t["text"], t["due_time"], snooze_count=t["snooze_count"]
            )
        for t in list_recurring_tasks(chat_id):
            exclude_days = t["exclude_days"].split(",") if t["exclude_days"] else []
            schedule_recurring_reminder(
                app.job_queue, chat_id, t["id"], t["text"], t["recurring_time"], exclude_days
            )

    print(f"Бот запущен (таймзона: {BOT_TZ}). Ожидание сообщений... (Ctrl+C для остановки)")
    app.run_polling()


if __name__ == "__main__":
    main()

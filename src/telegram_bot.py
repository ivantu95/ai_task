"""
Основной модуль Telegram-бота.
Использует long polling (не требует webhook, белого IP или ngrok)
и JobQueue для надежной отправки запланированных напоминаний.

Включает:
- Интерактивный Live Checklist (зачеркивание в 1 клик прямо в сообщении)
- Автоматические категории и фильтры (#покупки, #работа, #личное, #идеи)
- Отдельный режим списка покупок (/shopping, /buy)
- Вечерний обзор и рефлексию (Evening Wind-Down в 21:00)
- Контрольный переспрос невыполненных задач (Follow-up)
- Подсветку просроченных дел (⚠️ просрочено)
"""
import html
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
    get_tasks_completed_today, reschedule_overdue_tasks_to_tomorrow,
    get_tasks_for_date, get_task_dates_in_range, check_time_conflict, find_task_by_text,
    mark_in_progress,
)
from src.llm_parser import (
    parse_task, transcribe_voice, generate_weekly_insight, generate_history_answer,
    structure_smart_note,
)
from src.semantic_search import semantic_search

WEEKDAY_NUMBERS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
RU_WEEKDAYS = {"mon": "пн", "tue": "вт", "wed": "ср", "thu": "чт", "fri": "пт", "sat": "сб", "sun": "вс"}

RU_CATEGORIES = {
    "all": "📋 Все",
    "shopping": "🛒 Покупки",
    "work": "💼 Работа",
    "personal": "🏠 Личное",
    "ideas": "💡 Идеи",
}

RU_CATEGORY_TAGS = {
    "work": " <i>#работа</i>",
    "personal": " <i>#личное</i>",
    "shopping": " <i>#покупки</i>",
    "ideas": " <i>#идеи</i>",
    "general": "",
}

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
REMINDER_TIME = os.environ.get("REMINDER_TIME", "08:00")
EVENING_TIME = os.environ.get("EVENING_TIME", "21:00")
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
            InlineKeyboardButton("⏳ В процессе", callback_data=f"inp:{task_id}"),
        ],
        [
            InlineKeyboardButton("⏰ +15 мин", callback_data=f"snooze:{task_id}:15"),
            InlineKeyboardButton("⏰ +1 час", callback_data=f"snooze:{task_id}:60"),
            InlineKeyboardButton("📅 На завтра", callback_data=f"snooze:{task_id}:tomorrow"),
        ],
        [
            InlineKeyboardButton("🗑 Удалить", callback_data=f"del:{task_id}"),
        ],
    ])


def get_followup_keyboard(task_id: int) -> InlineKeyboardMarkup:
    """Клавиатура для контрольного переспроса."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да, сделано", callback_data=f"done:{task_id}"),
            InlineKeyboardButton("⏳ В процессе", callback_data=f"inp:{task_id}"),
        ],
        [
            InlineKeyboardButton("⏰ Напомнить через 30 мин", callback_data=f"snooze:{task_id}:30"),
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
    """Срабатывание разового напоминания."""
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


RU_MONTH_GENITIVE = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря"
]
RU_WEEKDAYS_LONG = [
    "Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"
]
RU_WEEKDAYS_SHORT = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def _format_daily_time(t: dict) -> str:
    """Форматирует время для отображения в ежедневном расписании/календаре."""
    if t.get("status") == "in_progress":
        return "⏳ В процессе"
    if not t.get("due_time"):
        return "📝"
    start_time = t["due_time"].split(" ")[1] if " " in t["due_time"] else t["due_time"]
    if t.get("end_time"):
        end_time = t["end_time"].split(" ")[1] if " " in t["end_time"] else t["end_time"]
        return f"⏱ {start_time} – {end_time}"
    return f"⏰ {start_time}"


def _format_task_due(t: dict) -> str:
    """Форматирует метку времени (с учетом диапазона start - end) и подсветкой просроченных задач."""
    if t.get("status") == "in_progress":
        return " (⏳ в процессе)"

    if not t.get("due_time"):
        return " (📝 заметка)"

    due_dt = _parse_due_datetime(t["due_time"])
    now_dt = datetime.now(BOT_TZ)
    is_overdue = due_dt and due_dt < now_dt and t.get("status") != "done"

    start_part = t["due_time"]
    end_part = t.get("end_time")

    if end_part:
        if start_part[:10] == end_part[:10] and " " in end_part:
            time_range = f"{start_part} – {end_part.split(' ')[1]}"
        else:
            time_range = f"{start_part} – {end_part}"
        prefix = "⚠️ (просрочено: " if is_overdue else " (⏱ "
        return f" {prefix}{time_range})"

    if is_overdue:
        return f" ⚠️ (просрочено: {start_part})"
    return f" (⏰ {start_part})"


def _build_calendar_view(chat_id: int, target_date: date) -> tuple[str, InlineKeyboardMarkup]:
    """Строит визуальный интерактивный календарь с сеткой недели и таймлайном выбранного дня."""
    start_of_week = target_date - timedelta(days=target_date.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    dates_with_tasks = get_task_dates_in_range(chat_id, start_of_week.isoformat(), end_of_week.isoformat())

    # Кнопки дней недели (Row 1: Пн-Чт, Row 2: Пт-Вс)
    week_buttons_row1 = []
    week_buttons_row2 = []

    for i in range(7):
        cur_day = start_of_week + timedelta(days=i)
        wd_name = RU_WEEKDAYS_SHORT[i]
        day_num = cur_day.day

        if cur_day == target_date:
            btn_label = f"• {wd_name} {day_num} •"
        elif cur_day.isoformat() in dates_with_tasks:
            btn_label = f"{wd_name} {day_num}•"
        else:
            btn_label = f"{wd_name} {day_num}"

        btn = InlineKeyboardButton(btn_label, callback_data=f"cal_d:{cur_day.isoformat()}")
        if i < 4:
            week_buttons_row1.append(btn)
        else:
            week_buttons_row2.append(btn)

    # Кнопки навигации (Row 3)
    prev_week = start_of_week - timedelta(days=7)
    next_week = start_of_week + timedelta(days=7)
    today_dt = date.today()

    nav_row = [
        InlineKeyboardButton("◀ Пред. неделя", callback_data=f"cal_w:{prev_week.isoformat()}"),
        InlineKeyboardButton("Сегодня", callback_data=f"cal_d:{today_dt.isoformat()}"),
        InlineKeyboardButton("След. неделя ▶", callback_data=f"cal_w:{next_week.isoformat()}"),
    ]

    # Данные за выбранный день
    tasks = get_tasks_for_date(chat_id, target_date.isoformat())
    day_name = RU_WEEKDAYS_LONG[target_date.weekday()]
    month_name = RU_MONTH_GENITIVE[target_date.month - 1]

    header = f"📅 <b>Календарь: {day_name}, {target_date.day} {month_name}</b>\n━━━━━━━━━━━━━━━━━━━━"

    check_buttons = []
    if not tasks:
        content = "<i>✨ На этот день ничего не запланировано.</i>"
    else:
        # Локальная нумерация от 1 до N
        _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(tasks)}

        lines = []
        chk_row = []
        for i, t in enumerate(tasks, 1):
            time_badge = _format_daily_time(t)
            mark = "🔴 " if t["priority"] == "high" else ""
            tag = RU_CATEGORY_TAGS.get(t.get("category"), "")
            lines.append(f"<b>{i}.</b> {time_badge} | {mark}{html.escape(t['text'])}{tag}")

            chk_row.append(InlineKeyboardButton(f"{i}. ✅", callback_data=f"chk_cal:{t['id']}:{target_date.isoformat()}"))
            if len(chk_row) == 4:
                check_buttons.append(chk_row)
                chk_row = []

        if chk_row:
            check_buttons.append(chk_row)

        content = "\n".join(lines)

    all_keyboard_rows = []
    if check_buttons:
        all_keyboard_rows.extend(check_buttons)
    all_keyboard_rows.append(week_buttons_row1)
    all_keyboard_rows.append(week_buttons_row2)
    all_keyboard_rows.append(nav_row)

    text_reply = f"{header}\n{content}"
    return text_reply, InlineKeyboardMarkup(all_keyboard_rows)


def _build_checklist(chat_id: int, category: str = "all") -> tuple[str, InlineKeyboardMarkup]:
    """Строит интерактивный чек-лист дел с кнопками зачеркивания и фильтрами по категориям."""
    cat_filter = None if category in (None, "all", "") else category
    tasks = list_pending_tasks(chat_id, category=cat_filter)
    cat_name = RU_CATEGORIES.get(category, "📋 Все задачи")

    _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(tasks)}

    filter_row = [
        InlineKeyboardButton("• Все •" if category == "all" else "📋 Все", callback_data="flt:all"),
        InlineKeyboardButton("• Покупки •" if category == "shopping" else "🛒 Покупки", callback_data="flt:shopping"),
        InlineKeyboardButton("• Работа •" if category == "work" else "💼 Работа", callback_data="flt:work"),
        InlineKeyboardButton("• Идеи •" if category == "ideas" else "💡 Идеи", callback_data="flt:ideas"),
    ]

    if not tasks:
        text = f"<b>{cat_name}:</b>\n<i>Нет активных задач в этом списке.</i>"
        keyboard = InlineKeyboardMarkup([filter_row])
        return text, keyboard

    lines = [f"<b>{cat_name}</b> ({len(tasks)}):"]
    for i, t in enumerate(tasks):
        mark = "🔴 " if t["priority"] == "high" else ""
        tag = RU_CATEGORY_TAGS.get(t.get("category"), "")
        lines.append(f"{i + 1}. {mark}{t['text']}{_format_task_due(t)}{tag}")

    lines.append("\n<i>Нажмите кнопку ниже, чтобы отметить выполненной:</i>")

    # Кнопки для завершения задач (по 4 в ряд)
    check_buttons = []
    current_row = []
    for i, t in enumerate(tasks[:16]):  # до 16 задач в кнопках
        current_row.append(InlineKeyboardButton(f"{i + 1}. ✅", callback_data=f"chk:{t['id']}:{category}"))
        if len(current_row) == 4:
            check_buttons.append(current_row)
            current_row = []
    if current_row:
        check_buttons.append(current_row)

    check_buttons.append(filter_row)
    return "\n".join(lines), InlineKeyboardMarkup(check_buttons)


def _build_today_reply(chat_id: int) -> str:
    tasks = list_pending_tasks(chat_id)
    today_str = date.today().isoformat()
    now_dt = datetime.now(BOT_TZ)

    todays_tasks = [
        t for t in tasks
        if (t["due_time"] and t["due_time"].startswith(today_str))
        or (t.get("status") in ("reminded", "in_progress"))
    ]
    if not todays_tasks:
        return "На сегодня нет запланированных задач."

    _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(todays_tasks)}
    lines = ["📅 <b>Задачи на сегодня:</b>"]
    for i, t in enumerate(todays_tasks):
        mark = "🔴 " if t["priority"] == "high" else ""
        time_part = t["due_time"].split(" ")[1] if t.get("due_time") and " " in t["due_time"] else ""
        due_dt = _parse_due_datetime(t["due_time"]) if t.get("due_time") else None

        if t.get("status") == "in_progress":
            suffix = " (⏳ в процессе)"
        elif due_dt and due_dt < now_dt:
            suffix = f" (⚠️ просрочено: {time_part})" if time_part else " ⚠️ (просрочено)"
        else:
            suffix = f" (в {time_part})" if time_part else ""

        tag = RU_CATEGORY_TAGS.get(t.get("category"), "")
        lines.append(f"{i + 1}. {mark}{t['text']}{suffix}{tag}")
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
    text_reply, kb = _build_checklist(chat_id, category="all")
    await update.message.reply_text(text_reply, parse_mode="HTML", reply_markup=kb)


async def shopping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    text_reply, kb = _build_checklist(chat_id, category="shopping")
    await update.message.reply_text(text_reply, parse_mode="HTML", reply_markup=kb)


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
            tag = RU_CATEGORY_TAGS.get(t.get("category"), "")
            lines.append(f"  {counter}. {mark}{t['text']}{_format_task_due(t)}{tag}")
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
    await update.message.reply_text(_build_today_reply(chat_id), parse_mode="HTML")


async def done_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    if not context.args:
        await update.message.reply_text("Использование: /done <номер> или /done <название задачи>")
        return

    arg = " ".join(context.args).strip()
    if arg.isdigit():
        reply = _mark_done_smart(chat_id, task_number=int(arg), job_queue=context.job_queue)
    else:
        reply = _mark_done_smart(chat_id, task_text=arg, job_queue=context.job_queue)
    await update.message.reply_text(reply)


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
    category = parsed.get("category", "general")
    due_time = parsed.get("due_time")
    end_time = parsed.get("end_time")

    task_id = add_task(
        chat_id, parsed["task"], due_time, priority, snooze_count=0, category=category, end_time=end_time
    )
    schedule_task_reminder(job_queue, chat_id, task_id, parsed["task"], due_time, snooze_count=0)

    if due_time and end_time:
        if due_time[:10] == end_time[:10] and " " in end_time:
            due_str = f" (⏱ {due_time} – {end_time.split(' ')[1]})"
        else:
            due_str = f" (⏱ {due_time} – {end_time})"
    elif due_time:
        due_str = f" (⏰ {due_time})"
    else:
        due_str = " (📝 заметка)"

    mark = "🔴 " if priority == "high" else ""
    tag = RU_CATEGORY_TAGS.get(category, "")

    # Проверка наложений (conflict detection)
    conflict_note = ""
    if due_time:
        conflicts = check_time_conflict(chat_id, due_time, end_time, exclude_id=task_id)
        if conflicts:
            conflict_names = []
            for c in conflicts[:2]:
                c_time = c["due_time"].split(" ")[1] if " " in c["due_time"] else c["due_time"]
                if c.get("end_time") and " " in c["end_time"]:
                    c_time = f"{c_time} – {c['end_time'].split(' ')[1]}"
                conflict_names.append(f"«{c['text']}» ({c_time})")
            conflict_note = f"\n⚠️ <i>Внимание: пересекается по времени с {', '.join(conflict_names)}</i>"

    return f"✅ <b>Добавлено:</b> {mark}{html.escape(parsed['task'])}{due_str}{tag}{conflict_note}"


def _mark_done_smart(
    chat_id: int, task_number: int | None = None, task_text: str | None = None, job_queue=None
) -> str:
    """Умное завершение задачи по номеру или фрагменту названия."""
    target_id = None
    target_text = ""

    # 1. Поиск по тексту (например: "я купил молоко", "сдал отчет")
    if task_text:
        matched = find_task_by_text(chat_id, task_text)
        if matched:
            target_id = matched["id"]
            target_text = matched["text"]

    # 2. Поиск по локальному номеру
    elif task_number is not None:
        target_id = _last_list_mapping.get(chat_id, {}).get(task_number)
        if target_id is None:
            active_tasks = list_pending_tasks(chat_id)
            if 1 <= task_number <= len(active_tasks):
                target_id = active_tasks[task_number - 1]["id"]
                target_text = active_tasks[task_number - 1]["text"]

        if target_id is not None and not target_text:
            task = get_task_by_id(chat_id, target_id)
            target_text = task.get("text", f"#{task_number}") if task else f"#{task_number}"

    if target_id is not None:
        if mark_done(chat_id, target_id):
            if job_queue:
                for job in job_queue.get_jobs_by_name(f"followup_{chat_id}_{target_id}"):
                    job.schedule_removal()
            return f"✅ Выполнено: «{target_text}»!"
        return f"Задача «{target_text}» уже была завершена ранее."

    recurring_task_id = _last_fired_recurring.get(chat_id)
    if recurring_task_id is not None:
        mark_recurring_done(chat_id, recurring_task_id)
        return "Отлично! Регулярная привычка отмечена выполненной за сегодня. 🔥"

    return "Не понял, какую задачу отметить. Вы можете указать номер (например, /done 1) или написать: «сделал <название задачи>»."


def _mark_in_progress_smart(
    chat_id: int, task_number: int | None = None, task_text: str | None = None, job_queue=None
) -> tuple[str, InlineKeyboardMarkup | None]:
    """Умный перевод задачи в статус 'in_progress' по номеру или фрагменту текста."""
    target_id = None
    target_text = ""

    if task_text:
        matched = find_task_by_text(chat_id, task_text)
        if matched:
            target_id = matched["id"]
            target_text = matched["text"]
    elif task_number is not None:
        target_id = _last_list_mapping.get(chat_id, {}).get(task_number)
        if target_id is None:
            active_tasks = list_pending_tasks(chat_id)
            if 1 <= task_number <= len(active_tasks):
                target_id = active_tasks[task_number - 1]["id"]
                target_text = active_tasks[task_number - 1]["text"]

        if target_id is not None and not target_text:
            task = get_task_by_id(chat_id, target_id)
            target_text = task.get("text", f"#{task_number}") if task else f"#{task_number}"
    else:
        fired = _last_fired.get(chat_id)
        if fired:
            text_fired, _ = fired
            matched = find_task_by_text(chat_id, text_fired)
            if matched:
                target_id = matched["id"]
                target_text = matched["text"]

    if target_id is not None:
        mark_in_progress(chat_id, target_id)
        if job_queue:
            for job in job_queue.get_jobs_by_name(f"followup_{chat_id}_{target_id}"):
                job.schedule_removal()
            for job in job_queue.get_jobs_by_name(f"task_{chat_id}_{target_id}"):
                job.schedule_removal()

        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Завершить", callback_data=f"done:{target_id}"),
                InlineKeyboardButton("⏰ Напомнить через 1 час", callback_data=f"snooze:{target_id}:60"),
            ]
        ])
        return (
            f"⏳ <b>В процессе:</b> «{target_text}»\n"
            f"Отличный фокус! Не буду отвлекать уведомлениями. Как завершите — нажмите кнопку ниже:",
            kb,
        )

    return (
        "Не понял, какую задачу перевести в статус «В процессе». Укажите номер или название (например: «делаю отчет»).",
        None,
    )


async def _process_smart_voice_note(chat_id: int, text: str, duration: int, job_queue) -> str:
    """Обрабатывает длинный монолог/поток сознания (AudioPen style):
    структурирует мысли, выделяет задачи с дедлайнами и сохраняет их в базу данных."""
    smart = structure_smart_note(text)

    title = smart.get("title") or "Голосовая заметка"
    summary = smart.get("summary") or text
    key_points = smart.get("key_points") or []
    category = smart.get("category") or "general"
    action_items = smart.get("action_items") or []

    created_tasks = []
    for item in action_items:
        task_text = item.get("task", "").strip()
        if not task_text:
            continue
        due_time = item.get("due_time")
        end_time = item.get("end_time")
        priority = item.get("priority", "normal")
        item_cat = item.get("category") or category
        if item_cat not in RU_CATEGORIES and item_cat != "general":
            item_cat = category

        task_id = add_task(
            chat_id=chat_id,
            text=task_text,
            due_time=due_time,
            priority=priority,
            snooze_count=0,
            category=item_cat,
            end_time=end_time,
        )
        if due_time:
            schedule_task_reminder(job_queue, chat_id, task_id, task_text, due_time, snooze_count=0)

        created_tasks.append({
            "id": task_id,
            "text": task_text,
            "due_time": due_time,
            "end_time": end_time,
            "priority": priority,
            "category": item_cat,
        })

    saved_note_id = None
    # Если задач не было, сохраняем саму мысль/заметку
    if not created_tasks:
        note_text = f"📝 {title}: {summary}"
        saved_note_id = add_task(
            chat_id=chat_id,
            text=note_text,
            due_time=None,
            priority="normal",
            snooze_count=0,
            category=category,
        )
    elif len(key_points) >= 2 or len(summary) > 120:
        # Если есть отдельный ценный контекст/мысли помимо задач, сохраняем конспект как заметку
        note_text = f"📝 {title}: {summary}"
        saved_note_id = add_task(
            chat_id=chat_id,
            text=note_text,
            due_time=None,
            priority="normal",
            snooze_count=0,
            category=category,
        )

    dur_str = f"{duration // 60}:{duration % 60:02d}" if duration else ""
    dur_badge = f" (<i>{dur_str}</i>)" if dur_str else ""
    tag = RU_CATEGORY_TAGS.get(category, "")

    header = f"🎙 <b>Умная голосовая заметка</b>{dur_badge}{tag}" if duration else f"📝 <b>Умная заметка</b>{tag}"

    blocks = [
        header,
        f"📌 <b>{html.escape(title)}</b>",
        f"<b>Суть:</b>\n{html.escape(summary)}",
    ]

    if key_points:
        points = "\n".join(f"• {html.escape(str(p))}" for p in key_points)
        blocks.append(f"<b>Ключевые мысли:</b>\n{points}")

    if created_tasks:
        task_lines = []
        for i, t in enumerate(created_tasks, 1):
            mark = "🔴 " if t["priority"] == "high" else ""
            if t.get("end_time") and t.get("due_time"):
                if t["due_time"][:10] == t["end_time"][:10] and " " in t["end_time"]:
                    t_due = f" (⏱ <i>{t['due_time']} – {t['end_time'].split(' ')[1]}</i>)"
                else:
                    t_due = f" (⏱ <i>{t['due_time']} – {t['end_time']}</i>)"
            elif t.get("due_time"):
                t_due = f" (⏰ <i>{t['due_time']}</i>)"
            else:
                t_due = ""
            t_tag = RU_CATEGORY_TAGS.get(t["category"], "")
            task_lines.append(f"{i}. {mark}{html.escape(t['text'])}{t_due}{t_tag}")
        blocks.append("📋 <b>Выделенные задачи:</b>\n" + "\n".join(task_lines))
    elif saved_note_id:
        blocks.append("<i>Сохранено в список заметок</i>")

    blocks.append("<i>Список дел: /list | Календарь: /calendar | Покупки: /shopping</i>")
    return "\n\n".join(blocks)
    return "\n\n".join(blocks)


async def note_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /note или /memo для текстовой структуризации мыслей и конспектов встреч."""
    chat_id = update.effective_chat.id
    register_user(chat_id)
    if not context.args:
        await update.message.reply_text("Использование: /note <напишите мысли, идею или итоги встречи>")
        return

    text = " ".join(context.args)
    reply_html = await _process_smart_voice_note(chat_id, text, duration=0, job_queue=context.job_queue)
    await update.message.reply_text(reply_html, parse_mode="HTML")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Любое сообщение, не являющееся /командой, анализируется на намерение (intent)."""
    chat_id = update.effective_chat.id
    register_user(chat_id)
    text = update.message.text.strip()

    words = text.split()
    if len(words) >= 30 and len(text) >= 150:
        reply_html = await _process_smart_voice_note(chat_id, text, duration=0, job_queue=context.job_queue)
        await update.message.reply_text(reply_html, parse_mode="HTML")
        return

    parsed = parse_task(text)

    intent = parsed.get("intent")

    if intent == "show_shopping":
        text_reply, kb = _build_checklist(chat_id, category="shopping")
        await update.message.reply_text(text_reply, parse_mode="HTML", reply_markup=kb)
        return
    if intent == "show_today":
        await update.message.reply_text(_build_today_reply(chat_id), parse_mode="HTML")
        return
    if intent == "show_calendar":
        target_str = parsed.get("target_date")
        target_dt = date.today()
        if target_str:
            try:
                target_dt = date.fromisoformat(target_str)
            except Exception:
                target_dt = date.today()
        text_reply, kb = _build_calendar_view(chat_id, target_dt)
        await update.message.reply_text(text_reply, parse_mode="HTML", reply_markup=kb)
        return
    if intent == "show_list":
        text_reply, kb = _build_checklist(chat_id, category="all")
        await update.message.reply_text(text_reply, parse_mode="HTML", reply_markup=kb)
        return
    if intent == "show_bydate":
        await update.message.reply_text(_build_bydate_reply(chat_id))
        return
    if intent == "mark_done":
        await update.message.reply_text(
            _mark_done_smart(
                chat_id,
                task_number=parsed.get("task_number"),
                task_text=parsed.get("task"),
                job_queue=context.job_queue,
            )
        )
        return
    if intent == "mark_in_progress":
        reply_text, kb = _mark_in_progress_smart(
            chat_id,
            task_number=parsed.get("task_number"),
            task_text=parsed.get("task"),
            job_queue=context.job_queue,
        )
        await update.message.reply_text(reply_text, parse_mode="HTML", reply_markup=kb)
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
    await update.message.reply_text(reply, parse_mode="HTML")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Скачивает голосовую заметку, транскрибирует через Groq Whisper
    и направляет результат в умный диктофон (AudioPen) или общий пайплайн."""
    chat_id = update.effective_chat.id
    register_user(chat_id)

    duration = update.message.voice.duration if update.message.voice else 0

    voice_file = await context.bot.get_file(update.message.voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
    await voice_file.download_to_drive(tmp_path)

    try:
        text = transcribe_voice(tmp_path)
    finally:
        os.remove(tmp_path)

    if not text or not text.strip():
        await update.message.reply_text("Не удалось распознать речь в голосовом сообщении. Попробуйте еще раз.")
        return

    words = text.strip().split()
    # Если длинное голосовое (>18 сек и не менее 12 слов, либо от 25 слов) — включаем умный диктофон (AudioPen)
    if (duration >= 18 and len(words) >= 12) or (len(words) >= 25):
        reply_html = await _process_smart_voice_note(chat_id, text, duration, context.job_queue)
        await update.message.reply_text(reply_html, parse_mode="HTML")
        return

    parsed = parse_task(text)
    intent = parsed.get("intent")

    if intent == "show_shopping":
        text_reply, kb = _build_checklist(chat_id, category="shopping")
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{text_reply}", parse_mode="HTML", reply_markup=kb)
        return
    if intent == "show_today":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{_build_today_reply(chat_id)}", parse_mode="HTML")
        return
    if intent == "show_calendar":
        target_str = parsed.get("target_date")
        target_dt = date.today()
        if target_str:
            try:
                target_dt = date.fromisoformat(target_str)
            except Exception:
                target_dt = date.today()
        text_reply, kb = _build_calendar_view(chat_id, target_dt)
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{text_reply}", parse_mode="HTML", reply_markup=kb)
        return
    if intent == "show_list":
        text_reply, kb = _build_checklist(chat_id, category="all")
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{text_reply}", parse_mode="HTML", reply_markup=kb)
        return
    if intent == "show_bydate":
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{_build_bydate_reply(chat_id)}")
        return
    if intent == "mark_done":
        reply = _mark_done_smart(
            chat_id,
            task_number=parsed.get("task_number"),
            task_text=parsed.get("task"),
            job_queue=context.job_queue,
        )
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{reply}")
        return
    if intent == "mark_in_progress":
        reply_text, kb = _mark_in_progress_smart(
            chat_id,
            task_number=parsed.get("task_number"),
            task_text=parsed.get("task"),
            job_queue=context.job_queue,
        )
        await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{reply_text}", parse_mode="HTML", reply_markup=kb)
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
    await update.message.reply_text(f"🎙 Распознано: «{text}»\n\n{reply}", parse_mode="HTML")


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
    """Обработка нажатий на инлайн-кнопки под напоминаниями, в списках и вечернем обзоре."""
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

    elif action == "inp":
        task_id = int(parts[1])
        task = get_task_by_id(chat_id, task_id)
        task_text = task.get("text", "") if task else ""
        mark_in_progress(chat_id, task_id)

        for job in context.job_queue.get_jobs_by_name(f"followup_{chat_id}_{task_id}"):
            job.schedule_removal()
        for job in context.job_queue.get_jobs_by_name(f"task_{chat_id}_{task_id}"):
            job.schedule_removal()

        in_progress_kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Завершить", callback_data=f"done:{task_id}"),
                InlineKeyboardButton("⏰ Напомнить через 1 час", callback_data=f"snooze:{task_id}:60"),
            ]
        ])
        await query.edit_message_text(
            f"⏳ <b>В процессе:</b> «{task_text}»\n"
            f"Отличный фокус! Не буду отвлекать короткими напоминаниями. Как освободитесь — нажмите «Завершить»:",
            parse_mode="HTML",
            reply_markup=in_progress_kb,
        )

    elif action == "chk":
        task_id = int(parts[1])
        current_cat = parts[2]
        task = get_task_by_id(chat_id, task_id)
        if mark_done(chat_id, task_id):
            for job in context.job_queue.get_jobs_by_name(f"followup_{chat_id}_{task_id}"):
                job.schedule_removal()

        # Обновляем список прямо в сообщении
        new_text, new_kb = _build_checklist(chat_id, category=current_cat)
        await query.edit_message_text(new_text, parse_mode="HTML", reply_markup=new_kb)

    elif action == "flt":
        target_cat = parts[1]
        new_text, new_kb = _build_checklist(chat_id, category=target_cat)
        await query.edit_message_text(new_text, parse_mode="HTML", reply_markup=new_kb)

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

    elif action == "ev_tomorrow":
        tomorrow_09 = (datetime.now(BOT_TZ) + timedelta(days=1)).strftime("%Y-%m-%d 09:00")
        count = reschedule_overdue_tasks_to_tomorrow(chat_id, tomorrow_09)
        await query.edit_message_text(
            f"✨ Незавершенные задачи ({count} шт.) успешно перенесены на завтра в 09:00!\n"
            f"Хорошего вечера и приятного отдыха 🌙"
        )

    elif action == "ev_dismiss":
        await query.edit_message_text("👌 Задачи оставлены в текущем расписании. Приятного вечера! 🌙")

    elif action == "cal_d":
        target_dt = date.fromisoformat(parts[1])
        new_text, new_kb = _build_calendar_view(chat_id, target_dt)
        await query.edit_message_text(new_text, parse_mode="HTML", reply_markup=new_kb)

    elif action == "cal_w":
        target_dt = date.fromisoformat(parts[1])
        new_text, new_kb = _build_calendar_view(chat_id, target_dt)
        await query.edit_message_text(new_text, parse_mode="HTML", reply_markup=new_kb)

    elif action == "chk_cal":
        task_id = int(parts[1])
        target_dt = date.fromisoformat(parts[2])
        if mark_done(chat_id, task_id):
            for job in context.job_queue.get_jobs_by_name(f"followup_{chat_id}_{task_id}"):
                job.schedule_removal()
        new_text, new_kb = _build_calendar_view(chat_id, target_dt)
        await query.edit_message_text(new_text, parse_mode="HTML", reply_markup=new_kb)


async def calendar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отображает интерактивный календарь с сеткой недели и расписанием дня."""
    chat_id = update.effective_chat.id
    register_user(chat_id)
    target_dt = date.today()
    if context.args:
        arg = context.args[0].lower()
        if arg in ("tomorrow", "завтра"):
            target_dt = date.today() + timedelta(days=1)
        elif arg in ("yesterday", "вчера"):
            target_dt = date.today() - timedelta(days=1)
        else:
            try:
                target_dt = date.fromisoformat(arg)
            except Exception:
                pass
    text_reply, kb = _build_calendar_view(chat_id, target_dt)
    await update.message.reply_text(text_reply, parse_mode="HTML", reply_markup=kb)


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

        lines = ["☀️ <b>Ваши активные дела и заметки:</b>"]
        for i, t in enumerate(tasks):
            tag = RU_CATEGORY_TAGS.get(t.get("category"), "")
            lines.append(f"{i + 1}. {t['text']}{_format_task_due(t)}{tag}")
        lines.append("\nЧтобы открыть интерактивный список с кнопками: /list")

        _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(tasks)}
        await context.bot.send_message(chat_id, "\n".join(lines), parse_mode="HTML")


async def send_evening_review(context: ContextTypes.DEFAULT_TYPE):
    """Вечерний обзор дня в 21:00: итоги и предложение перенести несделанное на завтра."""
    for chat_id in list_all_chat_ids():
        done_today = get_tasks_completed_today(chat_id)
        pending = list_pending_tasks(chat_id)
        today_str = date.today().isoformat()
        now_dt = datetime.now(BOT_TZ)

        # Невыполненные задачи, назначенные на сегодня или просроченные
        overdue_or_today = [
            t for t in pending
            if t.get("due_time") and (
                t["due_time"].startswith(today_str) or (_parse_due_datetime(t["due_time"]) and _parse_due_datetime(t["due_time"]) < now_dt)
            )
        ]

        if not overdue_or_today and done_today == 0:
            continue

        if not overdue_or_today and done_today > 0:
            await context.bot.send_message(
                chat_id,
                f"🌙 <b>Добрый вечер!</b>\n"
                f"Сегодня ты отлично поработал и закрыл <b>{done_today}</b> задач(и) 🎉\n"
                f"Все запланированные дела выполнены. Спокойного отдыха!",
                parse_mode="HTML",
            )
            continue

        lines = [
            f"🌙 <b>Вечерний обзор дня:</b>",
            f"✅ Выполнено сегодня: <b>{done_today}</b>",
            f"⏳ Остались незавершенными ({len(overdue_or_today)}):",
        ]
        for i, t in enumerate(overdue_or_today[:5]):
            lines.append(f"• {t['text']}")
        if len(overdue_or_today) > 5:
            lines.append(f"<i>...и еще {len(overdue_or_today) - 5} задач</i>")

        lines.append("\nПеренести незавершенные дела на завтра?")

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📅 Перенести всё на завтра 09:00", callback_data="ev_tomorrow"),
            ],
            [
                InlineKeyboardButton("👌 Оставить как есть", callback_data="ev_dismiss"),
            ]
        ])

        await context.bot.send_message(
            chat_id,
            "\n".join(lines),
            parse_mode="HTML",
            reply_markup=keyboard,
        )


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    register_user(update.effective_chat.id)
    await update.message.reply_text(
        "👋 <b>Привет! Я твой персональный AI-копилот для дел, заметок и встреч.</b>\n\n"
        "Общайся со мной как с живым ассистентом — <b>голосом</b> или <b>текстом</b> на обычном разговорном языке.\n\n"
        "🚀 <b>Главные возможности:</b>\n"
        "• <b>🎙 Умный диктофон (AudioPen)</b>: запишите голосовое с потоком мыслей — я уберу слова-паразиты, оформлю емкий конспект и сам добавлю найденные задачи в расписание.\n"
        "• <b>📅 Интерактивный календарь (/calendar)</b>: сетка недели, таймлайн дня по часам, интервалы времени (например, <i>«с 14:00 до 15:30»</i>) и проверка наложений.\n"
        "• <b>✅ Live-списки (/list и /shopping)</b>: интерактивные чек-листы с кнопками мгновенного вычеркивания в 1 клик без спама в чат.\n"
        "• <b>🏷 Автокатегории</b>: автоматически определяю категорию дела (<i>#покупки</i>, <i>#работа</i>, <i>#личное</i>, <i>#идеи</i>) с удобными фильтрами.\n"
        "• <b>🎯 Умное закрытие</b>: не нужно помнить цифры — просто напишите или скажите <i>«купил молоко»</i> или <i>«отчет сдал»</i>.\n"
        "• <b>🌙 Вечерний обзор (21:00)</b>: подведение итогов дня и возможность перенести незавершенные дела на завтра в 1 клик.\n\n"
        "💬 <b>Примеры фраз (голосом или текстом):</b>\n"
        "• <i>«Созвон с клиентом завтра с 15:00 до 16:30 #работа»</i>\n"
        "• <i>«Купить молоко, сыр и кофе»</i> ➔ добавит в /shopping\n"
        "• <i>«Напоминай каждый день в 09:00 пить воду кроме воскресенья»</i>\n"
        "• <i>«Что у меня в календаре на четверг?»</i>\n"
        "• <i>«Купил молоко»</i> ➔ вычеркнет из покупок\n\n"
        "📌 <b>Основные команды:</b>\n"
        "/calendar — интерактивный календарь и расписание\n"
        "/list — чек-лист активных дел с кнопками\n"
        "/shopping — отдельный список покупок\n"
        "/today — задачи на сегодня\n"
        "/note &lt;мысли&gt; — структурировать заметку или встречу\n"
        "/done &lt;номер или название&gt; — отметить выполненным\n"
        "/bydate — дела, сгруппированные по датам\n"
        "/recurring — ежедневные привычки\n"
        "/history &lt;вопрос&gt; — умный AI-поиск по выполненным делам\n"
        "/insights — сводка продуктивности",
        parse_mode="HTML"
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
    app.add_handler(CommandHandler(["calendar", "cal", "schedule"], calendar_cmd))
    app.add_handler(CommandHandler("list", list_tasks_cmd))
    app.add_handler(CommandHandler("shopping", shopping_cmd))
    app.add_handler(CommandHandler("buy", shopping_cmd))
    app.add_handler(CommandHandler(["note", "memo"], note_cmd))
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

    # Утренний дайджест в 08:00
    hour, minute = map(int, REMINDER_TIME.split(":"))
    app.job_queue.run_daily(send_daily_reminder, time=dtime(hour=hour, minute=minute, tzinfo=BOT_TZ))

    # Вечерний обзор дня в 21:00
    ehour, eminute = map(int, EVENING_TIME.split(":"))
    app.job_queue.run_daily(send_evening_review, time=dtime(hour=ehour, minute=eminute, tzinfo=BOT_TZ))

    # Еженедельный отчет в воскресенье
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

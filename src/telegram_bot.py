"""
Telegram version. No webhook, no ngrok, no Twilio sandbox expiry.

The bot uses long polling: it makes an OUTBOUND connection to Telegram's
servers and asks "any new messages for me?" every second or so. Since the
connection direction is bot -> Telegram (not the other way around), your
machine never needs a public URL.
"""
import os
import tempfile
from collections import defaultdict
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from src.db import (
    init_db, add_task, list_pending_tasks, mark_done, register_user, list_all_chat_ids,
    add_recurring_task, list_recurring_tasks, delete_task, get_weekly_stats,
    log_recurring_fired, mark_recurring_done, get_recurring_weekly_stats,
    archive_fired_task, get_task_history,
)
from src.llm_parser import parse_task, transcribe_voice, generate_weekly_insight, generate_history_answer
from src.semantic_search import semantic_search

WEEKDAY_NUMBERS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
REMINDER_TIME = os.environ.get("REMINDER_TIME", "08:00")
IST = ZoneInfo("Asia/Kolkata")

_last_list_mapping: dict[int, dict[int, int]] = {}  # chat_id -> {displayed number: actual db id}
_recurring_mapping: dict[int, dict[int, int]] = {}  # chat_id -> {displayed number: actual db id}
_last_fired: dict[int, tuple[str, int]] = {}  # chat_id -> (text, snooze_count) of last fired reminder
_last_fired_recurring: dict[int, int] = {}  # chat_id -> task_id of most recently fired recurring reminder


def _parse_due_datetime(due_time_str: str) -> datetime | None:
    try:
        naive = datetime.strptime(due_time_str, "%Y-%m-%d %H:%M")
        return naive.replace(tzinfo=IST)
    except (ValueError, TypeError):
        return None


async def send_task_reminder(context: ContextTypes.DEFAULT_TYPE):
    """One-time reminder. After sending it, the task is archived (not hard
    deleted) - it disappears from /list but stays queryable via history
    search, since we don't know for sure if the user actually did it."""
    chat_id = context.job.chat_id
    task_id, text, snooze_count = context.job.data
    await context.bot.send_message(chat_id, f"\u23F0 Reminder: {text}")
    archive_fired_task(chat_id, task_id)
    _last_fired[chat_id] = (text, snooze_count)


def schedule_task_reminder(
    job_queue, chat_id: int, task_id: int, text: str, due_time_str: str | None, snooze_count: int = 0
):
    due_dt = _parse_due_datetime(due_time_str) if due_time_str else None
    if due_dt and due_dt > datetime.now(IST):
        job_queue.run_once(
            send_task_reminder,
            when=due_dt,
            chat_id=chat_id,
            data=(task_id, text, snooze_count),
            name=f"task_{chat_id}_{task_id}",
        )


async def send_recurring_reminder(context: ContextTypes.DEFAULT_TYPE):
    """Recurring reminder - just sends the message. Never auto-deletes,
    since it's meant to repeat daily until the user stops it. Logs that
    today's occurrence fired, so weekly insights can track completion."""
    chat_id = context.job.chat_id
    task_id, text = context.job.data
    await context.bot.send_message(chat_id, f"\U0001F501 {text}\n\nReply 'done' jab ho jaye.")
    _last_fired[chat_id] = text
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
        time=dtime(hour=hour, minute=minute, tzinfo=IST),
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


def _build_list_reply(chat_id: int) -> str:
    tasks = list_pending_tasks(chat_id)
    if not tasks:
        return "Koi pending task nahi hai."
    _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(tasks)}
    lines = []
    for i, t in enumerate(tasks):
        mark = "\U0001F534 " if t["priority"] == "high" else ""
        lines.append(f"{i + 1}. {mark}{t['text']}")
    return "\n".join(lines)


def _build_today_reply(chat_id: int) -> str:
    tasks = list_pending_tasks(chat_id)
    today_str = date.today().isoformat()
    todays_tasks = [t for t in tasks if t["due_time"] and t["due_time"].startswith(today_str)]
    if not todays_tasks:
        return "Aaj ke liye koi due task nahi hai."
    _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(todays_tasks)}
    return "\n".join(f"{i + 1}. {t['text']} ({t['due_time'].split(' ')[1]})" for i, t in enumerate(todays_tasks))

def _build_search_reply(chat_id: int, query: str) -> str:
    if not query:
        return "Kya search karna hai, ye bata."

    tasks = list_pending_tasks(chat_id)
    matches = semantic_search(query, tasks, text_key="text", top_k=5, threshold=0.3)
    if not matches:
        return f"'{query}' se milta koi pending task nahi mila."

    _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(matches)}
    lines = []
    for i, t in enumerate(matches):
        mark = "\U0001F534 " if t["priority"] == "high" else ""
        due = f" (due {t['due_time']})" if t["due_time"] else ""
        lines.append(f"{i + 1}. {mark}{t['text']}{due}")
    return "\n".join(lines)


def _build_history_reply(chat_id: int, query: str) -> str:
    if not query:
        return "Kya dhundhna hai purane kaam mein, ye bata."

    history = get_task_history(chat_id)
    if not history:
        return "Abhi tak koi completed ya purana task record nahi hai."

    matches = semantic_search(query, history, text_key="text", top_k=8, threshold=0.25)
    if not matches:
        return f"'{query}' se related purana koi kaam nahi mila."

    return generate_history_answer(query, matches)


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
        return "Koi pending task nahi hai."

    grouped = _group_by_date(tasks)
    ordered_dates = sorted(grouped.keys(), key=lambda d: (d == "No due date", d))

    mapping = {}
    blocks = []
    counter = 1
    for date_str in ordered_dates:
        lines = [f"\U0001F4C5 {date_str}"]
        for t in grouped[date_str]:
            mapping[counter] = t["id"]
            time_part = t["due_time"].split(" ")[1] if t["due_time"] and " " in t["due_time"] else None
            suffix = f" ({time_part})" if time_part else ""
            mark = "\U0001F534 " if t["priority"] == "high" else ""
            lines.append(f"  {counter}. {mark}{t['text']}{suffix}")
            counter += 1
        blocks.append("\n".join(lines))

    _last_list_mapping[chat_id] = mapping
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
        await update.message.reply_text("Use: /done <number from your last list>")
        return

    shown_number = int(context.args[0])
    task_id = _last_list_mapping.get(chat_id, {}).get(shown_number)

    if task_id is None:
        await update.message.reply_text("Pehle /list, /bydate, ya /today chala ke number dekh le.")
        return

    if mark_done(chat_id, task_id):
        await update.message.reply_text(f"Task {shown_number} marked done!")
    else:
        await update.message.reply_text("Ye task pehle se complete ho chuka hai.")


def _build_recurring_reply(chat_id: int) -> str:
    tasks = list_recurring_tasks(chat_id)
    if not tasks:
        return "Koi recurring reminder set nahi hai."

    _recurring_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(tasks)}
    lines = []
    for i, t in enumerate(tasks):
        skip = f", except {t['exclude_days']}" if t["exclude_days"] else ""
        lines.append(f"{i + 1}. {t['text']} - daily at {t['recurring_time']}{skip}")
    lines.append("\nStop karne ke liye: /stoprecurring <number>")
    return "\n".join(lines)


async def recurring_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    await update.message.reply_text(_build_recurring_reply(chat_id))


def _stop_recurring_by_number(chat_id: int, task_number: int | None, job_queue) -> str:
    if task_number is None:
        return "Kaunsa recurring reminder, ye samajh nahi aaya. Pehle /recurring chala ke number dekh le."

    task_id = _recurring_mapping.get(chat_id, {}).get(task_number)
    if task_id is None:
        return "Pehle /recurring chala ke number dekh le."

    for job in job_queue.get_jobs_by_name(f"recurring_{chat_id}_{task_id}"):
        job.schedule_removal()
    delete_task(chat_id, task_id)
    return f"Recurring reminder {task_number} stop kar diya."


async def stop_recurring_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Use: /stoprecurring <number from /recurring>")
        return

    shown_number = int(context.args[0])
    await update.message.reply_text(_stop_recurring_by_number(chat_id, shown_number, context.job_queue))

# def _detect_voice_command(text: str) -> str | None:
#     """Simple keyword check so voice queries like 'show me today's tasks'
#     are treated as commands, not new tasks to add. Returns 'today',
#     'list', or None (meaning: treat this as a new task)."""
#     lower = text.lower()
#     if "today" in lower:
#         return "today"
#     if any(phrase in lower for phrase in ["show me", "total task", "all task", "list my task", "my pending", "what are my task"]):
#         return "list"
#     return None



async def _process_parsed_task(chat_id: int, parsed: dict, job_queue) -> str:
    """Shared logic for typed AND voice messages - given the parsed JSON
    from the LLM, saves the task (recurring or one-time), schedules its
    reminder, and returns the confirmation text to send back."""
    if parsed.get("recurring"):
        task_id = add_recurring_task(chat_id, parsed["task"], parsed["recurring_time"], parsed["exclude_days"])
        schedule_recurring_reminder(
            job_queue, chat_id, task_id, parsed["task"], parsed["recurring_time"], parsed["exclude_days"]
        )
        skip = f", except {', '.join(parsed['exclude_days'])}" if parsed["exclude_days"] else ""
        return f"Recurring reminder set ho gaya: {parsed['task']} - daily at {parsed['recurring_time']}{skip}"

    priority = parsed.get("priority", "normal")
    task_id = add_task(chat_id, parsed["task"], parsed.get("due_time"), priority, snooze_count=0)
    schedule_task_reminder(job_queue, chat_id, task_id, parsed["task"], parsed.get("due_time"), snooze_count=0)
    due = f" (due {parsed['due_time']})" if parsed.get("due_time") else ""
    mark = "\U0001F534 " if priority == "high" else ""
    return f"Task {task_id} add ho gaya: {mark}{parsed['task']}{due}"


def _mark_done_by_number(chat_id: int, task_number: int | None) -> str:
    if task_number is None:
        # No number given - assume they mean the recurring reminder that
        # just fired (e.g. replying "done" right after a workout reminder).
        recurring_task_id = _last_fired_recurring.get(chat_id)
        if recurring_task_id is not None:
            mark_recurring_done(chat_id, recurring_task_id)
            return "Nice! Aaj ka streak mark ho gaya."
        return "Kaunsa task, ye samajh nahi aaya. Pehle /list chala ke number dekh le."

    task_id = _last_list_mapping.get(chat_id, {}).get(task_number)
    if task_id is None:
        return "Pehle /list, /bydate, ya /today chala ke number dekh le, phir bata."

    if mark_done(chat_id, task_id):
        return f"Task {task_number} marked done!"
    return "Ye task pehle se complete ho chuka hai."


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Any message that isn't a /command is checked for intent first -
    it could be a query, a 'mark done' request, or a new task to add."""
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
        await update.message.reply_text(f"\U0001F4CA Weekly Insight:\n{_build_insights_reply(chat_id)}")
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
    """Downloads the voice note, transcribes it with Groq Whisper, then
    lets the LLM's intent classification decide if it's a query or a new
    task - same call that already extracts task/due_time, no extra cost."""
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
        await update.message.reply_text(f"Suna: \"{text}\"\n{_build_today_reply(chat_id)}")
        return
    if intent == "show_list":
        await update.message.reply_text(f"Suna: \"{text}\"\n{_build_list_reply(chat_id)}")
        return
    if intent == "show_bydate":
        await update.message.reply_text(f"Suna: \"{text}\"\n{_build_bydate_reply(chat_id)}")
        return
    if intent == "mark_done":
        await update.message.reply_text(f"Suna: \"{text}\"\n{_mark_done_by_number(chat_id, parsed.get('task_number'))}")
        return
    if intent == "snooze":
        minutes = parsed.get("snooze_minutes") or 10
        replies = await _snooze_last_fired(chat_id, minutes, context.job_queue)
        await update.message.reply_text(f"Suna: \"{text}\"\n" + "\n".join(replies))
        return
    if intent == "show_recurring":
        await update.message.reply_text(f"Suna: \"{text}\"\n{_build_recurring_reply(chat_id)}")
        return
    if intent == "stop_recurring":
        result = _stop_recurring_by_number(chat_id, parsed.get("task_number"), context.job_queue)
        await update.message.reply_text(f"Suna: \"{text}\"\n{result}")
        return
    if intent == "show_insights":
        await update.message.reply_text(f"Suna: \"{text}\"\n\U0001F4CA Weekly Insight:\n{_build_insights_reply(chat_id)}")
        return
    if intent == "search_task":
        await update.message.reply_text(f"Suna: \"{text}\"\n{_build_search_reply(chat_id, parsed.get('search_query'))}")
        return
    if intent == "search_history":
        await update.message.reply_text(f"Suna: \"{text}\"\n{_build_history_reply(chat_id, parsed.get('search_query'))}")
        return

    reply = await _process_parsed_task(chat_id, parsed, context.job_queue)
    await update.message.reply_text(f"Suna: \"{text}\"\n{reply}")


async def _snooze_last_fired(chat_id: int, minutes: int, job_queue) -> list[str]:
    """Shared logic for /snooze command AND natural language snooze
    ('snooze it for 10 minutes'). Returns one or two messages to send."""
    fired = _last_fired.get(chat_id)
    if fired is None:
        return ["Koi recent reminder nahi mila jise snooze karu."]

    text, prev_snooze_count = fired
    new_snooze_count = prev_snooze_count + 1

    new_due = (datetime.now(IST) + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M")
    task_id = add_task(chat_id, text, new_due, snooze_count=new_snooze_count)
    schedule_task_reminder(job_queue, chat_id, task_id, text, new_due, snooze_count=new_snooze_count)

    messages = [f"Theek hai, {minutes} minute mein phir yaad dila dunga: {text}"]
    if new_snooze_count >= 3:
        messages.append(
            f"\U0001F4A1 Maine notice kiya tune '{text}' {new_snooze_count} baar snooze kiya hai. "
            f"Shayad ek fixed better time set karna help karega - jaise kal subah? "
            f"Bas naya time bata de, main update kar dunga."
        )
    return messages


async def snooze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Use: /snooze <minutes>")
        return

    minutes = int(context.args[0])
    for msg in await _snooze_last_fired(chat_id, minutes, context.job_queue):
        await update.message.reply_text(msg)

def _build_insights_reply(chat_id: int) -> str:
    stats = {**get_weekly_stats(chat_id), **get_recurring_weekly_stats(chat_id)}
    if stats["created_this_week"] == 0 and stats["recurring_fired_this_week"] == 0:
        return "Is hafte abhi koi data nahi hai."
    return generate_weekly_insight(stats)


async def insights_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_user(chat_id)
    await update.message.reply_text(f"\U0001F4CA Weekly Insight:\n{_build_insights_reply(chat_id)}")


async def send_weekly_insight(context: ContextTypes.DEFAULT_TYPE):
    for chat_id in list_all_chat_ids():
        stats = {**get_weekly_stats(chat_id), **get_recurring_weekly_stats(chat_id)}
        if stats["created_this_week"] == 0 and stats["recurring_fired_this_week"] == 0:
            continue
        insight = generate_weekly_insight(stats)
        await context.bot.send_message(chat_id, f"\U0001F4CA Weekly Insight:\n{insight}")



async def send_daily_reminder(context: ContextTypes.DEFAULT_TYPE):
    for chat_id in list_all_chat_ids():
        tasks = list_pending_tasks(chat_id)

        if not tasks:
            await context.bot.send_message(chat_id, "Aaj koi pending task nahi hai. Badhiya!")
            continue

        lines = ["Aaj ke pending tasks:"]
        for i, t in enumerate(tasks):
            due = f" (due {t['due_time']})" if t["due_time"] else ""
            lines.append(f"{i + 1}. {t['text']}{due}")
        lines.append("\n/list chala ke number confirm kar, phir '/done <number>' bhej.")

        _last_list_mapping[chat_id] = {i + 1: t["id"] for i, t in enumerate(tasks)}
        await context.bot.send_message(chat_id, "\n".join(lines))


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    register_user(update.effective_chat.id)
    await update.message.reply_text(
        "Hi! Ab tu mujhe naya task bhej sakta hai jaise "
        "'remind me to call mum tomorrow 6pm', ya recurring jaise "
        "'remind me every day except sunday to workout at 6pm'. Voice note bhi bhej sakta hai.\n"
        "Commands: /list, /bydate, /today, /done, /recurring, /stoprecurring, /snooze"
    )


def main():
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()

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
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    hour, minute = map(int, REMINDER_TIME.split(":"))
    hour, minute = map(int, REMINDER_TIME.split(":"))
    app.job_queue.run_daily(send_daily_reminder, time=dtime(hour=hour, minute=minute, tzinfo=IST))
    app.job_queue.run_daily(
        send_weekly_insight, time=dtime(hour=hour, minute=minute, tzinfo=IST), days=(6,)
    )  # PTB days: 0=Monday...6=Sunday

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

    print("Bot started. Polling for messages... (Ctrl+C to stop)")
    app.run_polling()


if __name__ == "__main__":
    main()
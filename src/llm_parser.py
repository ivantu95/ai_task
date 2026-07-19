"""
This is the AI layer. Instead of forcing the user to type a rigid command
like "/add buy milk | 2026-07-13 | 18:00", we let them just type naturally
("remind me to buy milk tomorrow evening") and Groq's LLaMA model extracts
a clean task + due time as JSON.
"""
import os
import json
from datetime import datetime

from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You extract intent from a message to a personal task assistant bot.
Today's date and time is: {now}

Reply with ONLY valid JSON, no other text, no markdown fences, in this exact format:
{{"intent": "add_task" or "show_today" or "show_list" or "show_bydate" or
  "mark_done" or "snooze" or "show_recurring" or "stop_recurring" or "show_insights" or
  "search_task" or "search_history",
  "task": "short task description", "due_time": "YYYY-MM-DD HH:MM" or null,
  "recurring": true or false, "recurring_time": "HH:MM" or null, "exclude_days": [],
  "priority": "high" or "normal", "task_number": integer or null,
  "snooze_minutes": integer or null, "search_query": "keyword phrase" or null}}

Rules for intent (decide this FIRST):
- "show_today": asking to SEE/view tasks due today (e.g. "show me today's tasks").
- "show_list": asking to SEE/view all pending tasks (e.g. "show me all my tasks", "total tasks").
- "show_bydate": asking to SEE tasks grouped/organized by date (e.g. "show my tasks by date", "date wise list").
- "mark_done": saying a numbered task is complete (e.g. "mark 3rd task done", "task 2 is complete").
  Extract the number into task_number.
- "snooze": asking to delay/postpone/snooze the most recent reminder
  (e.g. "snooze it for 10 minutes", "remind me again in 15 mins", "not now,
  ask me later"). Extract the delay in minutes into snooze_minutes
  (convert hours to minutes). Default to 10 if a delay is implied but no
  number given.
- "show_recurring": asking to see recurring/daily reminders that are set up
  (e.g. "show my recurring reminders", "what daily reminders do I have").
- "stop_recurring": asking to stop/cancel a numbered recurring reminder
  (e.g. "stop recurring reminder 2", "cancel the 1st daily reminder").
  Extract the number into task_number.
- "show_insights": asking for a summary/analysis/report of how they're doing
  (e.g. "how am I doing this week", "show my insights", "give me a summary").
- "search_task": asking to FIND a specific CURRENT/PENDING task by topic
  (e.g. "find my grocery task", "do I have anything pending about the
  report"). This is about tasks not yet done. Extract the topic into
  search_query.
- "search_history": asking about PAST or completed work (e.g. "what did I
  do last week about finance", "have I done anything about the report
  before", "show my history for workout"). This is about tasks already
  completed or already reminded, NOT current pending ones. Extract the
  topic into search_query.
- "add_task": anything else - describes something to DO or be reminded
  about, even if it mentions "today" as a due time (e.g. "remind me to
  submit the report today evening" is add_task, NOT show_today, because
  it describes an action to perform, not a request to see a list).

For intents other than "add_task", set all unused fields to null/false/[]
(task, due_time, recurring, recurring_time, exclude_days stay at their
defaults; only task_number, snooze_minutes, or search_query get filled in
when relevant).

Rules for ONE-TIME tasks (intent = add_task, recurring = false):
- If no specific date/time is mentioned, set due_time to null.
- Resolve relative dates like "tomorrow" or "tonight" using today's date above.
- For phrases like "in N minutes" or "in N hours", calculate the EXACT time by
  adding N minutes/hours to the current time given above. Do this arithmetic
  carefully and precisely - do not round or guess.
- Example: if now is "2026-07-14 21:51" and the message says "in 2 minutes",
  due_time must be exactly "2026-07-14 21:53".
- Keep the task description short and action-oriented.

Rules for RECURRING tasks (intent = add_task, recurring = true):
- Trigger this when the message says the reminder should repeat daily/every
  day/regularly (e.g. "remind me every day to X at 6pm", "daily reminder to X").
- Set due_time to null, and recurring_time to the time mentioned in 24-hour
  "HH:MM" format.
- exclude_days is a list of lowercase 3-letter day codes (mon, tue, wed, thu,
  fri, sat, sun) for any days the message says to SKIP. Empty list if none
  mentioned (e.g. "except Sunday" -> exclude_days: ["sun"]).

Rules for priority (applies to add_task only):
- Set priority to "high" if the message uses words like "urgent", "important",
  "asap", "high priority", "critical". Otherwise "normal".
"""


def parse_task(message: str) -> dict:
    now = datetime.now().strftime("%Y-%m-%d %H:%M, %A")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(now=now)},
            {"role": "user", "content": message},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    fallback = {
        "intent": "add_task", "task": message, "due_time": None, "recurring": False,
        "recurring_time": None, "exclude_days": [], "priority": "normal",
        "task_number": None, "snooze_minutes": None, "search_query": None,
    }

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return fallback
    except json.JSONDecodeError:
        return fallback

    parsed.setdefault("intent", "add_task")
    parsed.setdefault("task", message)
    parsed.setdefault("recurring", False)
    parsed.setdefault("recurring_time", None)
    parsed.setdefault("exclude_days", [])
    parsed.setdefault("priority", "normal")
    parsed.setdefault("task_number", None)
    parsed.setdefault("snooze_minutes", None)
    parsed.setdefault("search_query", None)
    return parsed


def transcribe_voice(file_path: str) -> str:
    """Sends a voice note (ogg file) to Groq's Whisper model and returns
    the transcribed text, so voice messages can go through the same
    parse_task() pipeline as typed messages."""
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
        )
    return response.text




def generate_history_answer(query: str, matches: list[dict]) -> str:
    """The 'generation' half of RAG - takes the semantically retrieved past
    tasks and turns them into a natural-language answer to the user's
    question, instead of just dumping a raw list."""
    context_lines = []
    for m in matches:
        status = "completed" if m["status"] == "done" else "reminded (not confirmed done)"
        when = m.get("completed_at") or m.get("created_at")
        context_lines.append(f"- {m['text']} [{status}, {when}]")
    context_text = "\n".join(context_lines)

    prompt = f"""A user asked about their past tasks: "{query}"

Here are the matching past tasks retrieved from their history:
{context_text}

Write a short, natural 2-4 sentence answer in Hinglish (Hindi-English mix)
summarizing what you found. Mention specific tasks by name where relevant.
If the matches don't seem truly relevant to the question, say so honestly
instead of forcing a connection."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def generate_weekly_insight(stats: dict) -> str:
    """Turns raw numbers into a short, natural-language weekly summary."""
    prompt = f"""Here is a user's task completion data for this week vs last week:
- One-time tasks created this week: {stats['created_this_week']}
- One-time tasks completed this week: {stats['done_this_week']}
- Completion rate this week: {stats['completion_rate']}%
- Completion rate last week: {stats['prev_completion_rate']}%
- High priority tasks created this week: {stats['high_priority_created']}
- High priority tasks completed this week: {stats['high_priority_done']}
- Recurring daily reminders fired this week: {stats['recurring_fired_this_week']}
- Recurring daily reminders actually completed: {stats['recurring_completed_this_week']}
- Recurring completion rate: {stats['recurring_completion_rate']}%

Write a short, warm, encouraging 3-5 sentence summary in Hinglish (Hindi-English
mix, casual tone). Cover both one-time task completion AND recurring habit
consistency (e.g. daily workout) separately since they measure different
things. Compare this week's one-time completion rate to last week. Do not
be judgmental if numbers are low - be supportive and motivating instead."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()
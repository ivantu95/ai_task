# Telegram AI Task Copilot

An AI-powered personal task assistant that lives inside Telegram. Add tasks
by typing or speaking naturally, get reminded automatically, and ask it
questions about what you've done — no rigid commands required.

Built as a portfolio project to demonstrate applied LLM integration, RAG
(retrieval-augmented generation), scheduling/automation, and multi-user
system design — not just a CRUD app with a chatbot wrapper.

---

## Features

### Natural language task management
- Add tasks by typing **or speaking** ("напомни отправить отчет завтра в 18:00") — voice notes are transcribed via Groq Whisper (`whisper-large-v3`) and processed dynamically.
- A single LLM call classifies **intent** first (add a task? view a list? mark something done? search? ask for insights?) and extracts structured fields — so every feature works via natural language, not just slash commands.

### Smart Voice Dictaphone (AudioPen / Oasis style)
- **Умный диктофон мыслей**: при записи длинных голосовых сообщений (>18 сек или от 25 слов), бот распознает их как надиктованный поток мыслей или конспект встречи.
- **Очистка от мусора**: AI автоматически удаляет оговорки, слова-паразиты («эээ», «ну», «как бы», «короче») и повторы.
- **Структурированный конспект**: выделяет емкий заголовок (title), суть в 1-2 предложениях (summary) и ключевые тезисы списком (key_points).
- **Автоизвлечение задач и напоминаний**: из монолога вычленяются все конкретные поручения, каждому назначается точный дедлайн, и они автоматически ставятся в базу с напоминаниями.
- Также доступно для длинного текста через команду `/note <текст>` или `/memo <текст>`.

### Live Interactive Checklist & Categories
- **Live Checklist (`/list`)**: интерактивный список с инлайн-кнопками (`[1. ✅]`), позволяющий вычеркивать задачи прямо в сообщении в 1 клик без спама в чат.
- **Категории (`#покупки`, `#работа`, `#личное`, `#идеи`)**: задачи автоматически классифицируются, между ними можно переключаться фильтрами прямо в чек-листе.
- **Специализированный режим покупок (`/shopping`, `/buy`)**: удобный отдельный список продуктов и покупок с мгновенным вычеркиванием.

### Reminders & Proactive Assistant
- **One-time reminders** с поддержкой относительного времени («через 15 минут», «в эту пятницу в 11 утра») и таймзон.
- **Интерактивные кнопки под напоминаниями**: при срабатывании бот сразу предлагает кнопки `[✅ Выполнено]`, `[⏰ +15 мин]`, `[⏰ +1 час]`, `[📅 На завтра]`, `[🗑 Удалить]`.
- **Контрольный переспрос (Follow-up)**: если задача не была отмечена выполненной, через 45 минут бот аккуратно переспросит, удалось ли завершить дело.
- **Вечерний обзор (Evening Wind-Down в 21:00)**: бот подводит итоги дня, хвалит за выполненное и предлагает кнопку переноса незакрытых задач на завтра.
- **Recurring daily reminders** с исключениями дней недели («каждый день кроме воскресенья в 9:00»).
- **Priority detection** — срочные задачи получают статус high priority и маркер 🔴 вверху списков.

### Search & history (RAG)
- **Fast / Semantic search**:
  - **Легковесный режим (по умолчанию, `USE_LOCAL_EMBEDDINGS=false`)**: бот потребляет всего **~30 МБ RAM** и **0% CPU**. Поиск по активным задачам выполняется быстрым ранжированием по словам, а концептуальные вопросы по истории («что я делал на прошлой неделе по налогам») анализируются облачной нейросетью LLaMA через Groq API без нагрузки на сервер.
  - **Локальный режим (`USE_LOCAL_EMBEDDINGS=true`)**: оригинальная реализация с вычислением локальных векторных эмбеддингов через `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS. Код сохранен в проекте и включается при необходимости в `.env` (требует около 350-400 МБ RAM).
- **History search**: past tasks aren't deleted, they're archived. Ask
  natural questions like *"what did I do last week about finance?"* and
  the bot retrieves relevant archived tasks and generates a natural-language
  answer — a full retrieval-augmented generation pipeline, not just a list.

### Insights
- **Weekly AI-generated summary** (on-demand via a command, or automatically
  every Sunday): completion rate this week vs last week, and recurring
  habit consistency (e.g. "workout completed 5 of 7 days") — turned into a
  natural-language, encouraging summary by the LLM instead of raw numbers.

### Multi-user
- Any number of people can use the same bot. Tasks, reminders, and insights
  are fully isolated per Telegram user (`chat_id`-scoped).

---

## Architecture

```
User (text or voice) on Telegram
        |
        v
Telegram long polling (no webhook, no public URL needed)
        |
        v
Intent classification + parsing (Groq LLaMA 3.1) -----> Voice transcription
        |                                                (Groq Whisper, if voice)
        v
Route by intent:
  add_task / snooze / mark_done  --> SQLite (tasks, recurring_logs, users)
  search_task                    --> semantic_search() [HuggingFace + FAISS]
  search_history                 --> semantic_search() + generate_history_answer() [RAG]
  show_insights                  --> weekly stats query + generate_weekly_insight()
        |
        v
Reply sent back to user via Telegram
        |
        v
JobQueue (APScheduler under the hood) fires reminders at scheduled times,
independent of any incoming message
```

---

## Tech stack

| Purpose | Tool |
|---|---|
| Bot framework | `python-telegram-bot` (long polling + JobQueue) |
| NLP / intent parsing | Groq API — LLaMA 3.1 8B Instant |
| Speech-to-text | Groq API — Whisper large-v3 |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector search | FAISS (in-memory, built fresh per query) |
| Storage | SQLite (single-file, auto-migrating schema) |
| Scheduling | APScheduler (via JobQueue), timezone-aware (IST) |

---

## Setup

### 1. Create your bot
Message **@BotFather** on Telegram, send `/newbot`, follow the prompts, and
save the token it gives you.

### 2. Get a Groq API key
Sign up free at [console.groq.com](https://console.groq.com) and generate
an API key.

### 3. Install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Note: `sentence-transformers` downloads a small model (~90MB) on first run.

### 4. Configure environment
```bash
cp .env.example .env
# fill in TELEGRAM_BOT_TOKEN, GROQ_API_KEY, REMINDER_TIME
```

### 5. Find your chat ID (one-time)
Message your bot once on Telegram, then:
```bash
python get_chat_id.py
```

### 6. Run
```bash
python -m src.telegram_bot
```
No ngrok, no webhook, no public URL required — the bot polls Telegram
directly.

---

## Commands (all also work as natural language / voice)

| Command | What it does |
|---|---|
| `/start` | Register and get a quick intro |
| *(just type/speak)* | Add a task, e.g. "remind me to call mum tomorrow 6pm" |
| `/list` | Show pending tasks |
| `/bydate` | Show pending tasks grouped by date |
| `/today` | Show tasks due today |
| `/done <n>` | Mark task *n* complete |
| `/recurring` | Show active recurring reminders |
| `/stoprecurring <n>` | Stop recurring reminder *n* |
| `/snooze <minutes>` | Delay the most recently fired reminder |
| `/search <query>` | Semantic search over pending tasks |
| `/history <query>` | Ask about past/completed tasks (RAG) |
| `/insights` | Get this week's AI-generated summary |

---

## Database schema

**tasks**: `id, chat_id, text, due_time, status (pending/done/fired),
created_at, is_recurring, recurring_time, exclude_days, priority,
completed_at, snooze_count`

**users**: `chat_id, created_at`

**recurring_logs**: `id, task_id, chat_id, log_date, completed, created_at`
— one row per day a recurring reminder fires, used to compute habit
consistency.

Schema migrates automatically on startup (`init_db()` checks for missing
columns and adds them) — no manual migration scripts needed.

---

## Design decisions worth knowing

- **Long polling over webhooks**: no public URL/ngrok needed, unlike a
  Twilio/WhatsApp-based version of this same bot — simpler for local
  development and personal hosting.
- **In-memory FAISS index, rebuilt per query** instead of a persisted
  index: at personal-project scale (tens to low hundreds of tasks), this
  avoids index-sync complexity (no need to update the index every time a
  task is added or completed) while still using the real embedding +
  vector-search pipeline.
- **Archive, don't delete**: one-time reminders used to be hard-deleted
  once they fired. They're now soft-archived (`status='fired'`) so history
  search has something to retrieve — a small schema change that unlocked
  the RAG feature.
- **SQLite over a client-server DB**: appropriate for personal/small-scale
  use; would move to Postgres + a dedicated task queue (Celery/Redis) to
  support many concurrent users at scale.

## Known limitations
- In-memory state (`_last_list_mapping`, `_last_fired`, etc.) resets on
  restart — fine for a personal bot, would need Redis for a
  multi-instance production deployment.
- Groq's free tier has rate limits; adequate for personal use.
- Semantic search quality depends on the small `all-MiniLM-L6-v2` model —
  good enough for short task descriptions, not tuned for longer documents.
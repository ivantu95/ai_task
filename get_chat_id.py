"""
Run this ONCE, after you've sent any message (e.g. "hi") to your bot on
Telegram, to find your chat_id. Put that value in your .env file.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.environ["TELEGRAM_BOT_TOKEN"]
resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates").json()

if not resp.get("result"):
    print("No messages found yet.")
    print("Send any message to your bot on Telegram first, then rerun this script.")
else:
    chat_id = resp["result"][-1]["message"]["chat"]["id"]
    print(f"Your chat_id is: {chat_id}")
    print("Add this to your .env file as TELEGRAM_CHAT_ID")

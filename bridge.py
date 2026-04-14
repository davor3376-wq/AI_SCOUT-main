import os
import httpx
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# 1. Load your credentials
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
_chat_id_raw = os.getenv("TELEGRAM_CHAT_ID")
MY_ID = int(_chat_id_raw) if _chat_id_raw else 0
JULES_KEY = os.getenv("JULES_API_KEY")

# 2. Your specific Repo ID
SOURCE_NAME = "sources/github/davor3376-wq/AI_SCOUT"

logging.basicConfig(level=logging.INFO)

async def scout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Only listen to YOU
    if update.effective_chat.id != MY_ID:
        return
    
    user_prompt = " ".join(context.args)
    if not user_prompt:
        await update.message.reply_text("💡 Usage: /scout [your instruction]")
        return

    await update.message.reply_text(f"⏳ Sending task to Jules: '{user_prompt}'...")

    # Jules API Request
    url = "https://jules.googleapis.com/v1alpha/sessions"
    headers = {
        "X-Goog-Api-Key": JULES_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": user_prompt,
        "sourceContext": {
            "source": SOURCE_NAME,
            "githubRepoContext": {"startingBranch": "main"}
        },
        "title": f"Mobile Prompt: {user_prompt[:30]}"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                session_id = data.get("id")
                await update.message.reply_text(
                    f"✅ Jules started! Session ID: \n`{session_id}`\n\n"
                    f"Check your repo for the new branch soon!"
                )
            else:
                await update.message.reply_text(f"❌ Jules Error {response.status_code}: {response.text}")
        except Exception as e:
            await update.message.reply_text(f"❌ PC Error: {str(e)}")

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler('scout', scout))
    print("🚀 AI Scout Bridge is active. Waiting for Telegram messages...")
    application.run_polling()
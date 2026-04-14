import os
import asyncio
import logging
import threading
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger("Notifications")

_TELEGRAM_TIMEOUT = 10  # seconds

async def send_telegram_alert(message: str):
    """
    Sends a Telegram message to the configured chat ID.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.warning("Telegram configuration missing. Skipping alert.")
        return

    try:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=message, read_timeout=_TELEGRAM_TIMEOUT, write_timeout=_TELEGRAM_TIMEOUT)
        logger.info(f"Telegram alert sent: {message}")
    except TelegramError as e:
        logger.error(f"Failed to send Telegram alert: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending Telegram alert: {e}")

def send_alert_sync(message: str):
    """
    Fire-and-forget alert. Never blocks the calling thread.
    """
    try:
        loop = asyncio.get_running_loop()
        # In an async context — schedule without waiting
        loop.create_task(send_telegram_alert(message))
    except RuntimeError:
        # Worker thread — run in an isolated daemon thread so the mission
        # executor is not blocked waiting for a Telegram round-trip.
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(send_telegram_alert(message))
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        threading.Thread(target=_run, daemon=True).start()

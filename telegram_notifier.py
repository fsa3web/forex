import logging
import asyncio
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, token):
        self.bot = Bot(token)

    async def send_message(self, chat_id, message):
        try:
            logger.info(f"Attempting to send message to chat_id: {chat_id}")
            logger.debug(f"Message content: {message}")
            await self.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Message sent successfully to chat_id: {chat_id}")
        except TelegramError as e:
            logger.error(f"Failed to send message to chat_id: {chat_id}. Error: {str(e)}")
            logger.exception("Detailed error information:")
        except Exception as e:
            logger.error(f"Unexpected error while sending message to chat_id: {chat_id}. Error: {str(e)}")
            logger.exception("Detailed error information:")

    async def send_signal_notification(self, chat_id, pair, signal):
        message = f"📊 New Trading Signal for {pair} 📊\n\n"
        message += f"Action: {signal['action']}\n"
        message += f"Price: {signal['price']:.5f}\n"
        message += f"Stop Loss: {signal['stop_loss']:.5f}\n"
        message += f"Take Profit 1: {signal['tp1']:.5f}\n"
        message += f"Take Profit 2: {signal['tp2']:.5f}\n"
        message += f"Take Profit 3: {signal['tp3']:.5f}\n"
        message += f"Probability: {signal['probability']:.2f}"

        await self.send_message(chat_id, message)

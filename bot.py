import time
import logging
from aiogram import Bot, Dispatcher, executor, types
from utils import MODEL_NAME
from tg_token import TOKEN
from simplifier import Simplifier

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)
simplifier = Simplifier(MODEL_NAME)
logging.info(f'Model loaded successfully! {time.asctime()}')

@dp.message_handler(commands=["start"])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    user_name = message.from_user.first_name
    user_full_name = message.from_user.full_name
    logging.info(f'{user_id} {user_full_name} {time.asctime()}')
    await message.reply(f"Привет, я бот-упрощатель! Пришли мне текст, а я сделаю его более простым и понятным:)")

@dp.message_handler(content_types=['text'])
async def simplify(message: types.Message):
    text = message.text
    user_id = message.from_user.id
    logging.info(f'User {user_id} sent text: "{text}" {time.asctime()}')
    simplified_text = simplifier.simplify(text)
    logging.info(f'Text simplified: "{simplified_text}" {time.asctime()}')
    logging.info(f'{simplified_text}')
    await message.reply(f"{simplified_text}")

if __name__ == "__main__":
    executor.start_polling(dp)
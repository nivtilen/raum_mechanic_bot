from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from emb_utils import generate_answer
import pandas as pd

router = Router()

df = pd.read_json("embeddings.jsonl", lines=True)

@router.message(Command("start"))
async def cmd_start(message: Message):
    start_text = (
        "Привет! Я бот, отвечающий на вопросы по базе знаний об обслуживании автомобилей Toyota Raum.\n\n"
        "Просто задай вопрос, например:\n"
        "- Почему двигатель теряет мощность?\n"
        "- Какое масло заливать зимой в W211?\n\n"
        "Для подробной информации используй команду /help."
    )
    await message.answer(start_text)

@router.message(Command("help"))
async def cmd_help(message: Message):
    topic = "Тематика: Автомобильный форум по решению технических пробем с автомобилем Toyota Raum (1997-2003)."
    chunks_count = len(df)
    example_chunk = df['chunk'].iloc[0]
    help_text = "Задавайте любой вопрос по техническим проблемам с автомобилем и я отвечу на него, основываясь на информации из автомобильного форума.\n\n"

    await message.answer(
        f"{topic}\n\n"
        f"Количество текстов в базе: {chunks_count}\n\n"
        f"Пример текста:\n{example_chunk}\n\n"
        f"{help_text}"
    )


@router.message(F.text)
async def handle_text(message: Message):
    loading_message = await message.answer("Бот обрабатывает ваш запрос, это может занять некоторое время...")

    answer, textes = generate_answer(
        message.text,
        df
    )


    await loading_message.edit_text(answer)
    await message.answer("Релевантные тексты:\n" + "\n".join(textes[:5]))
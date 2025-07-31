import pandas as pd
from scipy.spatial.distance import cosine
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
keys = os.getenv("GEMINI_KEYS").split(",") if os.getenv("GEMINI_KEYS") else []

MODEL = "gemini-embedding-001"
FLASH_MODEL = "gemini-2.5-flash-lite-preview-06-17"
client = genai.Client(api_key=keys[0])

def get_top_similar_texts(
        query: str,
        df: pd.DataFrame,
        relatedness_fn = lambda x, y: 1 - cosine(x, y),
        top_n: int = 100,
) -> tuple[list[str], list[float]]:
    """
    Находит наиболее релевантные тексты из DataFrame по эмбеддингу запроса.

    Args:
        query (str): Запрос для сравнения.
        df (pd.DataFrame): DataFrame с колонками 'chunk' и 'embedding'.
        relatedness_fn (callable): Функция для вычисления схожести между векторами.
        top_n (int): Количество строк для возврата.

    Returns:
        tuple[list[str], list[float]]: Список строк и соответствующих им значений схожести.
    """
    query_embedding = client.models.embed_content(
        model=MODEL,
        contents=query
    )
    query_embedding = query_embedding.embeddings[0].values
    relatednesses = df['embedding'].apply(lambda x: relatedness_fn(query_embedding, x))
    top_indices = relatednesses.nlargest(top_n).index
    return df.loc[top_indices, 'chunk'].tolist(), relatednesses.loc[top_indices].tolist()

def build_rag_prompt(
        query: str,
        df: pd.DataFrame,
        token_budget: int = 3000
) -> tuple[str, list[str]]:
    """
    Создает промпт для модели на основе запроса и DataFrame.

    Args:
        query (str): Запрос для создания промпта.
        df (pd.DataFrame): DataFrame с колонками 'chunk' и 'embedding'.
        token_budget (int): Максимальное количество токенов в промпте.

    Returns:
        str: Сформированный промпт.
        list[str]: Список текстов, на основе которых был сформирован промпт.
    """
    top_texts, relatedness = get_top_similar_texts(query, df, top_n=50)
    system_prompt = "Ты помогаешь пользователям находить решения технических проблем с автомобилями. Ответ должен основываться только на информации из автомобильного форума, предоставленной в контексте."
    prompt = f"Вопрос: {query}\n\nКонтекст:\n"

    related_texts = []

    for text, rel in zip(top_texts, relatedness):
        print (f"Text: {text[:50]}... | Relatedness: {rel:.4f}")
        if rel > 0.7:
            related_texts.append(text)

    prompt = system_prompt + prompt + "\n\n"
    
    for i, text in enumerate(related_texts):
        if client.models.count_tokens(
            model=FLASH_MODEL,
            contents=prompt + f"{i + 1}. {text}\n"
        ).total_tokens > token_budget:
            break
        prompt += f"{i + 1}. {text}\n"

    return prompt + "\nДай краткий и точный ответ, основываясь только на контексте. Не добавляй ничего не касающегося вопроса.", related_texts

def generate_answer(
        query: str,
        df: pd.DataFrame,
        model: str = FLASH_MODEL,
        token_budget: int = 10000,

) -> tuple[str, list[str]]:
    """
    Генерирует ответ на основе запроса и DataFrame.

    Args:
        query (str): Запрос для генерации ответа.
        df (pd.DataFrame): DataFrame с колонками 'text' и 'embedding'.
        token_budget (int): Максимальное количество токенов в промпте.
        model (str): Модель для генерации ответа.

    Returns:
        str: Сгенерированный ответ.
        list[str]: Список текстов, на основе которых был сгенерирован ответ.
    """
    prompt, top_texts = build_rag_prompt(query, df, token_budget)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text.strip() if response and response.text else "Нет ответа.", top_texts
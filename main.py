import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Booking AI Prototype")

CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "").strip()
SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()
MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max").strip()
PASSPORT_LINK = "https://example.com/passport"

giga = GigaChat(credentials=CREDENTIALS, scope=SCOPE, verify_ssl_certs=False)

SYSTEM_PROMPT_TEMPLATE = """Ты — дружелюбный ИИ-ассистент сервиса бронирования жилья.
Отвечай на русском языке, кратко (2–4 предложения), вежливо и по делу.

Текущий статус бронирования гостя: {status}.

Правила:
1. Если паспорт НЕ получен и гость спрашивает про заселение, заезд, ключи или что ему делать — объясни, что сначала необходимо предоставить паспорт, и обязательно добавь ссылку для загрузки паспорта дословно, без пробелов и переносов внутри неё: {link}
2. Если паспорт получен и гость спрашивает, что делать дальше или какие следующие шаги — подтверди, что паспорт принят, и сообщи, что следующим этапом будет оплата залога.
3. Не выдумывай другие этапы, цены и сроки. Ссылку {link} давай только в случае из правила 1."""


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    passport_received: bool


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path(__file__).parent.joinpath("index.html").read_text(encoding="utf-8")


@app.post("/api/ask")
async def ask(req: AskRequest):
    if not CREDENTIALS:
        raise HTTPException(
            status_code=500, detail="В .env не задан GIGACHAT_CREDENTIALS"
        )

    status = "паспорт получен" if req.passport_received else "паспорт НЕ получен"
    system_text = SYSTEM_PROMPT_TEMPLATE.format(status=status, link=PASSPORT_LINK)

    try:
        response = giga.chat(
            Chat(
                model=MODEL,
                temperature=0.4,
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=system_text),
                    Messages(role=MessagesRole.USER, content=req.message),
                ],
            )
        )
        answer = response.choices[0].message.content
    except Exception as e:  # noqa: BLE001 — ловим всё сознательно: любой сбой LLM должен стать аккуратным 502
        raise HTTPException(status_code=502, detail=f"Ошибка GigaChat API: {e}")

    # страховка: если модель вдруг не вставила ссылку при неотправленном паспорте
    if not req.passport_received and PASSPORT_LINK not in answer:
        answer += f"\nСсылка для загрузки паспорта: {PASSPORT_LINK}"

    return {
        "reply": answer,
        "model": MODEL,
        "passport_received": req.passport_received,
    }

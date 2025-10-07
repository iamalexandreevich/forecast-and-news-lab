"""Advanced Telegram parsing pipeline built on Telethon."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sqlite3
import yaml
from io import BytesIO
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from pydantic import ValidationError
from telethon import TelegramClient
from telethon.errors import (
    ChannelPrivateError,
    FloodWaitError,
    RPCError,
    UsernameInvalidError,
)

from modules.schemas import ChannelConfig, TelegramMessage
from modules.storage import load_parquet, save_parquet


LOGGER = logging.getLogger(__name__)

DATA_DIR = Path("data/telegram")
RAW_DIR = DATA_DIR / "raw"
STATE_PATH = DATA_DIR / "state.json"
SESSION_NAME = DATA_DIR / "stock_forecasting_session"
LATEST_PARQUET = DATA_DIR / "latest.parquet"

DEFAULT_MAX_CONCURRENCY = 6
FLOOD_RETRY_ATTEMPTS = 3

LINK_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@([\w\d_]{4,32})")
HASHTAG_PATTERN = re.compile(r"#([\w\d_]{2,64})")
TICKER_PATTERN = re.compile(r"\b\$?[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b", re.IGNORECASE)


def _ensure_logger_configured() -> None:
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _unique_preserve(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_links_mentions_hashtags(text: Optional[str]) -> tuple[list[str], list[str], list[str]]:
    """Extract links, mentions and hashtags from the supplied text."""

    if not text:
        return [], [], []

    links = _unique_preserve(LINK_PATTERN.findall(text))
    mentions = [f"@{match}" for match in MENTION_PATTERN.findall(text)]
    hashtags = [f"#{match}" for match in HASHTAG_PATTERN.findall(text)]
    return links, _unique_preserve(mentions), _unique_preserve(hashtags)


def extract_tickers(text: Optional[str], stock_symbols: Optional[Sequence[str]] = None) -> list[str]:
    """Extract ticker symbols from text with optional allow-list filtering."""

    if not text:
        return []

    raw_matches = [match.lstrip("$") for match in TICKER_PATTERN.findall(text)]
    tickers = [match.upper() for match in raw_matches if match]

    if stock_symbols:
        allow = {symbol.strip().upper() for symbol in stock_symbols if symbol and symbol.strip()}
        if not allow:
            return _unique_preserve(tickers)
        tickers = [ticker for ticker in tickers if ticker in allow]

    return _unique_preserve(tickers)


def make_msg_hash(channel_username: str, dt_utc: datetime, text: str) -> str:
    """Predictable sha256 hash combining channel, timestamp and text."""

    import hashlib

    payload = f"{channel_username}|{dt_utc.isoformat()}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_state(path: Path = STATE_PATH) -> dict[str, int]:
    """Load per-channel watermark state."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: int(v) for k, v in raw.items()}


def save_state(path: Path, state: dict[str, int]) -> None:
    """Persist per-channel watermark state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2, sort_keys=True)


async def _login_with_phone(client: TelegramClient, phone: str, ui=None) -> None:
    """
    Авторизация через номер телефона с кодом из Telegram.
    При наличии 2FA просто пропускает запрос пароля.
    """
    # Отправляем код на телефон только если еще не отправляли
    if ui is None or 'telegram_phone_code_hash' not in ui.session_state:
        if ui is not None:
            LOGGER.info("[AUTH] Отправка кода на телефон %s", phone)
            ui.info(f"📱 Отправка кода подтверждения на {phone}...")

        sent_code = await client.send_code_request(phone)
        phone_code_hash = sent_code.phone_code_hash

        if ui is not None:
            ui.session_state.telegram_phone_code_hash = phone_code_hash
            LOGGER.info("[AUTH] Код отправлен, phone_code_hash сохранен")
        else:
            # Для CLI режима сохраняем в глобальной переменной (временное решение)
            globals()['_phone_code_hash'] = phone_code_hash
    else:
        if ui is not None:
            phone_code_hash = ui.session_state.telegram_phone_code_hash
            LOGGER.info("[AUTH] Используем сохраненный phone_code_hash")
        else:
            phone_code_hash = globals().get('_phone_code_hash')

    if ui is not None:
        # Проверяем, есть ли уже введенный код в session_state
        if 'telegram_entered_code' not in ui.session_state:
            ui.session_state.telegram_entered_code = ""
            LOGGER.info("[AUTH] Инициализирован telegram_entered_code")

        LOGGER.info("[AUTH] Текущий telegram_entered_code: '%s'", ui.session_state.telegram_entered_code)

        # Используем форму для предотвращения автоматического rerun
        with ui.form("telegram_code_form", clear_on_submit=False):
            code_input = ui.text_input(
                "Введите код из Telegram:",
                value=ui.session_state.telegram_entered_code,
                key="telegram_code_field",
                type="default",
                help="Введите код, полученный в Telegram"
            )
            submit_code = ui.form_submit_button("✅ Отправить код")

            if submit_code:
                LOGGER.info("[AUTH] Кнопка 'Отправить код' нажата, code_input='%s'", code_input)
                if code_input and code_input.strip():
                    ui.session_state.telegram_entered_code = code_input.strip()
                    LOGGER.info("[AUTH] Код сохранен в session_state: '%s'", ui.session_state.telegram_entered_code)
                else:
                    LOGGER.warning("[AUTH] Код пустой или содержит только пробелы")

        code = ui.session_state.telegram_entered_code

        if not code or not code.strip():
            LOGGER.info("[AUTH] Код не введен, ожидаем ввода...")
            ui.warning("⏳ Ожидание ввода кода...")
            raise RuntimeError("Waiting for code input")

        LOGGER.info("[AUTH] Попытка входа с кодом: '%s'", code)
    else:
        code = input("Введите код из Telegram: ")

    try:
        # Пробуем войти с кодом
        LOGGER.info("[AUTH] Вызов client.sign_in с телефоном %s, кодом и phone_code_hash", phone)
        await client.sign_in(phone, code.strip(), phone_code_hash=phone_code_hash)
        LOGGER.info("[AUTH] ✅ Успешная авторизация!")

        # Очищаем флаги после успешной авторизации
        if ui is not None:
            if 'telegram_entered_code' in ui.session_state:
                del ui.session_state.telegram_entered_code
            if 'telegram_phone_code_hash' in ui.session_state:
                del ui.session_state.telegram_phone_code_hash
            LOGGER.info("[AUTH] Флаги авторизации очищены после успешного входа")
    except Exception as e:
        LOGGER.error("[AUTH] ❌ Ошибка при авторизации: %s (%s)", type(e).__name__, str(e))

        # Если код неверный, очищаем его для повторного ввода
        error_msg = str(e).lower()
        if "phone_code_invalid" in error_msg or "code invalid" in error_msg:
            LOGGER.warning("[AUTH] Неверный код, очищаем для повторного ввода")
            if ui is not None:
                ui.error("❌ Неверный код. Попробуйте снова.")
                if 'telegram_entered_code' in ui.session_state:
                    ui.session_state.telegram_entered_code = ""
            raise RuntimeError("Invalid code, please try again")
        # Если требуется 2FA
        if "password" in error_msg or "2fa" in error_msg or "two-step" in error_msg:
            LOGGER.error("[AUTH] Требуется 2FA, но пароль не поддерживается")
            if ui is not None:
                ui.error("❌ Включена двухфакторная аутентификация (2FA). Невозможно войти без пароля.")
                ui.info("💡 Решение: Временно отключите 2FA в Telegram (Settings → Privacy and Security → Two-Step Verification), затем повторите попытку.")
            else:
                print("❌ 2FA включена. Отключите её в настройках Telegram и попробуйте снова.")
            raise RuntimeError("2FA enabled, cannot login without password")
        raise


async def initialize_client(api_id: int, api_hash: str, phone: Optional[str], ui=None) -> TelegramClient:
    """Create and authorize a Telethon client using a local session file."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session_name = str(SESSION_NAME)

    LOGGER.info("[CLIENT] Создание клиента с session_name=%s", session_name)

    # Configure SQLite to use WAL mode for better concurrency handling
    # This prevents "database is locked" errors when fetching multiple channels
    # Create a connection to configure the database before Telethon uses it
    db_path = f"{session_name}.session"
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.close()

    client = TelegramClient(session_name, api_id, api_hash)
    await client.connect()
    LOGGER.info("[CLIENT] Клиент подключен")

    # если уже авторизован — сразу возвращаем
    is_authorized = await client.is_user_authorized()
    LOGGER.info("[CLIENT] Проверка авторизации: %s", is_authorized)

    if is_authorized:
        LOGGER.info("[CLIENT] ✅ Пользователь уже авторизован, возвращаем клиент")
        return client

    # не авторизован — проверяем наличие телефона
    LOGGER.info("[CLIENT] Пользователь не авторизован, требуется вход")
    if not phone:
        LOGGER.error("[CLIENT] ❌ Номер телефона не указан")
        if ui is not None:
            ui.error("❌ Требуется номер телефона для первой авторизации")
            ui.info("💡 Укажите телефон в настройках боковой панели")
        raise RuntimeError("Phone number required for first authorization")

    # пытаемся войти через телефон + код из Telegram
    LOGGER.info("[CLIENT] Начинаем процесс авторизации через телефон")
    await _login_with_phone(client, phone, ui=ui)
    LOGGER.info("[CLIENT] ✅ Авторизация завершена, возвращаем клиент")
    return client


def _extract_forward_from(msg) -> Optional[str]:
    fwd = getattr(msg, "fwd_from", None)
    if not fwd:
        return None
    if getattr(fwd, "from_name", None):
        return fwd.from_name
    if getattr(fwd, "from_id", None):
        return str(fwd.from_id)
    if getattr(fwd, "peer", None):
        return str(fwd.peer)
    return None


def _extract_replies_count(msg) -> Optional[int]:
    replies = getattr(msg, "replies", None)
    if not replies:
        return None
    return getattr(replies, "replies", None)


async def fetch_channel_messages(
    client: TelegramClient,
    channel: ChannelConfig,
    since_id: Optional[int],
    since_days: int,
) -> list[TelegramMessage]:
    """Fetch recent messages for a single channel."""

    username = channel.normalized_username
    LOGGER.info("[FETCH] 🔄 Начало сбора из @%s (since_id=%s, days_back=%d)", username, since_id, since_days)

    try:
        LOGGER.info("[FETCH] 📡 Подключение к каналу @%s...", username)
        entity = await client.get_entity(username)
        LOGGER.info("[FETCH] ✅ Канал @%s найден, начинаем чтение сообщений", username)
    except Exception as exc:
        LOGGER.error("[FETCH] ❌ Ошибка подключения к @%s: %s", username, exc)
        raise

    min_id = since_id or 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    collected: list[TelegramMessage] = []

    skipped_old = 0
    skipped_empty = 0
    skipped_already_seen = 0
    message_counter = 0
    first_msg_date = None
    last_msg_date = None

    LOGGER.info("[FETCH] 📖 Начало итерации по сообщениям @%s (min_id=%d, cutoff=%s)", username, min_id, cutoff.isoformat() if since_days else "не задан")

    async for msg in client.iter_messages(entity, min_id=min_id):
        message_counter += 1

        # Логируем прогресс каждые 100 сообщений
        if message_counter % 100 == 0:
            LOGGER.info("[FETCH] 📊 @%s: обработано %d сообщений, собрано %d", username, message_counter, len(collected))

        if msg.id <= min_id:
            skipped_already_seen += 1
            # При движении от новых к старым можно завершить итерацию
            if not since_days:
                LOGGER.info("[FETCH] 🛑 @%s: достигнута граница уже обработанных сообщений (min_id=%d)", username, min_id)
                break
            continue

        msg_date = getattr(msg, "date", None)
        if not msg_date:
            skipped_empty += 1
            continue

        dt_utc = msg_date.astimezone(timezone.utc)

        # Отслеживаем диапазон дат для диагностики
        if first_msg_date is None:
            first_msg_date = dt_utc
        last_msg_date = dt_utc

        if since_days and dt_utc < cutoff:
            skipped_old += 1
            # При движении от новых к старым можно остановиться — всё остальное ещё старше
            LOGGER.info(
                "[FETCH] 🛑 @%s: достигнут предел периода (%s < %s), завершаем загрузку",
                username,
                dt_utc.strftime("%Y-%m-%d %H:%M"),
                cutoff.strftime("%Y-%m-%d %H:%M"),
            )
            break

        text = (getattr(msg, "message", None) or "").strip()
        raw_text = getattr(msg, "raw_text", None)
        if raw_text:
            raw_text = raw_text.strip()

        if not text and not raw_text:
            skipped_empty += 1
            continue

        base_text = text or raw_text or ""
        links, mentions, hashtags = extract_links_mentions_hashtags(base_text)
        tickers = extract_tickers(base_text)

        message = TelegramMessage(
            id=msg.id,
            channel_username=username,
            channel_title=channel.title,
            date_utc=dt_utc,
            text=text or base_text,
            raw_text=raw_text if raw_text != text else None,
            links=links,
            mentions=mentions,
            hashtags=hashtags,
            tickers=tickers,
            is_forwarded=bool(getattr(msg, "fwd_from", None)),
            fwd_from=_extract_forward_from(msg),
            views=getattr(msg, "views", None),
            forwards=getattr(msg, "forwards", None),
            replies=_extract_replies_count(msg),
            msg_hash=make_msg_hash(username, dt_utc, base_text),
        )
        collected.append(message)

    # Вывод диапазона дат для диагностики
    if first_msg_date and last_msg_date:
        LOGGER.info(
            "[FETCH] 📅 @%s: Диапазон дат сообщений: %s → %s",
            username, first_msg_date.strftime("%Y-%m-%d %H:%M"), last_msg_date.strftime("%Y-%m-%d %H:%M")
        )

    LOGGER.info(
        "[FETCH] ✅ @%s: ЗАВЕРШЕНО - обработано всего %d сообщений, собрано %d (пропущено: старые=%d, пустые=%d, уже_обработаны=%d)",
        username, message_counter, len(collected), skipped_old, skipped_empty, skipped_already_seen
    )

    # Дополнительная диагностика если собрано 0
    if len(collected) == 0 and message_counter > 0:
        LOGGER.warning(
            "[FETCH] ⚠️ @%s: Не собрано ни одного сообщения! Проверьте фильтры: days_back=%d, cutoff=%s",
            username, since_days, cutoff.isoformat()[:19] if since_days else "не задан"
        )
        if skipped_old > message_counter * 0.9:
            LOGGER.warning(
                "[FETCH] 💡 @%s: Большинство сообщений (%d из %d) старше %d дней. Попробуйте увеличить days_back",
                username, skipped_old, message_counter, since_days
            )

    return collected


def _resolve_credentials(
    api_id: Optional[int | str],
    api_hash: Optional[str],
    phone: Optional[str],
) -> tuple[int, str, Optional[str]]:
    api_id_value = api_id or os.getenv("TELEGRAM_API_ID")
    if not api_id_value:
        raise ValueError("Telegram api_id is required (argument or TELEGRAM_API_ID env).")

    try:
        api_id_int = int(api_id_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Telegram api_id must be an integer") from exc

    api_hash_value = api_hash or os.getenv("TELEGRAM_API_HASH")
    if not api_hash_value:
        raise ValueError("Telegram api_hash is required (argument or TELEGRAM_API_HASH env).")

    phone_value = phone or os.getenv("TELEGRAM_PHONE")
    return api_id_int, api_hash_value, phone_value


def _normalize_channels(channel_list: Sequence[ChannelConfig | str | dict]) -> list[ChannelConfig]:
    configs: list[ChannelConfig] = []
    for item in channel_list:
        if isinstance(item, ChannelConfig):
            configs.append(item)
            continue
        if isinstance(item, str):
            data = {"username": item.strip()}
        elif isinstance(item, dict):
            data = item
        else:
            raise TypeError(f"Unsupported channel entry type: {type(item)}")
        try:
            cfg = ChannelConfig(**data)
        except ValidationError as exc:
            raise ValueError(f"Invalid channel configuration: {data}") from exc
        configs.append(cfg)
    return configs


def _persist_results(raw_df: pd.DataFrame) -> None:
    if raw_df.empty:
        return

    today_name = datetime.now(timezone.utc).date().isoformat()
    daily_path = RAW_DIR / f"{today_name}.parquet"

    if daily_path.exists():
        existing = load_parquet(daily_path)
        combined = pd.concat([existing, raw_df], ignore_index=True)
    else:
        combined = raw_df.copy()

    combined = combined.drop_duplicates(subset="msg_hash", keep="last")
    save_parquet(combined, daily_path)

    try:
        latest_existing = load_parquet(LATEST_PARQUET)
        latest_combined = pd.concat([latest_existing, raw_df], ignore_index=True)
    except FileNotFoundError:
        latest_combined = raw_df.copy()

    latest_combined = latest_combined.drop_duplicates(subset="msg_hash", keep="last")
    latest_combined["date_utc"] = pd.to_datetime(latest_combined["date_utc"], utc=True)
    latest_combined = latest_combined.sort_values("date_utc", ascending=True)
    save_parquet(latest_combined, LATEST_PARQUET)


def filter_messages(
    df: pd.DataFrame,
    stock_symbols: Optional[Sequence[str]],
    keywords: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filter dataframe by tickers allow-list and/or keyword matches."""

    if df.empty:
        LOGGER.info("[FILTER] Пустой DataFrame, фильтрация не требуется")
        return df

    original_count = len(df)
    mask = pd.Series(False, index=df.index)

    ticker_matched = 0
    keyword_matched = 0

    if stock_symbols:
        allow = {symbol.strip().upper() for symbol in stock_symbols if symbol and symbol.strip()}
        if allow:
            ticker_mask = df["tickers"].apply(lambda items: bool(set(items) & allow))
            ticker_matched = ticker_mask.sum()
            mask = mask | ticker_mask
            LOGGER.info("[FILTER] По тикерам %s: найдено %d сообщений", list(allow), ticker_matched)

    if keywords:
        keywords_clean = [kw.strip() for kw in keywords if kw and kw.strip()]
        if keywords_clean:
            pattern = "|".join(re.escape(kw) for kw in keywords_clean)
            keyword_mask = df["text"].str.contains(pattern, case=False, na=False)
            keyword_matched = keyword_mask.sum()
            mask = mask | keyword_mask
            LOGGER.info("[FILTER] По ключевым словам (%d шт.): найдено %d сообщений", len(keywords_clean), keyword_matched)

    if not stock_symbols and not keywords:
        LOGGER.info("[FILTER] Фильтры не заданы, возвращаем все %d сообщений", original_count)
        return df

    filtered = df[mask].copy()
    LOGGER.info(
        "[FILTER] Итого: %d → %d сообщений (отфильтровано %d)",
        original_count, len(filtered), original_count - len(filtered)
    )
    return filtered


async def parse_channels(
    channel_cfgs: Sequence[ChannelConfig],
    stock_symbols: Optional[Sequence[str]],
    days_back: int,
    api_id: Optional[int | str] = None,
    api_hash: Optional[str] = None,
    phone: Optional[str] = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    keywords: Optional[Sequence[str]] = None,
    ui=None,
    progress_callback=None,
) -> pd.DataFrame:
    """Fetch, normalize and filter messages for the provided channel configs.

    Args:
        progress_callback: Optional callable(channel_username, status, count, error_msg)
                          where status is one of: 'start', 'complete', 'error', 'waiting'
    """

    _ensure_logger_configured()

    active_channels = [cfg for cfg in channel_cfgs if cfg.enabled]
    if not active_channels:
        LOGGER.info("No enabled channels supplied.")
        return pd.DataFrame(columns=TelegramMessage.model_fields.keys())

    LOGGER.info("[PARSE] Начало обработки %d активных каналов", len(active_channels))

    api_id_int, api_hash_str, phone_value = _resolve_credentials(api_id, api_hash, phone)

    state = load_state(STATE_PATH)
    updated_state = state.copy()

    LOGGER.info("[PARSE] Инициализация Telegram клиента...")
    client = await initialize_client(api_id_int, api_hash_str, phone_value, ui=ui)
    LOGGER.info("[PARSE] Клиент готов, начинаем сбор сообщений")

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[list[TelegramMessage]] = []

    async def process_channel(cfg: ChannelConfig) -> list[TelegramMessage]:
        if not cfg.enabled:
            return []

        username = cfg.normalized_username
        last_id = state.get(username)
        attempt = 0

        LOGGER.info("[PROCESS] 🎯 Начинаем обработку канала @%s", username)

        # Notify start of processing
        if progress_callback:
            progress_callback(username, "start", 0, None)

        while True:
            try:
                LOGGER.info("[PROCESS] 🔓 Получение семафора для @%s", username)
                async with semaphore:
                    LOGGER.info("[PROCESS] ✅ Семафор получен, запускаем fetch для @%s", username)
                    messages = await fetch_channel_messages(client, cfg, last_id, days_back)
                    LOGGER.info("[PROCESS] 📦 Получено %d сообщений из @%s", len(messages), username)
                break
            except FloodWaitError as exc:  # type: ignore[redundant-expr]
                wait_seconds = getattr(exc, "seconds", 5) + random.uniform(0.5, 1.5)
                LOGGER.warning("[PROCESS] ⏰ Flood wait для @%s: ожидание %.1f секунд", username, wait_seconds)

                # Notify waiting status
                if progress_callback:
                    progress_callback(username, "waiting", int(wait_seconds), None)

                attempt += 1
                if attempt >= FLOOD_RETRY_ATTEMPTS:
                    LOGGER.error("[PROCESS] ❌ Превышено число попыток FloodWait для @%s", username)
                    if progress_callback:
                        progress_callback(username, "error", 0, "Exceeded FloodWait retries")
                    return []
                await asyncio.sleep(wait_seconds)
            except (ChannelPrivateError, UsernameInvalidError) as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                LOGGER.error("[PROCESS] ❌ Пропущен канал @%s: %s", username, exc)
                if progress_callback:
                    progress_callback(username, "error", 0, error_msg)
                return []
            except RPCError as exc:
                error_msg = f"RPC error: {exc}"
                LOGGER.error("[PROCESS] ❌ RPC ошибка для @%s: %s", username, exc)
                if progress_callback:
                    progress_callback(username, "error", 0, error_msg)
                return []
            except Exception as exc:  # pylint: disable=broad-except
                error_msg = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("[PROCESS] ❌ Неожиданная ошибка при обработке @%s: %s", username, exc)
                if progress_callback:
                    progress_callback(username, "error", 0, error_msg)
                return []

        if messages:
            updated_state[username] = max(msg.id for msg in messages)
            LOGGER.info("[PROCESS] 💾 Обновлен state для @%s: last_id=%d", username, updated_state[username])

        # Notify completion
        if progress_callback:
            progress_callback(username, "complete", len(messages), None)

        LOGGER.info("[PROCESS] ✅ Обработка канала @%s завершена: %d сообщений", username, len(messages))
        return messages

    try:
        LOGGER.info("[PARSE] 🚀 Запуск параллельного сбора из %d каналов (max_concurrency=%d)", len(active_channels), max_concurrency)
        gather_results = await asyncio.gather(*(process_channel(cfg) for cfg in active_channels))
        results.extend(gather_results)
        LOGGER.info("[PARSE] ✅ Все каналы обработаны, получено %d результатов", len(results))
    finally:
        LOGGER.info("[PARSE] 🔌 Отключение клиента...")
        await client.disconnect()
        LOGGER.info("[PARSE] ✅ Клиент отключен")

    LOGGER.info("[PARSE] 📊 Объединение результатов...")
    all_messages: list[TelegramMessage] = [msg for channel_msgs in results for msg in channel_msgs]
    if not all_messages:
        LOGGER.warning("[PARSE] ⚠️ Не собрано ни одного сообщения из всех каналов")
        save_state(STATE_PATH, updated_state)
        return pd.DataFrame(columns=TelegramMessage.model_fields.keys())

    LOGGER.info("[PARSE] 📦 Всего собрано %d сообщений из %d каналов", len(all_messages), len(active_channels))

    LOGGER.info("[PARSE] 🔄 Создание DataFrame и удаление дубликатов...")
    raw_df = pd.DataFrame([msg.model_dump() for msg in all_messages])
    raw_df = raw_df.drop_duplicates(subset="msg_hash", keep="last")
    LOGGER.info("[PARSE] ✅ После удаления дубликатов: %d сообщений", len(raw_df))

    LOGGER.info("[PARSE] 📅 Преобразование дат и сортировка...")
    raw_df["date_utc"] = pd.to_datetime(raw_df["date_utc"], utc=True)
    raw_df.sort_values(["date_utc", "channel_username", "id"], inplace=True)

    LOGGER.info("[PARSE] 💾 Сохранение state и результатов...")
    save_state(STATE_PATH, updated_state)
    _persist_results(raw_df)
    LOGGER.info("[PARSE] ✅ Результаты сохранены в data/telegram/")

    LOGGER.info("[PARSE] 🔍 Начинаем фильтрацию по тикерам и ключевым словам...")
    filtered = filter_messages(raw_df, stock_symbols=stock_symbols, keywords=keywords)
    filtered.reset_index(drop=True, inplace=True)
    LOGGER.info("[PARSE] ✅ Парсинг завершён: %d сообщений готовы к анализу (из %d собранных)", len(filtered), len(raw_df))
    return filtered


def _parse_symbols_argument(arg: Optional[str]) -> list[str]:
    if not arg:
        return []
    return [symbol.strip().upper() for symbol in arg.split(",") if symbol.strip()]


def _parse_yaml_channels(path: Path) -> list[ChannelConfig]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    channels_data = data.get("channels", data)  # allow list or dict with key
    if not isinstance(channels_data, list):
        raise ValueError("channels.yml должен содержать список каналов")
    return [ChannelConfig(**item) for item in channels_data]


def _parse_yaml_keywords(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    keywords = data.get("keywords", data)
    if isinstance(keywords, list):
        return [str(kw) for kw in keywords]
    raise ValueError("keywords.yml должен содержать список keywords")


def run_telegram_parser(
    channel_list: Sequence[ChannelConfig | str | dict],
    stock_symbols: Sequence[str],
    days_back: int,
    api_id: Optional[int | str] = None,
    api_hash: Optional[str] = None,
    phone: Optional[str] = None,
    *,
    keywords: Optional[Sequence[str]] = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ui=None,
    progress_callback=None,
) -> pd.DataFrame:
    """Synchronous wrapper returning a filtered DataFrame for Streamlit/CLI."""

    channel_cfgs = _normalize_channels(channel_list)
    return asyncio.run(
        parse_channels(
            channel_cfgs=channel_cfgs,
            stock_symbols=stock_symbols,
            days_back=days_back,
            api_id=api_id,
            api_hash=api_hash,
            phone=phone,
            max_concurrency=max_concurrency,
            keywords=keywords,
            ui=ui,
            progress_callback=progress_callback,
        )
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telegram channels parser for stock sentiment analysis.")
    parser.add_argument("--channels", type=str, required=True, help="Path to channels YAML config")
    parser.add_argument("--days-back", type=int, default=7, help="How many days of history to fetch")
    parser.add_argument("--symbols", type=str, default="", help="Comma separated list of stock tickers")
    parser.add_argument("--keywords", type=str, default="", help="Path to keywords YAML (optional)")
    parser.add_argument("--api-id", type=str, default=None, help="Telegram api_id (overrides env)")
    parser.add_argument("--api-hash", type=str, default=None, help="Telegram api_hash (overrides env)")
    parser.add_argument("--phone", type=str, default=None, help="Phone number linked to the Telegram account")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY, help="Max concurrent channel fetches")
    parser.add_argument("--out", type=str, default="", help="Path to save resulting dataframe (csv or parquet)")
    return parser


def _handle_cli_output(df: pd.DataFrame, out_path: str) -> None:
    if not out_path:
        LOGGER.info("Fetched %d messages", len(df))
        return

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        df.to_parquet(target, index=False)
    else:
        df.to_csv(target, index=False)
    LOGGER.info("Saved dataframe to %s", target)


def main(argv: Optional[Sequence[str]] = None) -> None:
    _ensure_logger_configured()
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    channels_path = Path(args.channels)
    channel_cfgs = _parse_yaml_channels(channels_path)

    keywords: Optional[list[str]] = None
    if args.keywords:
        keywords_path = Path(args.keywords)
        keywords = _parse_yaml_keywords(keywords_path)

    symbols = _parse_symbols_argument(args.symbols)

    df = asyncio.run(
        parse_channels(
            channel_cfgs=channel_cfgs,
            stock_symbols=symbols,
            days_back=args.days_back,
            api_id=args.api_id,
            api_hash=args.api_hash,
            phone=args.phone,
            max_concurrency=args.max_concurrency,
            keywords=keywords,
            ui=None,
        )
    )

    if df.empty:
        LOGGER.info("No messages matched provided filters.")
    else:
        LOGGER.info("Collected %d messages from %d channels", len(df), len(channel_cfgs))

    if args.out:
        _handle_cli_output(df, args.out)
    else:
        with pd.option_context("display.max_rows", 10):
            LOGGER.info("%s", df.head(10))


if __name__ == "__main__":
    main()


__all__ = [
    "extract_links_mentions_hashtags",
    "extract_tickers",
    "make_msg_hash",
    "load_state",
    "save_state",
    "initialize_client",
    "fetch_channel_messages",
    "filter_messages",
    "parse_channels",
    "run_telegram_parser",
]

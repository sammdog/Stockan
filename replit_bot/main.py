"""Stockan Telegram display and manual whole-lot allocation."""

import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from threading import Lock

import requests
import telebot
from telebot import types

MSK = timezone(timedelta(hours=3))
CENT = Decimal("0.01")


def parse_capital(text):
    parts = text.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError("Укажи сумму, например /capital 7500")
    amount = parts[1].replace(" ", "").replace(",", ".")
    try:
        capital = Decimal(amount)
    except InvalidOperation as exc:
        raise ValueError("Укажи сумму числом, например /capital 7500,50") from exc
    if not capital.is_finite() or capital <= 0 or capital > Decimal("1000000000000"):
        raise ValueError("Укажи положительную сумму в рублях")
    capital = capital.quantize(CENT, rounding=ROUND_FLOOR)
    if capital == 0:
        raise ValueError("Укажи хотя бы 0,01 ₽")
    return capital


def allocate(report, capital):
    remaining = capital
    cap = Decimal(str(report["trade_notional_cap_rub"]))
    fee = Decimal(str(report["trade_fee_bps"])) / 10_000
    orders = []
    for item in report["trade_candidates"]:
        lot_size = int(item["lot_size"])
        lot_price = Decimal(item["buy_limit"]) * lot_size
        if lot_size <= 0 or lot_price <= 0:
            raise ValueError("В отчёте неверный размер или цена лота")
        lots = min(int(cap // lot_price), int(remaining // (lot_price * (1 + fee))))
        if lots <= 0:
            continue
        notional = lot_price * lots
        total = notional + (notional * fee).quantize(CENT, rounding=ROUND_CEILING)
        if total > remaining:
            lots -= 1
            if lots <= 0:
                continue
            notional = lot_price * lots
            total = notional + (notional * fee).quantize(CENT, rounding=ROUND_CEILING)
        remaining -= total
        orders.append((item, lots, lot_size * lots, total))
    return orders, remaining


def fetch_report(file_id):
    response = requests.get("https://drive.google.com/uc",
                            params={"export": "download", "id": file_id}, timeout=10)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Неверный формат отчёта")
    return data


def format_candidates(report):
    lines = ["Рекомендации Stockan"]
    if "trade_candidates" not in report:
        leaders = sorted(report.get("predictions", {}).items(),
                         key=lambda item: item[1], reverse=True)[:10]
        lines.append("Предыдущий формат отчёта")
        lines.extend(f"• {ticker} · {value:+.1f}%" for ticker, value in leaders)
        return "\n".join(lines)
    lines.append(f"Свеча: {report['signal_date']}")
    candidates = report["trade_candidates"]
    if not candidates:
        lines.append(report.get("trade_plan_block") or "Сегодня нет заявок по критериям стратегии.")
    for item in candidates:
        lines.extend((
            f"• {item['ticker']} · {item['name']}",
            f"  Покупка до {item['buy_limit']} ₽ за акцию · {item['lot_size']} шт. в лоте",
            f"  Оценка заявки {item['expected_order_return_pct']:+.2f}% · тейк +{item['take_profit_pct']:g}% от исполнения",
        ))
    if report.get("missing_trade_metadata"):
        lines.append("Без данных о лоте: " + ", ".join(report["missing_trade_metadata"]))
    return "\n".join(lines)


def build_bot(token, file_id, owner_id):
    bot = telebot.TeleBot(token)
    users_path = Path("users.txt")
    users_lock = Lock()

    @bot.message_handler(commands=["start", "help"])
    def help_command(message):
        try:
            with users_lock:
                users = users_path.read_text(encoding="utf-8").splitlines() if users_path.exists() else []
                if str(message.from_user.id) not in users:
                    with users_path.open("a", encoding="utf-8") as file:
                        file.write(f"{message.from_user.id}\n")
        except OSError:
            logging.exception("Could not update users.txt")
        bot.reply_to(message, "Stockan\n/results — список рекомендаций\n/status — дата отчёта\n"
                              "/capital 7500 — расчёт целых лотов в личном чате владельца.")

    @bot.message_handler(commands=["about"])
    def about(message):
        bot.reply_to(message, "Stockan · Анализ акций Московской биржи. "
                              "Рейтинг по медиане, фильтр ожидаемого результата заявки, тейк +3%.")

    @bot.message_handler(commands=["results", "status"])
    def results(message):
        try:
            report = fetch_report(file_id)
            if message.text.split()[0].split("@")[0] == "/status":
                stamp = datetime.fromisoformat(report["timestamp"].replace("Z", "+00:00")).astimezone(MSK)
                age = (datetime.now(MSK) - stamp).total_seconds() / 3600
                with users_lock:
                    users = len(users_path.read_text(encoding="utf-8").splitlines()) if users_path.exists() else 0
                text = (f"Отчёт: {stamp:%d.%m.%Y %H:%M} МСК · {age:.1f} ч назад\n"
                        f"Свеча: {report.get('signal_date', 'неизвестно')}\nПользователей: {users}")
            else:
                text = format_candidates(report)
            bot.reply_to(message, text)
        except (requests.RequestException, OSError, ValueError, KeyError, TypeError):
            logging.exception("Could not load Stockan report")
            bot.reply_to(message, "Не удалось прочитать отчёт Stockan.")

    @bot.message_handler(commands=["capital"])
    def capital_command(message):
        if message.chat.type != "private" or not owner_id or str(message.from_user.id) != owner_id:
            bot.reply_to(message, "Команда доступна владельцу в личном чате. Список бумаг — /results.")
            return
        try:
            capital = parse_capital(message.text)
        except ValueError as exc:
            bot.reply_to(message, str(exc))
            return
        try:
            report = fetch_report(file_id)
            if "trade_candidates" not in report:
                bot.reply_to(message, "Дождись отчёта в новом формате.")
                return
            stamp = datetime.fromisoformat(report["timestamp"].replace("Z", "+00:00")).astimezone(MSK)
            if stamp.date() != datetime.now(MSK).date():
                bot.reply_to(message, "Для расчёта заявок нужен отчёт, сформированный сегодня по МСК.")
                return
            if report.get("trade_plan_block"):
                bot.reply_to(message, report["trade_plan_block"])
                return
            orders, remaining = allocate(report, capital)
        except (requests.RequestException, ArithmeticError, ValueError, KeyError, TypeError):
            logging.exception("Could not calculate Stockan orders")
            bot.reply_to(message, "Не удалось рассчитать список заявок по этому отчёту.")
            return
        lines = [f"Бюджет: {capital:,.2f} ₽ · свеча: {report['signal_date']}"]
        for item, lots, shares, total in orders:
            lines.append(f"• {item['ticker']} · {item['name']}\n"
                         f"  {lots} лот. ({shares} шт.) × {item['buy_limit']} ₽ за акцию\n"
                         f"  {total:,.2f} ₽ с комиссией · тейк +{item['take_profit_pct']:g}% от исполнения")
        if not orders:
            lines.append("Нет заявок, проходящих фильтр и бюджет на целый лот.")
        lines.append(f"Свободный остаток: {remaining:,.2f} ₽")
        bot.reply_to(message, "\n".join(lines))

    return bot


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot = build_bot(os.environ["TELEGRAM_TOKEN"], os.environ["GOOGLE_DRIVE_FILE_ID"],
                    os.environ.get("YOUR_CHAT_ID", "").strip())
    bot.set_my_commands([
        types.BotCommand("results", "Рекомендации"),
        types.BotCommand("capital", "Рассчитать заявки на сумму"),
        types.BotCommand("status", "Дата отчёта"),
        types.BotCommand("about", "О проекте"),
        types.BotCommand("help", "Справка"),
    ])
    bot.infinity_polling(timeout=30)

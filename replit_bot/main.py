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


def ticker_cap(expected_pct):
    if expected_pct >= Decimal("1"):
        return Decimal("2500")
    if expected_pct >= Decimal("0.5"):
        return Decimal("2000")
    return Decimal("1500")


def allocate(report, capital):
    """Choose whole-lot orders that maximize estimated rubles per submission."""
    fee = Decimal(str(report["trade_fee_bps"])) / 10_000
    # Reserve each order rounded up to the ruble for the optimizer; invoice in kopecks.
    budget = int(capital)
    frontier = {0: (Decimal(0), ())}
    for item in report["trade_candidates"]:
        lot_size = int(item["lot_size"])
        lot_price = Decimal(item["buy_limit"]) * lot_size
        edge = Decimal(str(item["expected_order_return_pct"]))
        if lot_size <= 0 or not lot_price.is_finite() or lot_price <= 0 or not edge.is_finite():
            raise ValueError("В отчёте неверные параметры заявки")
        if edge <= 0:
            continue
        cap = ticker_cap(edge)
        max_lots = min(int(cap // lot_price), int(capital // (lot_price * (1 + fee))))
        by_cost = {}
        for lots in range(1, max_lots + 1):
            notional = lot_price * lots
            charge = (notional * fee).quantize(CENT, rounding=ROUND_CEILING)
            total = notional + charge
            reserve = int(total.to_integral_value(rounding=ROUND_CEILING))
            if reserve <= budget:
                by_cost[reserve] = (notional * edge / 100, (item, lots, lot_size * lots, total))
        next_states = frontier.copy()
        for spent, (profit, orders) in frontier.items():
            for reserve, (extra_profit, order) in by_cost.items():
                now = spent + reserve
                if now <= budget and (now not in next_states or profit + extra_profit > next_states[now][0]):
                    next_states[now] = (profit + extra_profit, (*orders, order))
        frontier = {}
        best = Decimal("-1")
        for spent, state in sorted(next_states.items()):
            if state[0] > best:
                frontier[spent] = state
                best = state[0]
    _, orders = next(reversed(frontier.values()))
    remaining = capital - sum((order[3] for order in orders), Decimal(0))
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
        if orders:
            expected = sum((Decimal(o[0]["buy_limit"]) * o[2] *
                            Decimal(str(o[0]["expected_order_return_pct"])) / 100 for o in orders), Decimal(0))
            lines.append(f"Оценка результата всех заявок: {expected:+,.2f} ₽")
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

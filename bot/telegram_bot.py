"""Telegram bot for Mneme. A thin remote mouth for `POST /api/ask`.

An independent process that depends on nothing but spec.md §2, so it can be
restarted, or left out of the demo entirely, without touching the backend.

Why Telegram rather than the LINE bot the spec names: spec.md §7 lists "現場網路
爛,LINE webhook 進不來" as a live risk, and it is a real one -- a webhook needs
the venue to route inbound traffic to a device on their LAN. Telegram's
getUpdates is long polling: outbound only, no public URL, no NAT hole, no TLS
certificate. The Orin dials out and hangs on a socket.

What this cannot do is work offline. The memory itself is entirely local -- the
camera, the VLM, the embeddings, the answers -- and pulling the network cable
proves it. The bot is a remote interface to that, so the cable takes the bot
with it and the web UI carries the demo, exactly as spec.md §7 prescribes.

The token is looked for in the environment first, then in `.env` at the repo
root (`TELEGRAM_API_KEY=...`), and `--token` overrides both.

    echo 'TELEGRAM_API_KEY=123456:ABC...' >> .env
    .venv/bin/python bot/telegram_bot.py --api http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import pathlib
import signal
import sys
import time

import requests

log = logging.getLogger("telegram")

POLL_TIMEOUT_S = 50
"""Telegram holds the request open this long when idle. Long polling is one
mostly-idle connection rather than a request per second."""

HTTP_TIMEOUT_S = POLL_TIMEOUT_S + 15
ASK_TIMEOUT_S = 180
"""An /api/ask on the Orin embeds the question, ranks, then generates. Seconds,
not milliseconds, and slower again while the VLM is mid-frame."""

MAX_PHOTOS = 3
BACKOFF_START_S = 2
BACKOFF_MAX_S = 60

ENV_FILE = pathlib.Path(__file__).resolve().parent.parent / ".env"
TOKEN_KEYS = ("TELEGRAM_API_KEY", "MNEME_TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN")


def token_from_env() -> str | None:
    """The bot token, from the environment or from `.env` at the repo root.

    Hand-parsed rather than python-dotenv: the main venv is deliberately thin
    (backend.md 8.2) and this is four lines of KEY=VALUE. The environment wins,
    so `MNEME_TELEGRAM_TOKEN=... ./start.sh` still overrides the file.
    """
    for key in TOKEN_KEYS:
        if os.environ.get(key):
            return os.environ[key]
    try:
        text = ENV_FILE.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in TOKEN_KEYS:
            return value.strip().strip("'\"") or None
    return None


HELP = (
    "問我這個房間發生過什麼事,例如:\n"
    "· 我的杯子在哪\n"
    "· 我剛剛在做什麼\n"
    "· 桌上有什麼東西\n\n"
    "答案只會來自這台 Orin 上看到的畫面。沒看到的事,我會直說沒看到。"
)


class Bot:
    def __init__(self, token: str, api: str) -> None:
        self.tg = f"https://api.telegram.org/bot{token}"
        self.api = api.rstrip("/")
        self.offset: int | None = None
        self.session = requests.Session()
        self.running = True

    # -- telegram ------------------------------------------------------

    def call(self, method: str, **params):
        r = self.session.post(f"{self.tg}/{method}", data=params, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        return r.json()

    def send(self, chat_id: int, text: str, reply_to: int | None = None) -> None:
        self.call(
            "sendMessage",
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_to_message_id=reply_to or "",
        )

    def send_photo(self, chat_id: int, image: bytes, caption: str) -> None:
        # Uploaded rather than linked: api.telegram.org would have to reach the
        # Orin to fetch a URL, which is the inbound path this design avoids.
        r = self.session.post(
            f"{self.tg}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"photo": ("frame.jpg", image, "image/jpeg")},
            timeout=HTTP_TIMEOUT_S,
        )
        r.raise_for_status()

    # -- mneme ---------------------------------------------------------

    def ask(self, question: str) -> dict:
        r = self.session.post(
            f"{self.api}/api/ask", json={"question": question}, timeout=ASK_TIMEOUT_S
        )
        r.raise_for_status()
        return r.json()

    def thumb(self, url: str) -> bytes | None:
        try:
            r = self.session.get(f"{self.api}{url}", timeout=30)
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            log.warning("thumb %s: %s", url, exc)
            return None

    def health(self) -> dict | None:
        try:
            r = self.session.get(f"{self.api}/api/health", timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return None

    # -- handling ------------------------------------------------------

    def handle(self, message: dict) -> None:
        chat_id = message["chat"]["id"]
        message_id = message.get("message_id")
        text = (message.get("text") or "").strip()
        if not text:
            return

        if text.startswith("/start") or text.startswith("/help"):
            self.send(chat_id, HELP, message_id)
            return

        if text.startswith("/status"):
            h = self.health()
            if h is None:
                self.send(chat_id, "後端沒有回應。", message_id)
                return
            self.send(
                chat_id,
                f"狀態 {h['status']} · sidecar {h['sidecar']} · {h['mode']}\n"
                f"事件 {h['event_count']} 筆 · {h['capture_fps']:.2f} fps\n"
                f"離線 {'是' if h['offline'] else '否'}\n"
                f"<code>{html.escape(h['vlm_model'])}</code>",
                message_id,
            )
            return

        self.call("sendChatAction", chat_id=chat_id, action="typing")
        try:
            result = self.ask(text)
        except requests.RequestException as exc:
            log.warning("ask failed: %s", exc)
            self.send(chat_id, "問答服務暫時無法使用,稍後再試。", message_id)
            return

        citations = result.get("citations") or []
        answer = result.get("answer") or ""
        footer = (
            f"\n\n<i>{len(citations)} 筆佐證 · {result.get('latency_ms', 0)} ms</i>"
            if citations
            else ""
        )
        self.send(chat_id, html.escape(answer) + footer, message_id)

        # The citations are the point: an answer without the frames behind it is
        # the hallucination this whole design is built to avoid.
        for c in citations[:MAX_PHOTOS]:
            image = self.thumb(c["thumb_url"])
            if image:
                stamp = c["ts"].replace("T", " ")[:19] + " UTC"
                self.send_photo(chat_id, image, f"{stamp} · {c['summary']}")

    # -- loop ----------------------------------------------------------

    def run(self) -> int:
        me = self.call("getMe")["result"]
        log.info("connected as @%s", me.get("username"))
        backoff = BACKOFF_START_S
        while self.running:
            try:
                updates = self.call(
                    "getUpdates",
                    offset=self.offset or "",
                    timeout=POLL_TIMEOUT_S,
                    allowed_updates='["message"]',
                )
                backoff = BACKOFF_START_S
            except requests.RequestException as exc:
                # Expected whenever the cable is out. Keep trying quietly: the
                # bot should rejoin by itself when the network comes back.
                log.info("poll failed (%s); retrying in %ds", exc, backoff)
                time.sleep(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2)
                continue

            for update in updates.get("result", []):
                self.offset = update["update_id"] + 1
                message = update.get("message")
                if not message:
                    continue
                try:
                    self.handle(message)
                except Exception:
                    # One bad message must not end the process; the next one
                    # may be fine, and offset has already advanced past it.
                    log.exception("handling update %s failed", update.get("update_id"))
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python bot/telegram_bot.py")
    parser.add_argument(
        "--api",
        default=os.environ.get("MNEME_API", "http://127.0.0.1:8080"),
        help="the mneme backend (spec.md 2)",
    )
    parser.add_argument(
        "--token",
        default=token_from_env(),
        help="bot token from @BotFather; TELEGRAM_API_KEY in .env (or in the "
        "environment) is preferred so it does not land in shell history",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    if not args.token:
        print(
            f"no bot token: put TELEGRAM_API_KEY=... in {ENV_FILE}, export it, "
            "or pass --token\n"
            "get one from @BotFather on Telegram",
            file=sys.stderr,
        )
        return 2

    bot = Bot(args.token, args.api)

    def stop(*_: object) -> None:
        bot.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    return bot.run()


if __name__ == "__main__":
    raise SystemExit(main())

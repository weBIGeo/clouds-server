import logging
import urllib.request
import config

logger = logging.getLogger(__name__)


def _send_ntfy(subject: str, body: str) -> None:
    topic = getattr(config, "ntfy_topic", None)
    if not topic:
        return
    url = f"{getattr(config, 'ntfy_server', 'https://ntfy.sh')}/{topic}"
    try:
        req = urllib.request.Request(
            url,
            data=body.encode("utf-8"),
            headers={"Title": subject, "Content-Type": "text/plain"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        logger.debug(f"ntfy notification sent: {subject}")
    except Exception as e:
        logger.warning(f"Failed to send ntfy notification: {e}")


def notify(subject: str, body: str) -> None:
    _send_ntfy(subject, body)

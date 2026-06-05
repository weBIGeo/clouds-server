import logging
import smtplib
import config
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def send_mail(subject: str, body: str) -> None:
    """Send an email using the SMTP settings in config. No-op if not configured."""
    if not getattr(config, "smtp_host", None) or not getattr(config, "notify_email", None):
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = getattr(config, "smtp_user", config.notify_email)
        msg["To"] = config.notify_email
        msg.set_content(body)
        with smtplib.SMTP(config.smtp_host, getattr(config, "smtp_port", 587)) as smtp:
            smtp.starttls()
            user = getattr(config, "smtp_user", None)
            password = getattr(config, "smtp_password", None)
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
        logger.debug(f"Notification email sent: {subject}")
    except Exception as e:
        logger.warning(f"Failed to send notification email: {e}")

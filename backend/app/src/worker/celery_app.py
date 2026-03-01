from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Use Redis as broker and backend
# Default to localhost if not specified in env
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "smarttcm_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.src.worker.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
)

if __name__ == "__main__":
    celery_app.start()

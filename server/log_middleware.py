import json
import logging
import os
from logging import Formatter
from starlette.middleware.base import BaseHTTPMiddleware


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()

    def format(self, record):
        json_record = {}
        json_record["message"] = record.getMessage()
        if "url" in record.__dict__:
            json_record["url"] = record.__dict__["url"]
        if "method" in record.__dict__:
            json_record["method"] = record.__dict__["method"]
        if "status_code" in record.__dict__:
            json_record["status_code"] = record.__dict__["status_code"]
        return json.dumps(json_record)


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logger = logging.root
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.handlers = [handler]
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logging.getLogger("uvicorn.access").disabled = True


class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        logger.info(
            "Request",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
            },
        )
        return response

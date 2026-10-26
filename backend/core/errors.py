"""Consistent public errors, without traceback or invalid JSON values."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


async def app_error(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}}
    )


async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Check the request fields.",
                "details": details,
            }
        },
    )

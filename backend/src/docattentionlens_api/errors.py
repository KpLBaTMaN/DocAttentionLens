"""Structured transport errors shared by routers, services and repositories."""
from fastapi import HTTPException
from pydantic import ValidationError


def fail(code: str, message: str, status: int = 422, **detail):
    raise HTTPException(status, {"code": code, "message": message, **detail})


def readable(exc: Exception) -> str:
    """One line per invalid parameter instead of Pydantic's multi-line report."""
    if not isinstance(exc, ValidationError):
        return str(exc)
    problems = []
    for error in exc.errors():
        name = ".".join(str(part) for part in error["loc"]) or "parameters"
        if error["type"] == "extra_forbidden":
            problems.append(f"Unknown parameter {name}.")
        else:
            problems.append(f"Parameter {name}: {error['msg'].removeprefix('Value error, ')}.")
    return " ".join(problems)

from fastapi import Request

from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_api.repositories.receipts import FileReceiptStore


def repository(request: Request) -> JobRepository:
    return request.app.state.repository


def receipt_store(request: Request) -> FileReceiptStore:
    return request.app.state.receipt_store

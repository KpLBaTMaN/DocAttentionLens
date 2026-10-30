"""Checksummed document study plans/results; worker result admission is fenced by JobRepository.finish."""
from docattentionlens.experiments.document_studies import DocumentStudy,freeze
from .frozen_records import FrozenRecordRepository


class DocumentStudyRepository(FrozenRecordRepository):
    folder="document_studies"; pattern=r"document_study_[a-f0-9]{32}"; key="study_id"; table="document_studies"
    model=DocumentStudy; freeze=staticmethod(freeze)
    unknown="Unknown document study"; integrity_failed="Document study failed integrity verification"
    prepared_only="Only prepared studies can be admitted outside worker publication"

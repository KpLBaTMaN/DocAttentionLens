"""Checksummed attention experiment plans/results; publication uses the job lease fence."""
from docattentionlens.experiments.attention_interventions import AttentionExperiment,freeze
from .frozen_records import FrozenRecordRepository


class AttentionExperimentRepository(FrozenRecordRepository):
    folder="attention_experiments"; pattern=r"attention_experiment_[a-f0-9]{32}"; key="experiment_id"; table="attention_experiments"
    model=AttentionExperiment; freeze=staticmethod(freeze)
    unknown="Unknown attention experiment"; integrity_failed="Attention experiment failed integrity verification"
    prepared_only="Only prepared experiments can be admitted outside worker publication"

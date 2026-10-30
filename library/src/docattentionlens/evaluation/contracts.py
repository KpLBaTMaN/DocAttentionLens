from typing import Literal
from pydantic import ConfigDict, Field, model_validator
from docattentionlens.datasets.contracts import Record as BaseRecord


class Record(BaseRecord):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

Rubric = Literal['boolean_strict_v1', 'normalized_exact_v1', 'human_adjudication_v1']
Role = Literal['engineering', 'independent_researcher']


class StudyCase(Record):
    protocol_id: str
    split: Literal['discovery', 'validation']
    rubric: Rubric
    answer_criteria: str = Field(min_length=20, max_length=3000)
    reviewer: str = Field(min_length=1, max_length=200)
    reviewer_role: Role
    review_notes: str = Field(min_length=20, max_length=5000)
    support_labels_checked: Literal[True]
    reference_answer_checked: Literal[True]


class StudyRequest(Record):
    name: str = Field(min_length=1, max_length=200)
    cases: list[StudyCase] = Field(min_length=1, max_length=8)
    repetitions: int = Field(ge=1, le=5)
    selection_criteria: str = Field(min_length=20, max_length=5000)
    exclusions: str = Field(max_length=5000)
    baseline_instructions: str = Field(min_length=20, max_length=5000)
    task_instructions: str = Field(min_length=20, max_length=5000)
    study_kind: Literal['engineering_pilot', 'researcher_evaluation']

    @model_validator(mode='after')
    def explicit(self):
        if len({c.protocol_id for c in self.cases}) != len(self.cases):
            raise ValueError('An experiment protocol can occur only once in a study')
        if self.study_kind == 'researcher_evaluation' and any(
                c.reviewer_role != 'independent_researcher' for c in self.cases):
            raise ValueError('Researcher evaluation requires declared independent reviews')
        for value in (self.name, self.selection_criteria, self.baseline_instructions,
                      self.task_instructions, *(c.reviewer for c in self.cases)):
            if not value.strip():
                raise ValueError('Substantive text is required')
        return self


class StudyRecord(Record):
    schema_version: Literal['0.6.0'] = '0.6.0'
    record_kind: Literal['evaluation_study'] = 'evaluation_study'
    study_id: str
    created_at: str
    producer_version: str
    dependencies: list[str]
    content_sha256: str | None = None
    specification: StudyRequest
    frozen_cases: list[dict]
    limitations: list[str]


class StudyRunRequest(Record):
    protocol_id: str
    repetition: int = Field(ge=1, le=5)


class ObservationRequest(StudyRunRequest):
    result_id: str


class ArmJudgement(Record):
    arm: Literal['baseline', 'render_control', 'variant', 'non_support_control']
    verdict: Literal['correct', 'incorrect', 'uncertain']
    rationale: str = Field(min_length=10, max_length=3000)


class AnswerReviewRequest(Record):
    result_id: str
    reviewer: str = Field(min_length=1, max_length=200)
    reviewer_role: Role
    judgements: list[ArmJudgement] = Field(min_length=4, max_length=4)

    @model_validator(mode='after')
    def all_arms(self):
        if len({j.arm for j in self.judgements}) != 4:
            raise ValueError('Review each arm exactly once')
        if not self.reviewer.strip():
            raise ValueError('Reviewer identity is required')
        return self


class UtilityTrialRequest(Record):
    protocol_id: str
    participant: str = Field(min_length=1, max_length=100)
    experience: str = Field(min_length=5, max_length=1000)
    task: Literal['locate_support', 'distinguish_context', 'reproduce_investigation']
    condition: Literal['baseline', 'docattentionlens']
    order: Literal[1, 2]
    elapsed_seconds: float = Field(gt=0, le=86400, allow_inf_nan=False)
    completed: bool
    diagnostic_correct: bool | None
    reproduction_success: bool | None
    unsupported_causal_assertion: bool | None
    assessor: str = Field(min_length=1, max_length=200)
    assessor_role: Role
    notes: str = Field(min_length=10, max_length=5000)
    matched_cases_and_compute_confirmed: Literal[True]

    @model_validator(mode='after')
    def meaningful(self):
        if not self.participant.strip() or not self.assessor.strip():
            raise ValueError('Participant pseudonym and assessor are required')
        return self


class EvaluationEvidence(Record):
    schema_version: Literal['0.6.0'] = '0.6.0'
    record_kind: Literal['answer_review', 'utility_trial']
    evidence_id: str
    study_id: str
    created_at: str
    producer_version: str
    dependencies: list[str]
    content_sha256: str | None = None
    payload: AnswerReviewRequest | UtilityTrialRequest

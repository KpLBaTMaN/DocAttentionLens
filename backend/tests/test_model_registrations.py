"""Every built-in model is wired the same way: listed, profiled, launchable and described truthfully.

Parametrised over the registrations and adapter profiles themselves, so a newly added
generator or retriever is checked without editing this file. Needs no database, GPU or
weights; `scripts/model_matrix.py` runs the real models.
"""
from importlib.util import find_spec
import re

import pytest

from docattentionlens_api.services.model_catalogue import builtin_models, generator_profile
from docattentionlens.generation.contracts import GENERATORS
from docattentionlens.retrieval.contracts import RETRIEVER_PROFILES

REGISTRATIONS = builtin_models()


def language(registration):
    return next(c for c in registration["architecture"]["components"] if c["name"] == "Language")


def test_every_adapter_profile_is_listed_once_and_nothing_else_is():
    identifiers = [r["registration_id"] for r in REGISTRATIONS]
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == set(GENERATORS) | set(RETRIEVER_PROFILES)
    assert not set(GENERATORS) & set(RETRIEVER_PROFILES)
    adapters = [r["capabilities"]["adapter_id"] for r in REGISTRATIONS]
    assert len(adapters) == len(set(adapters)), "each registration needs its own adapter version"


@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda r: r["registration_id"])
def test_registration_is_pinned_and_launchable(registration):
    revision = registration["revision"]
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision), "pin an exact commit"
    assert registration["model_id"] and registration["role"] in ("generation", "retrieval")
    assert registration["capabilities"]["can_run"] and registration["capabilities"]["attention_available"]
    component = language(registration)
    assert component["layers"] > 0 and component["query_heads"] > 0 and component["kv_heads"] > 0
    assert component["query_heads"] % component["kv_heads"] == 0


@pytest.mark.parametrize("registration", [r for r in REGISTRATIONS if r["role"] == "generation"], ids=lambda r: r["registration_id"])
def test_generator_registration_matches_its_profile(registration):
    profile = generator_profile(registration)
    assert profile is GENERATORS[registration["registration_id"]]
    assert (registration["model_id"], registration["revision"]) == (profile.model_id, profile.revision)
    assert registration["capabilities"]["adapter_id"] == profile.adapter_version
    component = language(registration)
    assert (component["layers"], component["query_heads"], component["kv_heads"]) == (profile.layers, profile.query_heads, profile.kv_heads)
    assert registration["capabilities"]["limits"]["required_free_gib"] == profile.required_free_gib


@pytest.mark.parametrize("registration", [r for r in REGISTRATIONS if r["role"] == "retrieval"], ids=lambda r: r["registration_id"])
def test_retriever_registration_matches_its_profile(registration):
    profile = RETRIEVER_PROFILES[registration["registration_id"]]
    assert (registration["name"], registration["model_id"], registration["revision"]) == (profile["name"], profile["model_id"], profile["revision"])
    # The catalogue describes each retriever separately; a new one must not inherit another's adapter or limits.
    capabilities = registration["capabilities"]
    assert capabilities["adapter_id"] == profile["adapter_version"]
    assert capabilities["limits"]["image_positions_per_page"] == profile["image_positions_per_page"]
    component = language(registration)
    assert (component["layers"], component["query_heads"]) == (profile["layers"], profile["query_heads"])
    assert f"{profile['layers']} language layers and {profile['query_heads']} query heads" in capabilities["summary"]
    assert isinstance(profile["causal"], bool)
    assert profile["attention_adapter_version"] != profile["adapter_version"]
    assert find_spec(f"docattentionlens.retrieval.{profile['module']}") is not None, "adapter module missing"
    # The capture store must accept a shard for every captured layer, and no further.
    from docattentionlens_api.repositories.captures import CaptureStore
    for kind in ("query", "document", "joint"):
        assert re.fullmatch(CaptureStore.array_pattern, f"attn_{kind}_l{profile['layers'] - 1}"), "last layer's attention shard is not storable"

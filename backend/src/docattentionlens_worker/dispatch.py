from docattentionlens.core.integrity import IntegrityInput, check_integrity


def execute(specification: dict) -> dict:
    if specification["kind"] != "integrity_check" or specification["operation_version"] != "utf8_sha256_v1":
        raise ValueError("Unsupported job operation or version")
    return check_integrity(IntegrityInput.model_validate(specification["inputs"])).model_dump()

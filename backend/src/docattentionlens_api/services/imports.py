"""Immutable document images and compatible local checkpoint packages."""
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
from uuid import uuid4

from PIL import Image, ImageOps, __version__ as PILLOW_VERSION
from psycopg.types.json import Jsonb
from docattentionlens.datasets.adapter import canonical
from docattentionlens.datasets.contracts import DatasetSummary, DocumentRecord, PageRecord, Snapshot, SourceRef
from docattentionlens_api.errors import fail
from docattentionlens_api.services.model_catalogue import builtin_models, describe_architecture, describe_capabilities

DOCUMENT_LIMIT = 50 * 1024**2
MODEL_LIMIT = 12 * 1024**3
UPLOAD_ADAPTER = "uploaded_page_images_v2"
# PyMuPDF is not thread-safe: one PDF render or text read at a time in this process.
PDF_LOCK = Lock()
MODEL_FILES = {"config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.json", "chat_template.jinja", "model.safetensors.index.json", "model.safetensors"}


def digest_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_json(path, value):
    """Expose a complete immutable manifest, including concurrent retries."""
    raw = canonical(value)
    temporary = path.parent / ("publication_" + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                fail("input_conflict", "Published input differs", 409)
    finally:
        temporary.unlink(missing_ok=True)


class ImportStore:
    def __init__(self, root):
        self.root = root / "imports"

    def session(self, identifier):
        if not re.fullmatch(r"upload_[a-f0-9]{32}", identifier):
            fail("invalid_reference", "Unknown upload", 404)
        folder = self.root / identifier
        if not (folder / "session.json").is_file():
            fail("invalid_reference", "Unknown upload", 404)
        return folder, json.loads((folder / "session.json").read_text())

    def start(self, kind):
        identifier = "upload_" + uuid4().hex
        folder = self.root / identifier
        folder.mkdir(parents=True)
        (folder / "session.json").write_bytes(canonical({"kind": kind}))
        return identifier

    async def upload(self, identifier, filename, request):
        folder, session = self.session(identifier)
        if (folder / "result.json").exists():
            fail("immutable_input", "This upload is already imported", 409)
        if not filename or len(filename) > 180 or filename in {".", ".."} or any(c in filename for c in '/\\:') or filename.startswith('.'):
            fail("invalid_filename", "Choose a simple filename without a directory")
        kind = session["kind"]
        if kind == "model":
            if filename not in MODEL_FILES and not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", filename):
                fail("unsupported_file", "Only configuration, tokenizer and safetensors checkpoint files are supported")
            limit = MODEL_LIMIT if filename.endswith('.safetensors') else 16 * 1024**2
        else:
            if Path(filename).suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
                fail("unsupported_file", "Upload PDF, PNG, JPEG or WebP documents")
            limit = DOCUMENT_LIMIT
        files = folder / "files"
        files.mkdir(exist_ok=True)
        if len(list(files.iterdir())) >= 64:
            fail("upload_limit", "A package supports at most 64 files")
        temporary = folder / ("partial_" + uuid4().hex)
        size = 0
        existing_size = sum(p.stat().st_size for p in files.iterdir() if p.name != filename)
        package_limit = MODEL_LIMIT if kind == "model" else DOCUMENT_LIMIT
        try:
            with temporary.open("xb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit or size + existing_size > package_limit:
                        fail("upload_limit", "File exceeds the supported upload size", 413)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if not size:
                fail("invalid_input", "Empty files cannot be imported")
            try:
                os.link(temporary, files / filename)
            except FileExistsError:
                if digest_file(temporary) != digest_file(files / filename):
                    fail("input_conflict", "This filename already contains different bytes", 409)
            return {"filename": filename, "bytes": size}
        finally:
            temporary.unlink(missing_ok=True)


def import_documents(documents, folder, filenames):
    """A multi-file package becomes one ordered page collection; no OCR/labels."""
    import pymupdf
    files = folder / "files"
    if not filenames or len(set(filenames)) != len(filenames) or set(filenames) != {p.name for p in files.iterdir()}:
        fail("invalid_input", "Review all uploaded files in their intended order")
    if sum((files / name).stat().st_size for name in filenames) > DOCUMENT_LIMIT:
        fail("upload_limit", "Document collection exceeds 50 MiB", 413)
    manifest = {"adapter": UPLOAD_ADAPTER, "renderers": {"PyMuPDF": pymupdf.VersionBind, "Pillow": PILLOW_VERSION}, "files": [{"filename": n, "sha256": digest_file(files / n)} for n in filenames], "pdf_render": "RGB, up to 144 dpi, longest edge at most 2048 pixels; visible annotations included", "image_orientation": "EXIF transposed; RGB PNG", "ocr": False, "layout_annotations": False}
    version = sha256(canonical(manifest)).hexdigest()
    dataset_id, document_id = "dataset_" + version, "document_" + version
    destination = documents.jobs.settings.artifact_root / "uploaded-pages" / version
    destination.mkdir(parents=True, exist_ok=True)
    pages, image_files, hashes = [], {}, {}

    def add(image, filename, native_page):
        if len(pages) >= 64:
            fail("upload_limit", "Document collection supports at most 64 pages")
        if image.width * image.height > 16_000_000:
            fail("upload_limit", "Image exceeds 16 million pixels")
        image = ImageOps.exif_transpose(image).convert("RGB")
        raw = BytesIO(); image.save(raw, format="PNG")
        checksum = sha256(raw.getvalue()).hexdigest()
        index = len(pages) + 1
        page_id = "page_" + sha256(f"{version}:{index}".encode()).hexdigest()
        path = destination / f"{index}.png"
        try:
            with path.open("xb") as stream:
                stream.write(raw.getvalue())
        except FileExistsError:
            if digest_file(path) != checksum:
                fail("input_conflict", "Rendered source image differs", 409)
        ref = SourceRef(dataset_id=dataset_id, dataset_version=version, domain="uploaded", document_id=document_id, page_number=index)
        pages.append(PageRecord(key=page_id, page_id=page_id, document_id=document_id, ref=ref, page_number=index, width=image.width, height=image.height, image_sha256=checksum, image_url=f"/api/v1/pages/{page_id}/image", orientation="portrait" if image.height > image.width else "landscape" if image.width > image.height else "square"))
        relative = f"{version}/{index}.png"
        image_files[page_id] = relative; hashes[relative] = checksum
        manifest.setdefault("page_sources", []).append({"page_id": page_id, "filename": filename, "original_page_number": native_page})

    for name in filenames:
        path = files / name
        if path.suffix.lower() == '.pdf':
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass or not 1 <= len(pdf) <= 64 or len(pdf) + len(pages) > 64:
                    fail("invalid_input", "PDF must be unlocked and contain 1–64 pages")
                for index, page in enumerate(pdf):
                    edge = max(page.rect.width, page.rect.height)
                    if not math.isfinite(edge) or edge <= 0:
                        fail("invalid_input", "PDF has invalid page dimensions")
                    scale = min(2, 2048 / edge)
                    pixels = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
                    add(Image.frombytes("RGB", (pixels.width, pixels.height), pixels.samples), name, index + 1)
        else:
            with Image.open(path) as image:
                add(image, name, 1)
    pdf_hash = manifest["files"][0]["sha256"] if len(filenames) == 1 and Path(filenames[0]).suffix.lower() == '.pdf' else ""
    document = DocumentRecord(key=document_id, document_id=document_id, ref=pages[0].ref.model_copy(update={"page_number": None}), native_id=document_id, filename=filenames[0] if len(filenames) == 1 else f"{len(filenames)} uploaded documents", total_pages=len(pages), page_ids=[p.page_id for p in pages], question_count=0, element_count=0, annotation_version="unannotated", source_pdf_sha256=pdf_hash, ocr_model="none", ocr_status="not_available")
    summary = DatasetSummary(dataset_id=dataset_id, name=document.filename, dataset_version=version, adapter_version=UPLOAD_ADAPTER, counts={"documents": 1, "pages": len(pages), "questions": 0, "elements": 0}, domain_counts={"uploaded": {"documents": 1, "pages": len(pages)}}, domain_aliases={}, element_types=[], element_categories=[], question_types=[], quality_counts={}, quality_issues=[], source_manifest=manifest, source_file_count=len(filenames), json_snapshot_sha256=sha256(canonical(manifest)).hexdigest(), geometry_validation="Page images only; no element boxes or evaluation labels imported.")
    snapshot = Snapshot(summary=summary, documents=[document], pages=pages, elements=[], questions=[], file_sha256=hashes, image_files=image_files, source_records={})
    digest = documents.store.put(snapshot)
    with documents.jobs.connect() as conn:
        conn.execute("INSERT INTO datasets(dataset_id,content_hash,summary) VALUES (%s,%s,%s) ON CONFLICT(dataset_id) DO NOTHING", (dataset_id, digest, Jsonb(summary.model_dump(mode="json"))))
    return document


class ModelStore:
    def __init__(self, root):
        self.root = root / "models"

    def list(self):
        result = builtin_models()
        for path in sorted(self.root.glob("model_*/manifest.json")):
            result.append(self.get(path.parent.name))
        return result

    def get(self, identifier):
        builtin = next((m for m in builtin_models() if m["registration_id"] == identifier), None)
        if builtin:
            return builtin
        if not re.fullmatch(r"model_[a-f0-9]{64}", identifier):
            fail("invalid_reference", "Unknown model registration", 404)
        path = self.root / identifier / "manifest.json"
        if not path.is_file():
            fail("invalid_reference", "Unknown model registration", 404)
        record = json.loads(path.read_text())
        if "model_" + sha256(canonical({k: v for k, v in record.items() if k != "registration_id"})).hexdigest() != identifier:
            fail("version_mismatch", "Model manifest failed verification", 409)
        config = self._configuration(path.parent / "files", record, "config.json")
        processor = self._configuration(path.parent / "files", record, "preprocessor_config.json", optional=True)
        role = record.get("role", "generation")
        return {**record, "role":role, "architecture":describe_architecture(config, processor),
            "capabilities":describe_capabilities(config, processor, role)}

    def _configuration(self, files, record, name, optional=False):
        if optional and name not in record["files"]:
            return {}
        path = files / name
        if not path.is_file() or digest_file(path) != record["files"].get(name):
            fail("version_mismatch", "Model configuration failed verification", 409)
        return json.loads(path.read_text())

    def register(self, folder, name, role="generation"):
        files = folder / "files"
        required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
        names = {p.name for p in files.iterdir()}
        if not required <= names:
            fail("incompatible_model", "Checkpoint needs config, preprocessor config and tokenizer files")
        config = json.loads((files / "config.json").read_text())
        if not isinstance(config, dict):
            fail("incompatible_model", "Model configuration must be a JSON object")
        processor = json.loads((files / "preprocessor_config.json").read_text()) if "preprocessor_config.json" in names else {}
        if not isinstance(config.get("model_type"), str) or not config["model_type"] or not isinstance(processor, dict):
            fail("incompatible_model", "Model type and processor configuration must be valid")
        weights = sorted(n for n in names if n.endswith('.safetensors'))
        if not weights:
            fail("incompatible_model", "Full safetensors weights are required; adapter-only and pickle weights are unsupported")
        if "model.safetensors.index.json" in names:
            shard_index = json.loads((files / "model.safetensors.index.json").read_text())
            if not isinstance(shard_index, dict):
                fail("incompatible_model", "Checkpoint index must be a JSON object")
            index = shard_index.get("weight_map", {})
            if not isinstance(index, dict) or not index or set(index.values()) != set(weights):
                fail("incompatible_model", "Checkpoint index must reference exactly the uploaded shards")
        elif weights != ["model.safetensors"]:
            fail("incompatible_model", "Sharded checkpoints require a safetensors index")
        for filename in weights:
            path = files / filename
            with path.open("rb") as stream:
                header_size = int.from_bytes(stream.read(8), "little")
                if not 2 <= header_size <= 16 * 1024**2:
                    fail("incompatible_model", "Invalid safetensors header")
                header = json.loads(stream.read(header_size))
            if not isinstance(header, dict):
                fail("incompatible_model", "Invalid safetensors header structure")
            tensors = [value for key, value in header.items() if key != "__metadata__"]
            if not tensors:
                fail("incompatible_model", "Weights contain no tensors")
            for tensor in tensors:
                if not isinstance(tensor, dict) or tensor.get("dtype") not in {"BF16", "F16", "F32"} or not isinstance(tensor.get("shape"), list):
                    fail("incompatible_model", "Only full floating-point checkpoints are supported")
                offsets = tensor.get("data_offsets", [])
                shape = tensor["shape"]
                if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets) or not 0 <= offsets[0] < offsets[1] <= path.stat().st_size - 8 - header_size:
                    fail("incompatible_model", "Invalid safetensors data extent")
                if any(type(n) is not int or n < 0 for n in shape) or math.prod(shape) * (4 if tensor["dtype"] == "F32" else 2) != offsets[1] - offsets[0]:
                    fail("incompatible_model", "Safetensors shapes do not match their data extents")
        checksums = {n: digest_file(files / n) for n in sorted(names)}
        capabilities = describe_capabilities(config, processor, role)
        manifest = {"name": name, "model_id": name, "role":role, "revision": sha256(canonical(checksums)).hexdigest(), "files": checksums,
            "validation": "File structure checked; full weight loading, processor mapping and capture equivalence checked at launch" if capabilities["can_run"] else "File structure checked; metadata only until a compatible extraction adapter is installed"}
        identifier = "model_" + sha256(canonical(manifest)).hexdigest()
        destination = self.root / identifier
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "files").mkdir(exist_ok=True)
        for filename in names:
            try:
                os.link(files / filename, destination / "files" / filename)
            except FileExistsError:
                if digest_file(destination / "files" / filename) != manifest["files"][filename]:
                    fail("input_conflict", "Existing checkpoint differs", 409)
        manifest["registration_id"] = identifier
        publish_json(destination / "manifest.json", manifest)
        return self.get(identifier)

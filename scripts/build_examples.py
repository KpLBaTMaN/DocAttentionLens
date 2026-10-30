"""Build the example datasets committed under data/. See docs/examples.md.

    python scripts/build_examples.py           # copy the annual report, then build the table export
    python scripts/build_examples.py --tables  # rebuild the table export only; no benchmark needed

The document step copies one annual report byte for byte from data/doclayout_mm_rag_bench/. The table
step reads data/TableGroundQA-Examples-v1/authoring/tables.json, requires every authored cell text to
equal the report's PDF text layer (ignoring whitespace), derives row, column and cell geometry from the
matched words, and writes a TableGroundQA export that the table adapter reads. Output is deterministic.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "library/src"))
from docattentionlens.datasets.adapter import read_dataset  # noqa: E402
from docattentionlens.datasets.table_grounding import read_table_dataset  # noqa: E402

BENCHMARK = ROOT / "data/doclayout_mm_rag_bench"
DOCUMENTS = ROOT / "data/doclayout_mm_rag_examples"
TABLES = ROOT / "data/TableGroundQA-Examples-v1"
DOMAIN, DOCUMENT_ID = "annual_reports", 30
CREATED_ON = "2026-10-29"
LINE_TOLERANCE = 2.5  # points: words whose centres differ by less share a text line
JOIN_GAP = 2.5        # points: closer words are drawn as one token (".97" is set as "." and "97")
PAD = 2.0             # points of margin between the outermost words and the table edge
# Files the table step writes; authoring/ and README.md are kept.
GENERATED = ("manifest.json", "documents", "pages", "structure", "qa", "checks")
VALUE_TYPES = {"integer", "decimal", "text"}  # the pilot's answer vocabulary; units say dollars or percent
GENERATOR = "build_examples_v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def squash(text):
    return "".join(text.split())


# ---------------------------------------------------------------- document example

def copy_document():
    """The benchmark's annual report, its pages, PDF, layout, OCR and questions, unchanged."""
    source = BENCHMARK / DOMAIN
    record = next(d for d in read_json(source / "documents.json")["documents"] if d["id"] == DOCUMENT_ID)
    target = DOCUMENTS / DOMAIN
    if target.exists():
        shutil.rmtree(target)
    relatives = [record[k] for k in ("source_pdf_file", "layout_file", "ocr_file", "annotations_file")]
    relatives += [page["image_file"] for page in record["pages"]]
    for relative in relatives:
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target / relative)
    write_json(target / "documents.json", {"documents": [record]})
    elements = len(read_json(source / record["layout_file"])["structured_layout_elements"])
    questions = len(read_json(source / record["annotations_file"])["questions_and_answers"])
    counts = {"total_pages": record["total_pages"], "total_annotations": questions, "total_elements": elements}
    write_json(DOCUMENTS / "manifest.json", {**counts, "domains": {DOMAIN: {"document_count": 1, **counts}}, "finalized_only": True,
        "example_of": {"dataset": "doclayout_mm_rag_bench", "domain": DOMAIN, "document_id": DOCUMENT_ID, "filename": record["filename"],
                       "copied": "Document record, page images, PDF, layout, OCR and question files are byte-identical to the benchmark."}})
    snapshot = read_dataset(DOCUMENTS, name=DOCUMENTS.name)
    print("document example:", snapshot.summary.counts)


# ---------------------------------------------------------------- table geometry

def text_lines(words, region):
    """Words whose centres lie in region, as text lines from top to bottom, each left to right."""
    inside = [w for w in words if region[0] <= (w[0] + w[2]) / 2 <= region[2] and region[1] <= (w[1] + w[3]) / 2 <= region[3]]
    lines = []
    for w in sorted(inside, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        centre = (w[1] + w[3]) / 2
        if lines and abs(lines[-1][0] - centre) < LINE_TOLERANCE:
            lines[-1][1].append(w)
        else:
            lines.append([centre, [w]])
    return [sorted(line, key=lambda w: w[0]) for _, line in lines]


def take(items, target, text_of, what):
    """Consume items while their joined text stays a prefix of target, until it equals target."""
    taken, got = [], ""
    while got != target:
        if not items:
            raise ValueError(f"{what}: the PDF text ends before '{target}' (matched '{got}')")
        following = got + text_of(items[0])
        if not target.startswith(following):
            raise ValueError(f"{what}: expected '{target}', the PDF continues '{following}'")
        got = following
        taken.append(items.pop(0))
    return taken


def pdf_text(words):
    """Words joined as drawn: a space only where the gap shows one."""
    out = ""
    for previous, w in zip([None, *words], words):
        out += ("" if previous is None or w[0] - previous[2] < JOIN_GAP else " ") + w[4]
    return out


def union(words):
    return [min(w[0] for w in words), min(w[1] for w in words), max(w[2] for w in words), max(w[3] for w in words)]


def layout_table(spec, words, region):
    """Match the authored grid to the PDF words; return cells, row and column bands and the table box (points)."""
    n = spec["columns"]
    lines = text_lines(words, region)
    rows = []
    for r, row in enumerate(spec["rows"]):
        kind = row.get("kind", "body")
        authored = [c if isinstance(c, dict) else {"text": c} for c in row["cells"]]
        if kind == "section":
            if len(authored) != 1:
                raise ValueError(f"{spec['table_id']} row {r + 1}: a section row has one cell")
            authored[0]["span"] = n
        if sum(c.get("span", 1) for c in authored) != n:
            raise ValueError(f"{spec['table_id']} row {r + 1}: cells cover {sum(c.get('span', 1) for c in authored)} of {n} columns")
        target = squash("".join(c["text"] for c in authored))
        taken = take(lines, target, lambda line: "".join(w[4] for w in line), f"{spec['table_id']} row {r + 1}")
        sequence = [w for line in taken for w in line]
        cells, column = [], 0
        for c, cell in enumerate(authored):
            span = cell.get("span", 1)
            matched = take(sequence, squash(cell["text"]), lambda w: w[4], f"{spec['table_id']} row {r + 1} cell {c + 1}")
            cells.append({"row": r, "columns": list(range(column, column + span)), "text": " ".join(cell["text"].split()),
                          "words": matched, "kind": kind, "first": c == 0})
            column += span
        rows.append({"kind": kind, "cells": cells, "words": [w for line in taken for w in line]})
    if lines:
        raise ValueError(f"{spec['table_id']}: unannotated text in the table region: " + " | ".join(pdf_text(line) for line in lines))

    every = [w for row in rows for w in row["words"]]
    x1, y1, x2, y2 = union(every)
    table = [x1 - PAD, y1 - PAD, x2 + PAD, y2 + PAD]
    extents = {}
    for row in rows:
        for cell in row["cells"]:
            if len(cell["columns"]) == 1 and cell["words"]:
                lo, _, hi, _ = union(cell["words"])
                a, b = extents.get(cell["columns"][0], (math.inf, -math.inf))
                extents[cell["columns"][0]] = (min(a, lo), max(b, hi))
    if sorted(extents) != list(range(n)):
        raise ValueError(f"{spec['table_id']}: every column needs a single-column cell with text")
    # Column edges sit midway between neighbouring columns' text; where text overlaps, the right column keeps its words.
    xs = [table[0]] + [(extents[c][1] + extents[c + 1][0]) / 2 if extents[c][1] <= extents[c + 1][0] else extents[c + 1][0] - .5
                       for c in range(n - 1)] + [table[2]]
    spans = [(min(w[1] for w in row["words"]), max(w[3] for w in row["words"])) for row in rows]
    for (_, bottom), (top, _) in zip(spans, spans[1:]):
        if bottom > top:
            raise ValueError(f"{spec['table_id']}: rows overlap")
    ys = [table[1]] + [(spans[r][1] + spans[r + 1][0]) / 2 for r in range(len(rows) - 1)] + [table[3]]
    for row in rows:
        for cell in row["cells"]:
            cell["box"] = [xs[cell["columns"][0]], ys[cell["row"]], xs[cell["columns"][-1] + 1], ys[cell["row"] + 1]]
            cell["pdf_box"] = union(cell["words"]) if cell["words"] else None
    row_bands = [[table[0], ys[r], table[2], ys[r + 1]] for r in range(len(rows))]
    column_bands = [[xs[c], table[1], xs[c + 1], table[3]] for c in range(n)]
    return rows, row_bands, column_bands, table


# ---------------------------------------------------------------- table export

def roles(cell):
    method, status = "assistant_visual_interpretation", "proposed"
    if cell["kind"] == "column_header":
        found = ["column_header"] + (["blank"] if not cell["text"] else [])
    elif cell["kind"] == "section" or cell["first"]:
        found = ["row_header"]
    else:
        found = ["data"] if cell["text"] else ["blank"]
    return [{"role": role, "method": method, "status": status, "reviewer_id": None} for role in found]


def scaled(box, s):
    return [v * s for v in box]


def shifted(box, dx, dy):
    return [box[0] - dx, box[1] - dy, box[2] - dx, box[3] - dy]


def accepted(text, extra):
    """The cell text, its unspaced form, the bare number and, for leading decimals, a 0-prefixed form."""
    values = [text, squash(text)]
    bare = squash(text).lstrip("$")
    values += [bare, bare.rstrip("%")]
    if bare.startswith("."):
        values += ["0" + bare, "$0" + bare]
    return list(dict.fromkeys(values + extra))


def overlay(image, crop_box, cells, answers, path):
    """Logical cells outlined, headers shaded, answer cells of the table's questions in red."""
    margin = 40
    window = (max(0, crop_box[0] - margin), max(0, crop_box[1] - margin), min(image.width, crop_box[2] + margin), min(image.height, crop_box[3] + margin))
    view = image.crop(window).convert("RGB")
    shade = Image.new("RGBA", view.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shade)
    for cell in cells:
        b = shifted(cell["page_box"], window[0], window[1])
        header = any(r["role"] in ("column_header", "row_header") for r in cell["roles"])
        draw.rectangle(b, fill=(66, 133, 244, 40) if header else None, outline=(30, 90, 200, 255), width=2)
        if cell["pdf_box_page"]:
            draw.rectangle(shifted(cell["pdf_box_page"], window[0], window[1]), outline=(0, 160, 90, 255), width=1)
    for cell in cells:
        if cell["cell_id"] in answers:
            draw.rectangle(shifted(cell["page_box"], window[0], window[1]), outline=(220, 30, 30, 255), width=5)
    view = Image.alpha_composite(view.convert("RGBA"), shade).convert("RGB")
    view.thumbnail((1600, 1600))
    path.parent.mkdir(parents=True, exist_ok=True)
    view.save(path, quality=85, optimize=True)


def build_tables():
    authoring = read_json(TABLES / "authoring/tables.json")
    for name in GENERATED:
        target = TABLES / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    source = read_dataset(DOCUMENTS, name=DOCUMENTS.name)
    record = read_json(DOCUMENTS / DOMAIN / "documents.json")["documents"][0]
    layout = {e["id"]: e for e in read_json(DOCUMENTS / DOMAIN / record["layout_file"])["structured_layout_elements"]}
    pdf_path = DOCUMENTS / DOMAIN / record["source_pdf_file"]
    pdf = pymupdf.open(pdf_path)
    meta = authoring["source"]
    document_id = f"dlmm-doc-annual-reports-{DOCUMENT_ID:04d}"
    page_ids = {}
    bindings, entries, question_lines, link_lines = {}, [], [], []

    for spec in authoring["tables"]:
        tid, number = spec["table_id"], spec["page_number"]
        page = pdf[number - 1]
        image_source = DOCUMENTS / DOMAIN / record["pages"][number - 1]["image_file"]
        image = Image.open(image_source)
        s = image.width / page.rect.width
        if abs(image.height / page.rect.height - s) > .01 * s:
            raise ValueError("Page image is not a uniform render of the PDF page")
        pid = page_ids.setdefault(number, f"dlmm-page-annual-reports-{DOCUMENT_ID:04d}-p{number:03d}")
        page_image = f"pages/{pid}/page.jpg"
        if not (TABLES / page_image).exists():
            (TABLES / page_image).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_source, TABLES / page_image)
        element = layout[spec["element_id"]]
        b = element["bbox"]
        region = [b["x1"] / s - 4, b["y1"] / s - 4, b["x2"] / s + 4, b["y2"] / s + 4]
        rows, row_bands, column_bands, table_pdf = layout_table(spec, page.get_text("words"), region)

        table_page = scaled(table_pdf, s)
        crop_box = (math.floor(table_page[0]), math.floor(table_page[1]), math.ceil(table_page[2]), math.ceil(table_page[3]))
        dx, dy = crop_box[0], crop_box[1]
        crop_path = f"pages/table_crops/{tid}.png"
        (TABLES / crop_path).parent.mkdir(parents=True, exist_ok=True)
        image.crop(crop_box).save(TABLES / crop_path, optimize=True)
        with Image.open(TABLES / crop_path) as saved:
            a = np.asarray(saved.convert("L"), dtype=np.float64).ravel()
        c = np.asarray(image.crop(crop_box).convert("L"), dtype=np.float64).ravel()
        correlation = float(np.corrcoef(a, c)[0, 1])
        crop_to_page = [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]]

        row_ids = [f"row-{i + 1:04d}" for i in range(len(rows))]
        column_ids = [f"column-{i + 1:04d}" for i in range(spec["columns"])]
        cells, issues = [], []
        for row in rows:
            for cell in row["cells"]:
                cell["cell_id"] = f"cell-{len(cells) + 1:05d}"
                cell["roles"] = roles(cell)
                cell["page_box"] = scaled(cell["box"], s)
                cell["pdf_box_page"] = scaled(cell["pdf_box"], s) if cell["pdf_box"] else None
                cell["pdf_text"] = pdf_text(cell["words"])
                cells.append(cell)
                if cell["pdf_box"] and (cell["pdf_box"][0] < cell["box"][0] - .5 or cell["pdf_box"][2] > cell["box"][2] + .5):
                    issues.append({"issue_id": f"{tid}-overflow-{cell['cell_id']}", "code": "text_crosses_logical_cell", "severity": "advisory",
                                   "affected_cell_ids": [cell["cell_id"]],
                                   "detail": f"The text '{cell['text']}' runs past its column edge; its logical box keeps the column partition."})
        issues.append({"issue_id": f"{tid}-review", "code": "annotation_not_independently_reviewed", "severity": "advisory", "affected_cell_ids": [],
                       "detail": "Rows, columns, spans, header roles and questions are AI generated. Every cell text matches the PDF "
                                 "text layer; geometry comes from the matched words. Independent human review is pending."})

        # Each question's answer cell and lookup path, resolved before the snapshot that records them as header links.
        by_id = {c["cell_id"]: c for c in cells}
        header_rows = [i for i, row in enumerate(rows) if row["kind"] == "column_header"]
        planned = []
        for q in spec["questions"]:
            if q["value_type"] not in VALUE_TYPES:
                raise ValueError(f"{tid}: value type '{q['value_type']}' is outside {sorted(VALUE_TYPES)}")
            r = [i for i, row in enumerate(rows) if row["kind"] == "body" and row["cells"][0]["text"] == q["row"]]
            columns = [i for cell in rows[header_rows[-1]]["cells"] if cell["text"] == q["column"] for i in cell["columns"]]
            if len(r) != 1 or len(columns) != 1:
                raise ValueError(f"{tid}: question '{q['question']}' names no unique row and column")
            r, column = r[0], columns[0]
            answer = next(c for c in rows[r]["cells"] if c["columns"] == [column])
            section = next((rows[i]["cells"][0] for i in range(r - 1, -1, -1) if rows[i]["kind"] == "section"), None)
            row_path = ([section["cell_id"]] if section else []) + [rows[r]["cells"][0]["cell_id"]]
            column_path = [c["cell_id"] for i in header_rows for c in rows[i]["cells"] if column in c["columns"] and c["text"]]
            support = []
            for text in q.get("support", []):
                found = [c["cell_id"] for c in cells if c["text"] == text]
                if len(found) != 1:
                    raise ValueError(f"{tid}: support text '{text}' names no unique cell")
                support += found
            rationale = (f"Find the row '{q['row']}'" + (f" in the '{section['text']}' section" if section else "")
                         + f", then the column '{q['column']}'"
                         + "".join(f" under '{by_id[i]['text']}'" for i in column_path[:-1]) + "; read their intersecting cell."
                         + (f" The header '{by_id[support[0]]['text']}' gives the unit." if support else "")
                         + (f" {spec['unit_note']}" if spec.get("unit_note") else ""))
            if squash(answer["text"]) != squash(answer["pdf_text"]):
                raise ValueError(f"{tid}: answer differs from its PDF cell text")
            planned.append({"spec": q, "qid": f"{tid.replace('-table-', '-question-')}-{len(planned) + 1:02d}", "row": r, "column": column,
                            "answer": answer, "row_path": row_path, "column_path": column_path, "support": support, "rationale": rationale})
        page_binding = {"status": "page_grounding_verified", "page_id": pid, "table_bbox_page": table_page,
                        "crop_to_page": {"from_frame": "crop_pixels", "to_frame": "page_image_pixels", "matrix": crop_to_page},
                        "provenance": "The crop is cut from the page image, so crop pixels plus the crop origin give page pixels."}

        table_record = {
            "schema_version": "0.1.0", "record_type": "table", "table_id": tid, "table_revision": "v0001", "document_id": document_id,
            "title": spec["title"], "size": spec["size"],
            "source": {"dataset": source.summary.name, "dataset_revision": source.summary.dataset_version, "domain": DOMAIN,
                       "document_id": DOCUMENT_ID, "filename": record["filename"], "page_number": number, "layout_element_id": spec["element_id"],
                       "source_table_key": f"{DOMAIN}/{DOCUMENT_ID}/element/{spec['element_id']}",
                       "normalization_method": "authored_rows_matched_to_pdf_words_v1"},
            "page_binding": page_binding,
            "bbox": shifted(table_page, dx, dy), "bbox_frame": "crop_pixels",
            "assets": [{"asset_id": f"{tid}-crop", "kind": "table_crop", "path": crop_path, "sha256": digest(TABLES / crop_path),
                        "width": crop_box[2] - crop_box[0], "height": crop_box[3] - crop_box[1],
                        "frame": {"frame_id": "crop_pixels", "units": "pixels", "origin": "top_left", "bbox_format": "xyxy_continuous_edges"}}],
            "rows": [{"row_id": row_ids[i], "index": i, "kind": rows[i]["kind"], "bbox": shifted(scaled(band, s), dx, dy), "bbox_frame": "crop_pixels"}
                     for i, band in enumerate(row_bands)],
            "columns": [{"column_id": column_ids[i], "index": i, "bbox": shifted(scaled(band, s), dx, dy), "bbox_frame": "crop_pixels"}
                        for i, band in enumerate(column_bands)],
            "cells": [{"cell_id": c["cell_id"], "row_ids": [row_ids[c["row"]]], "column_ids": [column_ids[i] for i in c["columns"]],
                       "bbox": shifted(c["page_box"], dx, dy), "bbox_frame": "crop_pixels",
                       "geometry_method": "spanning cell" if len(c["columns"]) > 1 else "row-column intersection",
                       "text": {"raw": c["text"], "normalized": c["text"], "normalization_policy": "whitespace_v1",
                                "method": "assistant_visual_read_matched_to_pdf_text", "status": "unreviewed", "reviewer_id": None},
                       "roles": c["roles"]} for c in cells],
            "header_links": [{"target_cell_id": p["answer"]["cell_id"], "row_header_path": p["row_path"], "column_header_path": p["column_path"],
                              "support_cell_ids": p["support"], "method": "assistant_visual_interpretation", "status": "proposed", "reviewer_id": None}
                             for p in planned],
            "quality": {"status": "needs_review", "publishable": False, "issues": issues},
            "revision_provenance": {"authored_on": authoring["authored_on"], "authoring": "authoring/tables.json", "builder": "scripts/build_examples.py"},
        }
        table_path = f"structure/{tid}/revisions/v0001/table.json"
        table_sha = write_json(TABLES / table_path, table_record)

        native = {
            "schema_version": "1.0.0", "record_type": "authored_pdf_table_annotation", "table_id": tid,
            "source_pdf_file_name": record["filename"], "page_index": number - 1, "coordinate_frame": "PDF points, top-left origin of the page",
            "pdf_file_name": record["source_pdf_file"].rsplit("/", 1)[-1], "pdf_page_index": number - 1,
            "pdf_full_page_bbox": [0.0, 0.0, page.rect.width, page.rect.height],
            "pdf_table_bbox": table_pdf, "pdf_table_wrap_bbox": table_pdf,
            "rows": [{"pdf_row_bbox": band, "is_column_header": rows[i]["kind"] == "column_header"} for i, band in enumerate(row_bands)],
            "columns": [{"pdf_column_bbox": band} for band in column_bands],
            "cells": [{"row_nums": [c["row"]], "column_nums": c["columns"], "pdf_bbox": c["pdf_box"], "pdf_text_content": c["pdf_text"],
                       "is_column_header": c["kind"] == "column_header", "is_projected_row_header": c["kind"] == "section",
                       "pdf_words": [[w[0], w[1], w[2], w[3], w[4]] for w in c["words"]]} for c in cells],
        }
        native_path = f"structure/{tid}/original_pdf_annotation.json"
        native_sha = write_json(TABLES / native_path, native)

        to_image = [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]]
        identity_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        grounding_path = f"structure/{tid}/page_grounding.json"
        questions, table_lines, answer_checks = [], [], []
        for p in planned:
            q, answer, qid = p["spec"], p["answer"], p["qid"]
            answer_record = {"type": "extractive_cell", "cell_ids": [answer["cell_id"]], "text": answer["text"],
                             "accepted_texts": accepted(answer["text"], q.get("accepted", [])), "value_type": q["value_type"], "unit": q.get("unit")}
            grounding = {"row_ids": [row_ids[p["row"]]], "column_ids": [column_ids[p["column"]]], "row_header_path": p["row_path"],
                         "column_header_path": p["column_path"], "support_cell_ids": p["support"], "rationale": p["rationale"]}
            question = {"schema_version": "0.1.0", "record_type": "question", "question_id": qid, "question_revision": "v0001",
                        "question_set_id": "annual-report-examples-v1", "question_family_id": qid.replace("-question-", "-family-"),
                        "question_type": "lookup", "status": "draft", "question": q["question"],
                        "scope": {"document_id": document_id, "page_id": pid, "table_id": tid, "table_revision": "v0001", "table_sha256": table_sha,
                                  "upstream_split": None, "split_group_id": document_id},
                        "input": {"mode": "full_page", "page_id": pid, "asset_id": f"{pid}-image"}, "answer": answer_record, "grounding": grounding,
                        "normalization_policy": "whitespace_v1",
                        "authoring": {"method": "assistant_example", "generator_version": GENERATOR,
                                      "source_text_method": "assistant_visual_read_matched_to_pdf_text", "authored_on": authoring["authored_on"]},
                        "review": {"status": "unreviewed", "reviewer_id": None, "reviewed_at": None, "issue_ids": [f"{tid}-review"]}}
            line = json.dumps(question, ensure_ascii=False)
            question_lines.append(line)
            table_lines.append(line)
            link_lines.append(json.dumps({"question_id": qid, "question_revision": "v0001", "question_sha256": sha256(line.encode()).hexdigest(),
                "table_id": tid, "table_revision": "v0001", "table_sha256": table_sha, "page_id": pid, "source_pdf_page_number": number,
                "grounding_path": grounding_path, "answer": answer_record, "answer_boxes_page_pdf": [answer["box"]],
                "answer_boxes_page_image": [answer["page_box"]], "grounding": grounding, "answer_cell_pdf_text": answer["pdf_text"],
                "answer_cell_pdf_text_matched": True, "source_qa_binding_status": "text_matched_not_human_approved"}, ensure_ascii=False))
            questions.append(answer["cell_id"])
            answer_checks.append({"question_id": qid, "answer_cell_id": answer["cell_id"], "expected_answer": answer["text"],
                                  "pdf_cell_text": answer["pdf_text"], "matched": True})
        question_path = f"qa/questions/{tid}.jsonl"
        (TABLES / question_path).parent.mkdir(parents=True, exist_ok=True)
        (TABLES / question_path).write_bytes(("\n".join(table_lines) + "\n").encode("utf-8"))

        overlay_path = f"checks/overlays/{tid}.jpg"
        overlay(image, crop_box, cells, set(questions), TABLES / overlay_path)
        grounding_record = {
            "schema_version": "1.0.0", "record_type": "table_page_grounding", "grounding_revision": "v0001", "table_id": tid,
            "table_revision": "v0001", "table_sha256": table_sha, "page_id": pid,
            "page_image_sha256": digest(TABLES / page_image), "page_image_width": image.width, "page_image_height": image.height,
            "coordinate_frames": {"original_pdf": "PDF points of the report page; top-left origin; y increases down",
                                  "page_pdf": "the same page (no separate page PDF)", "page_image": f"benchmark page JPEG pixels ({round(s * 72)} DPI); top-left; xyxy continuous edges"},
            "source_annotation": {"path": native_path, "sha256": native_sha, "archive_member": None,
                                  "document_json_sha256": digest(DOCUMENTS / DOMAIN / "documents.json"),
                                  "source_pdf_file_name": record["filename"], "original_pdf_page_index": number - 1},
            "transforms": {"original_pdf_to_page_pdf": identity_matrix, "page_pdf_to_page_image": to_image, "crop_image_to_page_image": crop_to_page},
            "source_table_key": f"{DOMAIN}/{DOCUMENT_ID}/element/{spec['element_id']}",
            "table_bbox_original_pdf": table_pdf, "table_bbox_page_pdf": table_pdf, "table_bbox_page_image": table_page,
            "cells": [{"cell_id": c["cell_id"], "row_ids": [row_ids[c["row"]]], "column_ids": [column_ids[i] for i in c["columns"]],
                       "source_cell_indices": [i], "bbox_original_pdf": c["box"], "bbox_page_pdf": c["box"], "bbox_page_image": c["page_box"],
                       "native_span_exact_match": True, "native_bboxes_pdf": [c["pdf_box"]] if c["pdf_box"] else []} for i, c in enumerate(cells)],
            # Native row and column boxes are the extents of their words; logical bands tile the table.
            "rows": [{"row_id": row_ids[i], "bbox_original_pdf": band, "bbox_page_pdf": band, "bbox_page_image": scaled(band, s),
                      "native_bbox_pdf": union(rows[i]["words"])} for i, band in enumerate(row_bands)],
            "columns": [{"column_id": column_ids[i], "bbox_original_pdf": band, "bbox_page_pdf": band, "bbox_page_image": scaled(band, s),
                         "native_bbox_pdf": union([w for c in cells if c["columns"] == [i] for w in c["words"]])} for i, band in enumerate(column_bands)],
            "verification": {"checked_on": CREATED_ON, "automatic_passed": True, "image_registration_normalized_correlation": correlation,
                             "minimum_correlation": .99, "cell_text_matches_pdf": len(cells), "answer_cell_pdf_text_matches": len(questions),
                             "answer_cell_pdf_text_checks": answer_checks, "answer_text_checks_all_matched": True, "all_cells_within_page": True,
                             "native_cells": len(cells), "preserved_logical_cells": len(cells), "native_span_exact_matches": len(cells),
                             "visual_check": "assistant_checked", "visual_checked_on": CREATED_ON, "visual_check_overlay": overlay_path,
                             "independent_semantic_human_review": False},
        }
        grounding_sha = write_json(TABLES / grounding_path, grounding_record)
        bindings.setdefault(number, []).append({"table_id": tid, "table_revision": "v0001", "table_sha256": table_sha,
            "grounding_path": grounding_path, "grounding_sha256": grounding_sha, "binding_status": "page_grounding_verified",
            "crop_to_page_transform_verified": True, "table_bbox_page": table_page,
            "crop_to_page": {"from_frame": "crop_pixels", "to_frame": "page_image_pixels", "matrix": crop_to_page}})
        entries.append({"table_id": tid, "title": spec["title"], "size": spec["size"],
            "source_table_key": f"{DOMAIN}/{DOCUMENT_ID}/element/{spec['element_id']}", "table_revision": "v0001",
            "table_path": table_path, "table_sha256": table_sha, "question_count": len(questions), "crop_path": crop_path,
            "question_path": question_path, "question_sha256": digest(TABLES / question_path),
            "page_id": pid, "page_path": f"pages/{pid}/page.json", "document_id": document_id,
            "row_count": len(rows), "column_count": spec["columns"], "cell_count": len(cells),
            "unknown_text_cell_count": sum(c["text"] is None for c in cells)})
        print(f"{tid}: {len(rows)} rows x {spec['columns']} columns, {len(cells)} cells, {len(questions)} questions, "
              f"{len(issues) - 1} overflow notes, correlation {correlation:.6f}")

    pdf_sha = digest(pdf_path)
    refs = lambda chosen: [{"table_id": e["table_id"], "table_revision": e["table_revision"], "sha256": e["table_sha256"], "path": e["table_path"]} for e in chosen]
    for number, pid in page_ids.items():
        image = Image.open(TABLES / f"pages/{pid}/page.jpg")
        rect = pdf[number - 1].rect
        write_json(TABLES / f"pages/{pid}/page.json", {
            "schema_version": "1.0.0", "record_type": "exported_document_page", "page_id": pid, "document_id": document_id,
            "table_ids": [b["table_id"] for b in bindings[number]], "source_pdf_page_number": number, "source_pdf_page_count": record["total_pages"],
            "image_path": f"pages/{pid}/page.jpg", "image_sha256": digest(TABLES / f"pages/{pid}/page.jpg"),
            "width_pdf_points": rect.width, "height_pdf_points": rect.height, "render_dpi": round(image.width / rect.width * 72),
            "image_width": image.width, "image_height": image.height, "source_pdf_sha256": pdf_sha,
            "source_page_image": f"{DOCUMENTS.name}/{DOMAIN}/{record['pages'][number - 1]['image_file']}",
            "coordinate_frames": {"table_cells": "table crop pixels, top-left origin, continuous xyxy edges",
                                  "page_image": f"{round(image.width / rect.width * 72)} DPI page JPEG pixels, top-left origin",
                                  "original_pdf": "PDF points of the report page, top-left origin, y increases down"},
            "table_refs": refs([e for e in entries if e["page_id"] == pid]),
            "table_bindings": bindings[number]})
    write_json(TABLES / f"documents/{document_id}/document.json", {
        "schema_version": "1.0.0", "record_type": "exported_document", "document_id": document_id, "title": meta["title"],
        "source_document_key": f"{DOMAIN}/{DOCUMENT_ID}", "filename": record["filename"], "doi": None, "license": None,
        "attribution": f"{meta['title']} ({record['filename']}), from the doclayout-mm-rag benchmark",
        "identity": {"status": "benchmark_document", "method": f"Same report as {DOCUMENTS.name} {DOMAIN} document {DOCUMENT_ID}; page images and PDF are byte-identical"},
        "source": f"{DOCUMENTS.name}: the doclayout-mm-rag benchmark's {DOMAIN} document {DOCUMENT_ID}",
        "source_pdf": {"path": f"{DOCUMENTS.name}/{DOMAIN}/{record['source_pdf_file']}", "sha256": pdf_sha,
                       "page_count": record["total_pages"], "full_document_included": False},
        "split_group_id": document_id, "upstream_split": None,
        "page_ids": list(page_ids.values()), "table_ids": [e["table_id"] for e in entries], "table_refs": refs(entries)})
    (TABLES / "qa").mkdir(parents=True, exist_ok=True)
    (TABLES / "qa/questions.jsonl").write_bytes(("\n".join(question_lines) + "\n").encode("utf-8"))
    (TABLES / "qa/page_grounding.jsonl").write_bytes(("\n".join(link_lines) + "\n").encode("utf-8"))
    write_json(TABLES / "manifest.json", {
        "schema_version": "1.0.0", "export_format": "table-qa-pilot-export", "export_id": TABLES.name, "created_on": CREATED_ON,
        "domain": DOMAIN, "source_dataset": source.summary.name, "source_dataset_revision": source.summary.dataset_version,
        "question_count": len(question_lines), "table_count": len(entries), "document_count": 1, "page_count": len(page_ids),
        "status": "draft_unreviewed_example_export", "model_input_mode": "full_page", "approved_release": False,
        "coordinate_convention": "Benchmark page JPEG pixels (300 DPI), top-left origin. Cell, row and column boxes are PDF text-layer word "
                                 "extents scaled to the page; the table crop is cut from that page image.",
        "geometry_validation": "Every authored cell text matches the PDF text layer; boxes are derived from the matched words. "
                               "Independent human review pending.",
        "annotation_method": authoring["authoring_method"],
        "tables": entries})
    files = sorted(p.relative_to(TABLES).as_posix() for p in TABLES.rglob("*") if p.is_file())
    listed = [f for f in files if f not in ("README.md", "checks/SHA256SUMS.txt")]
    (TABLES / "checks/SHA256SUMS.txt").write_bytes("".join(f"{digest(TABLES / f)}  {f}\n" for f in listed).encode())
    snapshot = read_table_dataset(TABLES)
    sizes = Counter(len(t.cells) for t in snapshot.tables)
    print("table example:", snapshot.summary.counts, "cells per table:", sorted(sizes))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", action="store_true", help="rebuild only the table export from the committed annual report")
    if not parser.parse_args().tables:
        copy_document()
    build_tables()


if __name__ == "__main__":
    main()

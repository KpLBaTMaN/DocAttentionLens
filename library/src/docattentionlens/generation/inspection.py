"""Attention allocation and consumed-pixel rendering, independent of GPU runtime."""
from io import BytesIO
import numpy as np
from PIL import Image
from .contracts import PredictionStep


def attention_row(record, values, step, layer, head):
    from ..methods.raw_attention import attention_row as calculate
    return calculate(record, values, step, layer, head)


def pixel_preview(values, grid, mean, std, rescale):
    """Invert actual saved normalized patches; RGB rendering is rounded to uint8.

    Each merged group contains four row-major square patches (14px for Qwen2/2.5-VL,
    16px for Qwen3-VL; read from the saved row width), with duplicated temporal
    frames. This is a display inversion, not a new model input.
    """
    t,h,w = grid
    p = int(round((values.shape[1] / 6) ** .5)) if values.ndim == 2 else 0
    if t != 1 or h % 2 or w % 2 or p < 1 or values.shape != (t*h*w, 3*2*p*p):
        raise ValueError("Unsupported still-image patch geometry")
    pixels = values.reshape(t,h//2,w//2,2,2,3,2,p,p)
    pixels = pixels.transpose(0,6,5,1,3,7,2,4,8).reshape(2,3,h*p,w*p)[0]
    rgb = (pixels*np.asarray(std,dtype=np.float32)[:,None,None]+np.asarray(mean,dtype=np.float32)[:,None,None])/rescale
    image = Image.fromarray(np.clip(np.rint(rgb.transpose(1,2,0)),0,255).astype(np.uint8))
    stream=BytesIO(); image.save(stream, format="PNG"); return stream.getvalue()


def source_view(record, source_index):
    """Present a retained question-token row (or, on full-attention captures, any prefill row) as the capture's only prediction row.

    Existing row methods then apply unchanged: the source's next input token
    is its prediction, and its causal support ends at the source.
    """
    row = next((r for r in record.source_rows if r.source_index == source_index), None)
    full = getattr(record, "full_attention", None)
    if row is None and full is not None and 0 <= source_index < full.position_count:
        # Full-attention captures keep every prefill row: any prompt token (template, question beyond the
        # retained rows, image patch) is read exactly from the saved triangle.
        from .full_attention import prefill_row_name
        source, predicted = record.positions[source_index], record.positions[source_index + 1]
        step = PredictionStep(step=0, pass_kind="prefill", source_index=source_index, consumed_token_id=source.token_id,
            consumed_piece=source.piece, predicted_index=source_index + 1, predicted_token_id=predicted.token_id,
            predicted_piece=predicted.piece, target_count=source_index + 1, array=prefill_row_name(source_index))
        return record.model_copy(update={"steps": [step]}), step
    if row is None:
        raise ValueError("This token's source row was not captured; only full-attention captures keep every prompt row")
    step = PredictionStep(step=0, pass_kind="prefill", source_index=row.source_index, consumed_token_id=row.token_id,
        consumed_piece=row.piece, predicted_index=row.predicted_index, predicted_token_id=row.predicted_token_id,
        predicted_piece=row.predicted_piece, target_count=row.target_count, array=row.array)
    return record.model_copy(update={"steps": [step]}), step


def received_column(record, target_index, layer, head, read_array):
    """Weight one target receives from every retained prediction and question row."""
    from ..methods.attention_received import received
    if not 0 <= layer < record.layer_count or not 0 <= head < record.query_head_count:
        raise ValueError("Layer or head is outside the captured coverage")
    def rows():
        for step in record.steps:
            yield step.source_index, "prediction", f"{step.step} → {step.predicted_piece}", read_array(step.array)[layer, head]
        for source in record.source_rows:
            yield source.source_index, "question", source.piece, read_array(source.array)[layer, head]
    return received(record.capture_id, target_index, layer, head, rows())

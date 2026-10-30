# Backend and library notes

Measurements and history moved out of the Python source, so the code keeps only the
rules it applies. Sections follow the files they explain.

## library/src/docattentionlens/retrieval/colqwen.py

### `EMBEDDING_KERNEL_TOLERANCES` and `SCORE_KERNEL_TOLERANCES`

The retrievers extract attention with FP32-arithmetic eager attention and score with
BF16 SDPA. The tolerances between the two were calibrated per checkpoint on 201
questions with scripts/retriever_kernel_calibration.py. A few sensitive positions move
single coordinates far more than the rest, so embeddings are compared by their mean
per-position L2 distance, and scores per eligible query position. Rotary position errors
are caught by the independent position check, not by these tolerances.

- ColQwen2 (32 pages): embeddings at most 0.024, masking errors 0.09 to 1.1; scores at
  most 0.005. Tolerances: embeddings 0.04, scores 0.01.
- ColQwen2.5 (47 pages): embeddings median 0.024, p99 0.043, at most 0.059, masking
  errors 0.14 to 1.2; scores p99 0.0044, at most 0.019. Its 36 BF16 SDPA layers drift
  further from FP32 arithmetic; FP32 eager and FP32 SDPA agree exactly. Tolerances:
  embeddings 0.09, scores 0.03.

The calibration reports were written to var/verification/colqwen-kernel-calibration.json
and var/verification/colqwen25-kernel-calibration.json, which are local and no longer
kept. The script's docstring gives the commands that regenerate them.

## backend/src/docattentionlens_api/repositories/jobs.py

### `gpu_memory`

Plans admit a model by the free memory CUDA reports to the worker process
(`torch.cuda.mem_get_info`), which is the figure every adapter checks before loading.
Under Docker Desktop/WSL2, nvidia-smi reported about 1.3 GB less (13 October 2026),
which refused Qwen3-VL-8B after any earlier model. The first call creates the CUDA
context that every GPU job needs anyway.

## library/src/docattentionlens/methods/integrated_gradients.py

### Integration path

The attribution integrates over each consumed visual token's embedding as it enters the
language model. The pixel-space path through the vision tower was tested on a saved
Qwen2.5-VL capture and did not converge by 512 steps (its integrand oscillates faster
than the steps resolve), while the embedding path converged by 128 steps.

## library/src/docattentionlens/methods/_literature.py

Each identifier was checked against arXiv, Crossref or the publisher on 14 October 2026.
The audit record is in Git history:
`git show c833fc7:design/evidence/general-analysis-literature-audit.json`.

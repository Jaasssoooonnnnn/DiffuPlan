# Preparing data

## Evaluation

The inference CLI accepts a JSON array, JSONL, or a text file with one question per line. JSON records use question and answer fields; answer can be a string or a list of aliases. The examples directory contains two input-format examples.

The evaluation questions are in benchmarks/, one JSON array per dataset with id, question, and answer fields. Pass --dataset with a file name (without .json) to evaluate it; the CLI reads benchmarks/ by default and reports metrics per dataset.

## Teacher traces and SFT

Teacher trajectories are collected with Diffuplan/scripts/collect_teacher_traces.py. Install requirements/teacher.txt in a separate environment, serve the teacher with scripts/launch_teacher_vllm.sh, start the retriever, list the teacher URLs in configs/teacher.yaml, and run:

    python Diffuplan/scripts/collect_teacher_traces.py \
      --config configs/teacher.yaml \
      --input /path/to/train_questions.jsonl \
      --data-source hotpotqa_train \
      --output-dir data/teacher/hotpotqa

The teacher writes an indexed plan, searches, records subanswers as #A_i lines, rewrites dependent steps in <updated_#Q_j> blocks, and may replan the unresolved suffix once. A trajectory is kept when its answer F1 exceeds 0.65, it used at least one retrieval, and its plan, edits, and actions are well formed; a second pass resamples rejected questions. Complete trajectories are written to traces.jsonl, and trajectories that reached the six-search budget and then answered are written to answer_direct.jsonl.

The converter accepts canonical event records through --input, or full teacher traces through --full-trace-input. A full trace record has schema_version set to teacher_trace_text, with question, trace_text, and gold_answers fields. trace_text contains the teacher's plan, reasoning, tool requests, retrieved information, and answer.

    python Diffuplan/scripts/build_sft_from_canonical_traces.py \
      --full-trace-input data/teacher/hotpotqa/traces.jsonl \
      --answer-direct-trace-input data/teacher/hotpotqa/answer_direct.jsonl \
      --tokenizer-path /path/to/base_model \
      --output-dir data/sft

The converter writes action records, full-canvas edit records, a mixed SFT file, and a summary. Update and replan edit targets generate several partially resolved corruption states. Use the model tokenizer associated with the checkpoint.

SFT/scripts/build_action_upweighted_sft.py replicates prepared records with the multiplicities used in the paper: 1 for search, answer, and edit records, 1.5 for updates, and 3 for replans and search-limit recovery records. Pass the resulting mixed_sft.jsonl, or an existing preweighted corpus, to scripts/train_sft.sh.

## Retrieval assets

The retriever uses intfloat/e5-base-v2, a FAISS index named e5_Flat.index, and a Wiki18 passage corpus named wiki-18.jsonl. Set RETRIEVER_ASSET_DIR to the directory holding the index and corpus, or set INDEX_FILE and CORPUS_FILE separately. The index and corpus must use the same passage ordering.

## Checkpoints

The repository contains model implementation and configuration files. Supply base, SFT, and RL weights separately. SFT can export unfused HF checkpoints; SFT/tools/convert_llada2_fused_to_unfused.py converts an existing fused checkpoint when needed.

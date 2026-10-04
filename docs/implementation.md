# Implementation map

| Component | Files |
| --- | --- |
| Terminal actions and prompt | Diffuplan/diffuplan_core/actions.py, prompting.py |
| Plan localization and commit | Diffuplan/diffuplan_core/plan_editor.py |
| Search/edit execution loop | Diffuplan/diffuplan_core/rollout.py |
| Denoising and expansion | Diffuplan/diffuplan_core/sampling.py |
| dInfer runtime adapters | Diffuplan/diffuplan_core/dinfer_runtime/ |
| Teacher trajectory collection | Diffuplan/scripts/collect_teacher_traces.py, configs/teacher.yaml |
| Action/edit SFT records | Diffuplan/scripts/build_sft_from_canonical_traces.py |
| EPM objective | SFT/dFactory/tasks/dataset/diffuplan_joint_loss.py |
| Joint SFT and attention masks | SFT/dFactory/tasks/train_llada2_bd.py |
| Agentic TraceRL | RL/rl_editable_plan_fused.py, losses.py, trace_to_rows.py |
| Frozen SFT anchor | RL/sft_anchor.py |
| Retrieval service | Diffuplan/retrieval_server.py |

The RL configuration uses four data-parallel rollout workers, an FSDP learner, and a fixed SFT reference on B200 GPUs. The retriever shards its index across the visible GPUs. Optional SM100 kernels are installed under .kernel_runtime from a pinned DeepGEMM revision.

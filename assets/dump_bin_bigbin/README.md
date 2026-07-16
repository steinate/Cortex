# dump_bin_bigbin Demo

This is a compact LeRobot-format demo for the three-stage subtask annotation
workflow. It contains episodes `000000` through `000009` only.

The demo includes parquet state/action data and synchronized head, left-wrist,
and right-wrist videos. It is intended for validating the annotation pipeline,
not for training or reporting benchmark results.

From the repository root, use the default scripts:

```bash
bash scripts/run_scripts/run_vlm_annotation.sh 0 1 2 3 4
bash scripts/run_scripts/run_manual_annotation_server.sh

conda activate InternVLA
bash scripts/run_scripts/run_dynamic_matching.sh
```

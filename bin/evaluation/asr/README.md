# ASR Evaluation Programs

## Evaluate ASR Inference Result
**Data** should be in JSON lines formst, each line should at least 
contains two fields, one for target (groundtruth), one for model output.
```bash
python ./bin/evaluation/asr/eval.py ./demo_configs/evaluation/asr/eval.json
```



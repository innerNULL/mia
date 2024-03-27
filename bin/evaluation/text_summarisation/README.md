# Text-Summarization Evaluation Related Programs

## All-in-One Evaluation Program
Can try with 
```shell
python bin/evaluation/text_summarisation/eval_all_in_one_standalone.py ./demo_configs/evaluation/text_summarisation/eval_all_in_one_standalone.json
```
And it will out put JSON lines file, each line contains reference, candidate and metrics name/value.
Currently 4 metrics are supported:
* ROUGE
* METEOR
* [BERTScore](https://arxiv.org/abs/1904.09675): Implemented by myself, not as fast as open-sourced solution, but handled most margin cases.
* Average Sentence Similarity Based on Decode-Only LM

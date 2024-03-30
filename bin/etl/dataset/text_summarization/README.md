# Text-Summarization Dataset ETLs
Some ETL programs can generate text-summarization datasets.

## Medical/Clinical Reports Summarizasion Datasets
### MIMIC-III Exam Reports Summarization Dataset
Run following command to have a try:
```bash
python bin/etl/dataset/text_summarization/build_med_report_summarization_dataset.py ./demo_configs/etl/dataset/text_summarization/build_med_report_summarization_dataset.mimiciii.json
```

Most medical exam reports are following specific format, there are some fields 
like findings, impressions, indications, diagnosis, condifaions, etc. Typically 
we use "impression" as summary, which is summarization models' target/label, 
and "findings" as full text. which is input of the model.


In MIMIC-III NOTEEVENTS.csv, there are all kinds of medical notes or reports, 
here we only use reports part, which means 
`CATEGORY = 'Radiology' or CATEGORY = 'Echo'`.

The output of this ETL program is a JSON lines file, which contains only 2 
fields represents "impression" and "findings" seperately. and the fields 
name can be set by `target_text_col` and `input_text_col` in config file.


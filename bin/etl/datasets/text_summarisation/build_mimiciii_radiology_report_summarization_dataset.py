# -*- coding: utf-8 -*-
# file: build_radiology_report_summarization_dataset.py
# date: 2024-03-27


import pdb
import sys
import os
import json
import duckdb
import re
from typing import Dict, List, Tuple


SQL_QUERY_RAW_RADIOLOGY_REPORT: str = """
select TEXT from "__NOTEEVENTS_PATH__" where CATEGORY = 'Radiology' 
"""


def full_report_check(full_report: str) -> bool:
    if "IMPRESSION:" not in full_report:
        return False
    if len(full_report.split("IMPRESSION:")) != 2:
        return False
    return True


def sample_processing(full_report: str) -> Dict[str, str]:
    findings: str = ""
    impression: str = ""
    findings, impression = full_report.split("IMPRESSION:")
    findings = findings.strip("\n").strip(" ").replace("_", "")
    impression = impression.strip("\n").strip(" ").replace("_", "")

    findings = re.sub(r'\s+', ' ', findings)
    impression = re.sub(r'\s+', ' ', impression)
    #findings = re.sub(r'\s+', '\n', findings)
    #impression = re.sub(r'\s+', '\n', impression)
    return {"findings": findings, "impression": impression}


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    
    in_file = open(configs["mimic_noteevents_path"], "r")
    out_file = open(configs["output_path"], "w")
    
    raw_rediology_reports = duckdb.query(
        SQL_QUERY_RAW_RADIOLOGY_REPORT.replace(
            "__NOTEEVENTS_PATH__", configs["mimic_noteevents_path"]
        )
    )

    cnt: int = 0
    raw_sample: Tuple = raw_rediology_reports.fetchmany(1)
    while raw_sample and cnt < configs["max_data_size"]:
        raw_sample = raw_rediology_reports.fetchmany(1)[0][0]
        if not full_report_check(raw_sample):
            continue
        curr_sample: Dict[str, str] = sample_processing(raw_sample)
        out_file.write(json.dumps(curr_sample, ensure_ascii=False) + "\n")
        cnt += 1
    
    in_file.close()
    out_file.close()
    print("out file located at '%s'" % configs["output_path"])

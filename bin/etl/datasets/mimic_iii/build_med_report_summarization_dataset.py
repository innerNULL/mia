# -*- coding: utf-8 -*-
# file: build_radiology_report_summarization_dataset.py
# date: 2024-03-27


import pdb
import sys
import os
import json
import duckdb
import re
from tqdm import tqdm
from typing import Dict, List, Tuple


FINDINGS_FIELDS: List[str] = [
    "FINDING", "Finding", "finding",
    "FINDINGS", "findings", "findings"
]

INDICATION_FIELDS: List[str] = [
    "INDICATION", "Indication", "indication"
]

IMPRESSION_FIELDS: List[str] = [
    "IMPRESSION", "Impression", "impression",
    "IMPRESSIONS", "Impressions", "impressions"
]

CONCLUSIONS_FIELDS: List[str] = [
    "CONCLUSIONS", "Conclusions", "conclusions"
]

KEY_FIELDS: List[str] = [
    "GENERAL COMMENTS",
    #"Height", "Weight", 
    #"BSA", "BP", "HR", 
    #"Status", 
    #"Date/Time", 
    #"Test", "Doppler", "Contrast", "Technical Quality", 
    #"LEFT ATRIUM", "RIGHT ATRIUM/INTERATRIAL SEPTUM", "LEFT VENTRICLE", "RIGHT VENTRICLE", 
    #"AORTA", "AORTIC VALVE", "MITRAL VALVE",
    #"TRICUSPID VALVE", "PERICARDIUM",
] + FINDINGS_FIELDS + INDICATION_FIELDS + IMPRESSION_FIELDS + CONCLUSIONS_FIELDS

MINIMUM_FINDINGS_LENGTH: int = 10

MINIMUM_IMPRESSION_LENGTH: int = 10

SQL_QUERY_RAW_RADIOLOGY_REPORT: str = """
with 
med_note_with_impressions as (
  select CATEGORY, TEXT from "__NOTEEVENTS_PATH__" 
  where contains(TEXT, 'IMPRESSION:')
  and CATEGORY != 'Discharge summary'
)
select * from med_note_with_impressions;
"""


def parse_med_report(
    input_text: str, key_fields: List[str]=KEY_FIELDS
) -> Dict[str, str]:
    # Define the pattern to match the allowed uppercase words and their corresponding values
    pattern = r'(' + '|'.join(key_fields) + r'):\s*(.*?)(?=\n(?:' + '|'.join(key_fields) + r')|$)'

    # Find all matches in the input text
    matches = re.findall(pattern, input_text, re.DOTALL)

    # Create a dictionary to store the results
    result_dict = {}

    # Iterate over matches and populate the dictionary
    for match in matches:
        key = match[0].strip()
        value = match[1].strip()
        if key in result_dict:
            result_dict[key] += '\n' + value
        else:
            result_dict[key] = value

    return result_dict


def merge_fields(sample: Dict[str, str], target_fields: List[str]) -> str:
    output: str = ""
    for field in target_fields:
        val: str = sample.get(field, "")
        output = output if val == "" else (output + "\n" + val)
    return output


def text_clean_naive(text: str) -> str:
    text = text.strip("\n").strip(" ").replace("_", "")
    text = re.sub(r'\s+', ' ', text)
    #text = re.sub(r'\s+', '\n', text)
    return text


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
    raw_sample: Tuple = raw_rediology_reports.fetchmany(1)[0]
    with tqdm(total=configs["max_data_size"]) as pbar:
        while raw_sample and cnt <= configs["max_data_size"]:
            category: str = raw_sample[0]
            med_text: str = raw_sample[1]
            parsed_text: Dict[str, str] = parse_med_report(med_text)

            findings: str = merge_fields(parsed_text, FINDINGS_FIELDS)
            findings = text_clean_naive(findings)
            if len(findings) <= MINIMUM_FINDINGS_LENGTH:
                findings = merge_fields(parsed_text, INDICATION_FIELDS + CONCLUSIONS_FIELDS)
                findings = text_clean_naive(findings)
                
            impression: str = merge_fields(parsed_text, IMPRESSION_FIELDS)
            impression = text_clean_naive(impression)

            if len(findings) <= MINIMUM_FINDINGS_LENGTH \
                or len(impression) <= MINIMUM_IMPRESSION_LENGTH:
                pass
            else: 
                curr_sample: Dict = {
                    "source": configs["mimic_noteevents_path"], 
                    configs["target_text_col"]: impression, 
                    configs["input_text_col"]: findings
                }
                out_file.write(json.dumps(curr_sample, ensure_ascii=False) + "\n")
                cnt += 1
                pbar.update(1)

            raw_sample = raw_rediology_reports.fetchmany(1)[0]
    
    in_file.close()
    out_file.close()
    print("out file located at '%s'" % configs["output_path"])

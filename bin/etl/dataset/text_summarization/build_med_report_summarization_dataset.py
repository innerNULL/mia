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
from duckdb.duckdb import DuckDBPyRelation


FINDINGS_FIELDS: List[str] = [
    "FINDING", "Finding", 
    #"finding",
    "FINDINGS", "Findings", 
    #"findings"
]

INDICATION_FIELDS: List[str] = [
    "INDICATION", "Indication", 
    #"indication"
]

IMPRESSION_FIELDS: List[str] = [
    "IMPRESSION", "Impression", 
    #"impression",
    "IMPRESSIONS", "Impressions", 
    #"impressions",
    "IMP", "Imp", 
    #"imp",
    "IMIMPRESSION",
    "USRIMPRESSION",
    "IIMPRESSION"
]

CONCLUSIONS_FIELDS: List[str] = [
    "CONCLUSIONS", "Conclusions", 
    #"conclusions"
]

CLINICAL_HISTORY_FIELDS: List[str] = [
    "MEDICAL HISTORY", 
    "CLINICAL HISTORY",
    "REPORT HISTORY"
]

KEY_FIELDS: List[str] = [
    "GENERAL COMMENTS", 
    "TECHNIQUE",
    "COMPARISON",
    "RECOMMENDATIONS",
    "Suggestion",
    "SUGGESTION",
    #"Height", "Weight", 
    #"BSA", "BP", "HR", 
    #"Status", 
    #"Date/Time", 
    #"Test", "Doppler", "Contrast", "Technical Quality", 
    #"LEFT ATRIUM", "RIGHT ATRIUM/INTERATRIAL SEPTUM", "LEFT VENTRICLE", "RIGHT VENTRICLE", 
    #"AORTA", "AORTIC VALVE", "MITRAL VALVE",
    #"TRICUSPID VALVE", "PERICARDIUM",
] + FINDINGS_FIELDS + INDICATION_FIELDS + IMPRESSION_FIELDS + CONCLUSIONS_FIELDS + CLINICAL_HISTORY_FIELDS

MINIMUM_FINDINGS_LENGTH: int = 10

MINIMUM_IMPRESSION_LENGTH: int = 5

SQL_QUERY_RAW_MED_REPORT: str = """
with 
med_note_with_impressions as (
  select __RAW_TEXT_COL__ 
  from __LOADER__('__NOTEEVENTS_PATH__') __WHERE_STATEMENT__
)
select * from med_note_with_impressions;
"""


def duckdb_load_csv_or_jsonl(
    path: str, raw_text_col: str, ext_columns: List[str], where_statement: str
):
    # read_json_auto
    ext: str = path.split(".")[-1]
    sql: str = SQL_QUERY_RAW_MED_REPORT
    if ext == "csv":
        sql = sql.replace("__LOADER__", "read_csv_auto")
    elif ext == "jsonl":
        sql = sql.replace("__LOADER__", "read_json_auto")
    else:
        raise Exception("Not support %s format" % ext)
    
    target_cols: str = ",".join([raw_text_col] + ext_columns)
    sql = sql\
        .replace("__NOTEEVENTS_PATH__", path)\
        .replace("__RAW_TEXT_COL__", target_cols)\
        .replace("__WHERE_STATEMENT__", where_statement)
    print("running following SQL:")
    print(sql)
    return duckdb.query(sql)


def parse_med_report(
    input_text: str, key_fields: List[str]=KEY_FIELDS
) -> Dict[str, str]:
    # Define the pattern to match the allowed uppercase words and their corresponding values
    pattern = r'(' + '|'.join(key_fields) + r'):*(.*?)(?=(?:' + '|'.join(key_fields) + r')|$)'

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
    text = text.strip(":")
    text = text[2:] if text.startswith("s:") or text.startswith("S:") else text
    text = text.strip("\n").strip(" ")
    
    text = text.strip(":")
    text = text[2:] if text.startswith("s:") or text.startswith("S:") else text
    text = text.strip("\n").strip(" ")

    text = text.replace("_", "") 
    text = re.sub(r'\s+', ' ', text)
    #text = re.sub(r'\s+', '\n', text)
    return text


if __name__ == "__main__":
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)

    if os.path.exists(configs["output_path"]):
        raise Exception("Path %s is already exists" % configs["output_path"])
    
    in_file = open(configs["med_report_data_path"], "r")
    out_file = open(configs["output_path"], "w")
    
    raw_med_reports: DuckDBPyRelation = duckdb_load_csv_or_jsonl(
        configs["med_report_data_path"], 
        configs["raw_text_col"], 
        configs["ext_cols"],
        configs["sql_where_statement"] 
    )
    
    total: int = 0
    cnt: int = 0
    invalid_cases: List[Dict[str, str]] = []
    raw_sample: List[Tuple] = raw_med_reports.fetchmany(1)
    with tqdm(total=configs["max_data_size"]) as pbar:
        while raw_sample and cnt <= configs["max_data_size"]:
            curr_sample: Dict = {}
            for i, value in enumerate(raw_sample[0][1:]):
                curr_sample[configs["ext_cols"][i]] = value

            med_text: str = raw_sample[0][0]
            parsed_text: Dict[str, str] = parse_med_report(med_text)

            findings: str = merge_fields(parsed_text, FINDINGS_FIELDS)
            findings = text_clean_naive(findings)
            impression: str = merge_fields(parsed_text, IMPRESSION_FIELDS)
            impression = text_clean_naive(impression)

            #if not configs["strict_mode"] and len(findings) <= MINIMUM_FINDINGS_LENGTH:
            #    findings = merge_fields(parsed_text, INDICATION_FIELDS + CONCLUSIONS_FIELDS)
            #    findings = text_clean_naive(findings)
            
            #TODO@20240402_1004:
            # Currently regex has some problem, sometime it will not only parse "findings:" 
            # but also "findings"
            #if not configs["strict_mode"]:
            if not configs["strict_mode"] and len(findings) <= MINIMUM_FINDINGS_LENGTH:
                findings = med_text
            for impression_col in IMPRESSION_FIELDS:
                if impression_col in parsed_text:
                    findings = findings.replace(parsed_text[impression_col], "")
                    findings = findings.replace("%s:" % impression_col, "")
            findings = text_clean_naive(findings)
             
            if len(findings) <= MINIMUM_FINDINGS_LENGTH \
                    or len(impression) <= MINIMUM_IMPRESSION_LENGTH \
                    or (len(findings) - len(impression)) / len(findings) < 0.2: #or len(findings) - len(impression) < 20:
                if len(invalid_cases) < 100:
                    parsed_text["raw_text"] = med_text
                    parsed_text["processed_impression"] = impression
                    parsed_text["processed_findings"] = findings
                    parsed_text["impression_length"] = len(impression)
                    parsed_text["findings_length"] = len(findings)
                    invalid_cases.append(parsed_text)
            else: 
                #if abs(len(impression) - len(findings)) < 10:
                #    pdb.set_trace()
                #if len(impression) > len(findings):
                #    pdb.set_trace()
                curr_sample["source"] = configs["med_report_data_path"]
                curr_sample[configs["target_text_col"]] = impression
                curr_sample[configs["input_text_col"]] = findings
                
                if configs["debug_mode"]:
                    curr_sample["med_text"] = med_text 
                out_file.write(json.dumps(curr_sample, ensure_ascii=False) + "\n")
                cnt += 1
                pbar.update(1)

            raw_sample = raw_med_reports.fetchmany(1)
            total += 1
    
    in_file.close()
    out_file.close()
    print("out file located at '%s'" % configs["output_path"])
    print("%i out %i sample are valid" % (cnt, total))
    
    if configs["debug_mode"]:
        print("Can check invalid sample in `invalid_cases`")
        pdb.set_trace()

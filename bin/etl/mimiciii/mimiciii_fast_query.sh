

#set -x


MIMICIII_ROOT_DIR="./"
LABEL_PATTERN="blood pressure"
OUTPUT_PATH="output.csv"
TARGET_HADM_ID=199972
TARGET_CHARTDATE="2186-09-01"


QUERY_TEMP=\
"""
with 
d_item as (
  select ITEMID, LABEL 
  from 
  read_csv('${MIMICIII_ROOT_DIR}/D_ITEMS.csv', auto_detect=true, null_padding=true)
),
d_cpt as (
  select
  CATEGORY,
  SECTIONRANGE, 
  SECTIONHEADER, 
  SUBSECTIONRANGE, 
  SUBSECTIONHEADER, 
  CODESUFFIX, 
  MINCODEINSUBSECTION, 
  MAXCODEINSUBSECTION
  from 
  read_csv('${MIMICIII_ROOT_DIR}/D_CPT.csv', auto_detect=true, null_padding=true)
),
d_labitem as (
  select ITEMID, LABEL
  from
  read_csv('${MIMICIII_ROOT_DIR}/D_LABITEMS.csv', auto_detect=true, null_padding=true)
),
d_icd_procedures as (
  select
  ICD9_CODE, SHORT_TITLE, LONG_TITLE 
  from
  read_csv('${MIMICIII_ROOT_DIR}/D_ICD_PROCEDURES.csv', auto_detect=true, null_padding=true)
),
cptevents as (
  select 
  SUBJECT_ID, HADM_ID, 
  COSTCENTER, CHARTDATE, CPT_CD, CPT_NUMBER, CPT_SUFFIX, 
  TICKET_ID_SEQ, SECTIONHEADER, SUBSECTIONHEADER, DESCRIPTION
  from  
  read_csv(
    '${MIMICIII_ROOT_DIR}/CPTEVENTS.csv', 
    null_padding=true,
    columns={
      'ROW_ID': 'INT',
      'SUBJECT_ID': 'INT',
      'HADM_ID': 'VARCHAR',
      'COSTCENTER': 'VARCHAR',
      'CHARTDATE': 'VARCHAR',
      'CPT_CD': 'VARCHAR',
      'CPT_NUMBER': 'INT',
      'CPT_SUFFIX': 'VARCHAR',
      'TICKET_ID_SEQ': 'VARCHAR',
      'SECTIONHEADER': 'VARCHAR',
      'SUBSECTIONHEADER': 'VARCHAR',
      'DESCRIPTION': 'VARCHAR'
    }
  )
),
chartevents as (
  select 
  HADM_ID, 
  ITEMID,
  strftime(CHARTTIME, '%Y-%m-%d %H:%M:%S') as CHARTTIME,
  case 
    when VALUEUOM = '?F' then ((VALUENUM - 32.0) * 5.0 / 9.0) 
    else VALUENUM 
  end as VALUENUM,
  case 
    when VALUEUOM = '?F' then 'C' 
    when VALUEUOM = '?C' then 'C' 
    else VALUEUOM 
  end as VALUEUOM,
  RESULTSTATUS,
  STOPPED
  from
  read_csv('${MIMICIII_ROOT_DIR}/CHARTEVENTS.csv', auto_detect=true, null_padding=true)
),
prescriptions as (
  select
  HADM_ID, STARTDATE, ENDDATE, DRUG_TYPE, DRUG, DOSE_VAL_RX, DOSE_UNIT_RX
  from 
  read_csv('${MIMICIII_ROOT_DIR}/PRESCRIPTIONS.csv', auto_detect=true, null_padding=true)
),
procedures as (
  select 
  SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
  from 
  read_csv('${MIMICIII_ROOT_DIR}/PROCEDURES_ICD.csv', auto_detect=true, null_padding=true)
  where 
  SUBJECT_ID is not null
  and HADM_ID is not null
  and ICD9_CODE is not null
),
labevents as (
  select 
  HADM_ID, ITEMID, CHARTTIME, VALUE, VALUENUM, VALUEUOM, FLAG
  from 
  read_csv('${MIMICIII_ROOT_DIR}/LABEVENTS.csv', auto_detect=true, null_padding=true)
  where 
  CHARTTIME is not null
  and HADM_ID is not null
  and ITEMID is not null
),
noteevents as (
  select
  HADM_ID, CHARTDATE, CHARTTIME, STORETIME, CATEGORY, DESCRIPTION, ISERROR, TEXT
  from 
  read_csv('${MIMICIII_ROOT_DIR}/NOTEEVENTS.csv', auto_detect=true, null_padding=true, parallel=false)
),
soap_note as (
  select 
  HADM_ID, CHARTDATE, CHARTTIME, STORETIME, CATEGORY, DESCRIPTION, ISERROR, TEXT
  from 
  noteevents
  where 
  CATEGORY in ('Nursing/other', 'Nursing', 'Physician')
  and (TEXT like '%Subjective%' or TEXT like '%Objective%' or TEXT like '%Assessment%' or TEXT like '%Plan%')
  and TEXT is not null
),
procedures_expand as (
  select t1.*, t2.SHORT_TITLE, t2.LONG_TITLE 
  from procedures as t1 
  inner join d_icd_procedures as t2 
  on t1.ICD9_CODE = t2.ICD9_CODE
),
surgery as (
  select * from procedures_expand 
  where lower(LONG_TITLE) like '%surgery%'
),
chartevents_expand as (
  select 
  t1.*, 
  case 
    when t2.LABEL = 'Temperature Fahrenheit' then 'Temperature Celsius'  
    else t2.LABEL
  end as LABEL
  from chartevents as t1 inner join d_item as t2
  on t1.ITEMID = t2.ITEMID
),
labevents_expand as (
  select
  t1.*, t2.LABEL
  from labevents as t1 inner join d_labitem as t2
  on t1.ITEMID = t2.ITEMID
),
eda_chartevents as (
  select 
  CHARTTIME,
  VALUENUM, 
  VALUEUOM, 
  RESULTSTATUS, 
  STOPPED,
  LABEL
  from 
  chartevents_expand
  where
  HADM_ID = ${TARGET_HADM_ID}
  and CHARTTIME like '%${TARGET_CHARTDATE}%'
  and lower(LABEL) like '%${LABEL_PATTERN}%'
  and VALUENUM is not null
  and VALUEUOM is not null
  order by CHARTTIME desc
),
eda_prescriptions as (
  select 
  *, 
  (strptime('${TARGET_CHARTDATE} 00:00:00', '%Y-%m-%d %H:%M:%S') - interval 1 day) as last_24h
  from prescriptions
  where
  HADM_ID = ${TARGET_HADM_ID}
  and (
    (STARTDATE <= '${TARGET_CHARTDATE} 00:00:00' and ENDDATE >= '${TARGET_CHARTDATE} 00:00:00')
    or 
    (STARTDATE >= last_24h and ENDDATE <= last_24h)
  )
  order by STARTDATE desc, ENDDATE desc
),
eda_soap_note as (
  select TEXT from soap_note
  where 
  HADM_ID = ${TARGET_HADM_ID}
  and TEXT like '%Assessment%'
  order by CHARTTIME desc 
  limit 1
),
eda_labevents_expand as (
  select * from labevents_expand
  where 
  HADM_ID = ${TARGET_HADM_ID}
  order by CHARTTIME desc
),
eda_procedures_expand as (
  select 
  SHORT_TITLE,LONG_TITLE
  from procedures_expand 
  where 
  HADM_ID = ${TARGET_HADM_ID}
),
eda_cptevents as (
  select 
  COSTCENTER, CHARTDATE, CPT_CD, CPT_NUMBER, CPT_SUFFIX, SECTIONHEADER, SUBSECTIONHEADER, DESCRIPTION
  from cptevents
  where
  HADM_ID = ${TARGET_HADM_ID}
  order by CHARTDATE desc
),
eda_surgery as (
  select * from surgery
  where
  HADM_ID = ${TARGET_HADM_ID}
),
eda_keyevents_cpt as (
  select 
  COSTCENTER,CHARTDATE,CPT_CD,CPT_NUMBER,CPT_SUFFIX,TICKET_ID_SEQ,SECTIONHEADER,SUBSECTIONHEADER,DESCRIPTION
  from cptevents where HADM_ID = ${TARGET_HADM_ID}
  order by CHARTDATE desc
),
eda_keyevents_lab as (
  select 
  CHARTTIME,VALUE,VALUENUM,VALUEUOM,FLAG,LABEL
  from labevents_expand
  where
  HADM_ID = ${TARGET_HADM_ID}
  and CHARTTIME < '2186-09-02 00:00:00'
  order by CHARTTIME desc
)
select * from eda_keyevents_lab
"""


DUMP_DQL=\
"""
COPY(
${QUERY_TEMP}
) to '${OUTPUT_PATH}' (HEADER, DELIMITER ',');
"""


echo -e "${DUMP_DQL}" | duckdb
echo "Output is dumped at ${OUTPUT_PATH}"
echo "Output examples:"
echo "select * from read_csv('${OUTPUT_PATH}', auto_detect=true) limit 20;" | duckdb

# EHR_DREAM_challenge-COVID-19


## Problem Statement

For patients with a positive RT-PCR for COVID-19 and who were tested at an outpatient visit, which patients were admitted to the hospital within 21 days of their RT-PCR test.


## Input: 
EHR data from UWMedicine including geographic, demographic, and clinical measures. Data will be provided as an OMOP Limited Data Set. A goldstandard.csv file will be made available to the models during the training phase on which you can train your models.

## Output: 
A score between 0 and 1 indicating the likelihood that the patient will be hospitalized. We define hospitalization as any visit with a visit_concept_id of [9201 (Inpatient), 9203 (Emergency Room), 32037 (Intensive Care)]. The prediction window (hospitalization within 21 days) starts at the latest available positive test result in the measurement table.

## Performance metrics: 
AUPR, AUROC, balanced accuracy.

## Significance: 
This is designed to be a clinically relevant question that will give insight into patients who are testing positive outside the hospital and then having complications after their positive test. Identifying high risk individuals who are at risk for hospitalization is important when trying to ration scarce medical resources.

## Submission queue:
"COVID-19 DREAM Challenge - Question 2"

## Submission Format:
- The pipeline will expect a 'predictions.csv' file to be written to the '/output' directory, under the name /output/predictions.csv
- Two columns are expected in the predictions.csv: 'person_id' and 'score'.
score must be a number n where 0 <= n <= 1. A prediction file with a number outside this expected range, or a null, NaN, or other non-number, will cause your submission to be invalid.
- The predictions.csv file must have all the person_ids from the person.csv file of the evaluation dataset. A predictions.csv output file, that either has a person_id not in the person.csv file or is missing a person_id from the person.csv file, will invalidate your submission. The confidence level that the patient is COVID positive must be saved to a second column named score. 
- 


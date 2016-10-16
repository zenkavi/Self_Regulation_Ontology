The code in this directory is used to prepare the data for processing.

* process_itemlevel.py: generate separate tsv and json files in the Derived_data
directory for each survey
* process_health.py: process k6_health data
* process_alcohol_drug.py: process data from drug/alcohol questionnaire
* process_demographics.py: process demographic data
* cleanup_items_for_mirt_cv.py: collapse survey responses that occur too infrequently

To run everything, you can use:

make dataprep

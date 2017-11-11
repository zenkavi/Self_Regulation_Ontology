The code in this directory is used to prepare the data for processing.

* process_itemlevel.py: generate separate tsv and json files in the Derived_data
directory for each survey
* process_health.py: process k6_health data
* process_alcohol_drug.py: process data from drug/alcohol questionnaire
* process_demographics.py: process demographic data
* cleanup_items_for_mirt_cv.py: collapse survey responses that occur too infrequently

To run everything, you can use:

    make dataprep

### Metadata

To prepare the metadata, run

    make metadata

The metadata schema is aligned as closely as possible to the
(NIMH Data Archive Data Dictionary) [https://ndar.nih.gov/data_dictionary.html].
It contains the following entries:

For each instrument (task or survey):
* Title: full text name of instrument
* ShortName: short name for instrument (from NDADD)
* URL:
  * CognitiveAtlasURL: URL for instrument in Cognitive Atlas
  * NDAURL: URL for instrument in NDA Data Dictionary


For each item within an instrument:
* ElementName: item identifier
* NDAElementName: identifier for item from NIH Data Archive DD
* DataType: one of ['String','Integer','Float','Boolean']
* VariableScope: one of ['Item','Subscale','Scale']
* VariableDefinition: None, or a dictionary defining the ElementNames included in the subscale and their weights
* VariableUnits: None, or one of ['Ordinal','Seconds','Probability','Percentage','Rate','Count','Other']
* VariableType: one of ['SurveyResponse','SurveySummary','ResponseTime','Accuracy','DDMDriftRate','DDMNondecisionTime','DDMThreshold','LearningRate','Load','DiscountRate','Span']
* Size: max size for strings,empty otherwise
* Required: one of ['Required','Recommended']
* Description: full text description of item (for survey items this is the item text)
* ValueRange: range of possible values
* Notes: For survey items, including semicolon delimited list of choice descriptions

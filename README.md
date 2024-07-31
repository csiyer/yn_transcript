# Transcript Analysis 

Hello! This directory contains code to read trial transcript PDFs and do 2 analyses:
1. `yesno.py`: How many yes/no questions each witness is asked by each examiner, and how many times they are interrupted. (`yesno.ipynb` does the same, but as an interactive notebook--for testing purposes)
2. `word_search.py`: How many times, and where, a certain set of keywords appears in the transcript - by default it will search for the terms in "UPDATED Internal HCRC RJA Glossary of racist language".


`dev` contains the files that I used to develop and test this program, including a dataset of labeled questions that I used to test and evaluate which model to use in the final version.

The first time `yesno.py` runs, it will create a directory called `model_local` containing a downloaded/personal copy of the [question classification model](https://huggingface.co/PrimeQA/tydi-boolean_question_classifier-xlmr_large-20221117) I am using. This model was not written by me, but the local version offers a quick and easy way to query the model to classify questions in the transcript.


**TO RUN**:
- Install necessary dependencies. If you would like to use the same python virtual environment I used (recommended!), follow instructions [here](https://python.land/virtual-environments/virtualenv#How_to_create_a_Python_venv) to create a virtual environment using `Python 3.12.3` and install all libraries within the `requirements.txt` file found in this folder.
- Run script(s)
    - `yesno.py` can be run from any shell / command line with the following format: `python yesno.py /path/to/RT/directory` with the filepath to a folder of RTs. 
        - On the example transcripts I used (a full guilt + penalty phase trial; excluded here to not be public), this takes about 25 minutes.
        - This will produce a CSV output containing the name of each witness, and how many yes/no questions + total questions they are asked by each examiner (defense/prosecution).
    - `word_search.py` can be run with `python word_search.py /path/to/RT/directory`
        - Add `search_terms=/optional/path/to/csv/of/additional/search/terms` to the end of the command if you want to include additional search terms (beyond those found in "UPDATED Internal HCRC RJA Glossary of racist language"--saved to `word_search_terms_default.csv`). These terms should be saved as a CSV file with each word/term, separated with commas. 
        - On the example transcripts, this takes ~1 min to run.


*NOTES*:
- These scripts do not work perfectly!! There are two main reasons:
    1. Every python PDF reader is imperfect, and misses words/lines/characters that are important in parsing the text. ESPECIALLY TRUE for the estimated "true page numbers" read from the top right corner of the PDFs by the word search script--these are often wrong.
    2. Language is hard to classify! Consider the question "Do you remember him telling you anything else?" This is technically a yes/no question, but is really asking for more. Alternatively, "Was he happy or sad that day?" is effectively a yes/no question, but isn't technically. This program and the model used here will not be perfect on questions like these.


Author: Chris Iyer, updated 7/30/24

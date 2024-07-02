# Transcript Yes/No Question Analysis 

Hello! This code will read trial transcript PDFs and for each witness (and each questioner) quantify how many yes/no questions that witness is asked.

The `yesno.py` script is the main product here (`yesno.ipynb` does the same thing, but as interactive notebook--for testing purposes).

`dev` contains the files that I used to develop and test this program, including a set of example trial transcripts, and a dataset of labeled questions that I used to test and evaluate which model to use in the final version).


**TO RUN**:
- Install necessary dependencies. If you would like to use the same python virtual environment I used, follow instructions [here](https://python.land/virtual-environments/virtualenv#How_to_create_a_Python_venv) to create a virtual environment using `Python 3.12.3` and install all libraries within the `requirements.txt` file found in this folder.
- Run `yesno.py` from any shell / command line with the following format: `python yesno.py /path/to/input/directory`

The script will produce an output CSV file containing: for each witness, for each examiner (defense/prosecution), the # of yes/no questions and the # of total questions asked.

*NOTES*:
- OPENAI KEY HAS BEEN PHASED OUT. Previously, this program queried a GPT model and needed an OpenAI API key, but no longer! Now, it uses a [locally-downloaded model](https://huggingface.co/PrimeQA/tydi-boolean_question_classifier-xlmr_large-20221117?text=is+this+a+question%3F) that works better, faster, and cheaper.
- This script does not work perfectly!! There are two main reasons:
    1. Every python PDF reader is imperfect, and misses words/lines/characters that are important in parsing the text. 
    2. Language is hard to classify! Consider the question "Do you remember him telling you anything else?" This is technically a yes/no question, but is really asking for more. Alternatively, "Was he happy or sad that day?" is effectively a yes/no question, but isn't technically. This program and the model used here will not be perfect on questions like these.


Author: Chris Iyer, updated 7/2/24

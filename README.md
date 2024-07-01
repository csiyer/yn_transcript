# Transcript Yes/No Question Analysis 

Hello! This code will read trial transcript PDFs and for each witness (and each questioner) quantify how many yes/no questions that witness is asked.

The `yesno.py` script and the `yesno.ipynb` do the same thing, but one is an interactive notebook (mostly for testing purposes).

`question_dataset.csv` contains my manually-labeled training set of all the questions in the example transcript, along with my manual labels of if it is a yes/no question. This is for (1) testing the candidate models I decided between, and (2) additional training for this pre-trained model.

`example_transcripts` contains the transcripts that I used to test and develop this program

`question_datasets` contains CSVs of questions parsed from the transcripts used for testing and evaluating which model to use 

To run:
- Install necessary dependencies. If you would like to use the same python virtual environment I used, follow instructions [here](https://python.land/virtual-environments/virtualenv#How_to_create_a_Python_venv) to create a virtual environment using `Python 3.12.3` and the `requirements.txt` file found in this folder.
- Run `yesno.py` from any shell / command line with the following format: `python yesno.ipynb /path/to/input/directory`
- NOTE: OPENAI KEY IS BEING PHASED OUT. I will replace GPT queries with queries to a locally-downloaded HuggingFace model that will work better, faster, and cheaper. IGNORE!

    Previous: You will have to generate an OpenAI secret API key on the [OpenAI website](https://platform.openai.com/api-keys). The key can either be saved locally to a file called `key.txt` in this folder, that will be read automatically by the script. Or, you can introduce the key in the command line call below by adding `--openai_key=API_KEY_HERE`.
- Lastly, you can add a flag `--thorough` to run a more thorough version, which queries GPT more times (better) but is a bit slower. *It is recommended to use this flag!*

The script will produce an output CSV file containing: for each witness, for each examiner (defense/prosecution), the # of yes/no questions and the # of total questions asked.

PLEASE NOTE: This does not work perfectly!! There are two reasons:
1. Language is hard to parse! Consider the question "Do you remember him telling you anything else?" This is technically a yes/no question, but is really asking for more. Alternatively, "Was he happy or sad that day?" is effectively a yes/no question, but isn't technically. This program doesn't have clear answers to these and other unclear language structures and thus will not be perfect. 
2. Every python PDF reader is imperfect, and misses words/lines/characters that are important in parsing the text. 


Author: Chris Iyer
Updated: 6/26/24

# Transcript Yes/No Question Analysis 

Hello! This code will read trial transcript PDFs and for each witness (and each questioner) quantify how many yes/no questions that witness is asked.

The `yesno.py` script and the `yesno.ipynb` do the same thing, but one is an interactive notebook (for testing purposes).

To run:
    - Install necessary dependencies. If you would like to use the same python virtual environment I used, follow instructions [here](https://python.land/virtual-environments/virtualenv#How_to_create_a_Python_venv) to create a virtual environment using `Python 3.12.3` and the `requirements.txt` file found here.
    - Run `yesno.py` from any shell / command line with the following format: `python yesno.ipynb /path/to/input/directory`
    - You will have to generate an OpenAI secret API key on the [OpenAI website](https://platform.openai.com/api-keys). The key can either be saved locally to a file called `key.txt` in this folder, that will be read automatically by the script. Or, you can introduce the key in the command line call below by adding `--openai_key=API_KEY_HERE`.
    - Lastly, you can add a flag `--thorough` to run a more thorough version, which queries GPT more times (better) but is much slower.

The script will produce an output CSV file containing: for each witness, for each examiner (defense/prosecution), the # of yes/no questions and the # of total questions asked.

NOTE: ***THIS DOES NOT WORK PERFECTLY***
There are two reasons:
    1. Yes/no questions are identified first if the answer is a yes/no answer. If it is unclear, I send the original
        question as a query to GPT. Some answers, though, really don't look like yes/no answers even if the question was yes/no.
        Language is hard to parse! (and sending every question to GPT as a query blows up the runtime)
    2. Every python PDF reader is imperfect, and misses words/lines/characters that are important in parsing the text. 


Author: Chris Iyer, Habeas Corpus Resource Center
Updated: 6/17/24

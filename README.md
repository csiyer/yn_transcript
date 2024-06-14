# Transcript Yes/No Question Analysis 

Hello! This code will read trial transcript PDFs and for each witness (and each questioner) quantify how many yes/no questions that witness is asked.

The `yesno.py` script and the `yesno.ipynb` do the same thing, but one is an interactive notebook.

To run:
    - go into `yesno.py` and edit the path to a directory of transcript files (currently, running on the `example_transcripts` from Case Thomas).
    - run the python script!
This will result in the writing of a text file containing witness statistics, for each examiner (defense/prosecution), of the # of yes/no questions and the # of total questions asked.


NOTE: ***THIS DOES NOT WORK PERFECTLY***
There are two reasons:
    1. I classify questions as yes-no questions first based on if the answer is a yes/no answer, and if it is unclear, I send the original
        question as a query to GPT. Some answers, though, really don't look like yes/no answers even if the question was yes/no.
        Language is hard to parse! (and sending every question to GPT as a query blows up the runtime)
    2. Every python PDF reader is imperfect, and misses words/lines/characters that are important in parsing the text. 


Author: Chris Iyer
Updated: 6/14/24

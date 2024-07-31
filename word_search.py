"""
This script analyses a transcript to search for specific keywords. It uses certain helper functions from yesno.py
It can be run with: python word_search.py /path/to/RT/directory /path/to/txt/file/of/words
"""
# %pip install -r requirements.txt

import os, re, argparse, csv
from datetime import datetime
from pypdf import PdfReader
from tqdm import tqdm
from collections import defaultdict
from yesno import *


# helper function for reading search terms
def csv_to_arr(csv_path):
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        arr = [word.upper() for row in reader for word in row]
    return arr
     
def parse_inputs(SEARCH_TERM_PATH='./word_search_terms.csv'):
    """
    Parses command line arguments for (1) input directory of RT files, and (2) optional CSV file of additional search terms.
    Returns:
        input_dir (str): Path to the input directory containing RT files.
        search_terms (arr): All search terms, including additional ones, to search for.
    """
    parser = argparse.ArgumentParser(description="Process input paths for RT files and optional search terms.")
    parser.add_argument('path', type=str, nargs='?', default='./dev/example_transcripts', help='Path to the input directory of transcript files.')
    parser.add_argument('--search_terms', type=str, default=None, help='Path to the optional CSV file of additional search terms.')
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise ValueError(f"The input directory '{args.path}' does not exist or is not a directory.")
    if args.search_terms and not os.path.isfile(args.search_terms):
            raise ValueError(f"The CSV file '{args.search_terms}' does not exist or is not a file.")

    print(f'Running program on files at {args.path}')
    print(f'Using default search terms from {SEARCH_TERM_PATH}')

    search_terms = csv_to_arr(SEARCH_TERM_PATH)
    if args.search_terms: 
        print(f'Adding additional search terms from {args.path}')
        search_terms.extend(csv_to_arr(args.search_terms))

    return args.path, search_terms


# helper function for PDF reading: gets the page number from the text of one page of a PDF
def get_page_number(page_text, last_num):
    """
    Pages will either start with a line containing just the page number, or several lines containing all the line numbers, with the last one 
    also containing the page number. This function isolates just the page number.
    """
    lines = [l for l in page_text.split('\n') if bool(re.search(r'\S', l)) ] # filters out lines with only whitespace
    if len(lines) == 0: return 'unknown'
    
    def no_punctuation(t):
        return re.sub(r'[().,?!\-"\':;/]', '', t)

    # first, let's see if the first line is a digit -- this is probably the page number
    firstline = no_punctuation(lines[0]).strip()
    if firstline.isdigit():
        if int(firstline) > 100 or (int(firstline) < 100 and str(last_num+1) in firstline): # digit is bigger than line numbers or exactly equal to the last number plus one: this is probably the page number!
            return firstline
    if re.sub(' ', '', firstline) == str(last_num+1):
        return str(last_num+1)
    
    # otherwise, either the first line isn't a digit, or it is a low number (likely a line number, not a page number)
    # so, we'll loop through and look for the page number
    # a first loop will look for the last_num+1
    for l in lines:
        l = no_punctuation(l).strip()
        if last_num > 50 and str(last_num+1) in l: # if the line is short and we see the next number up from the last, it's safe to conclude it's that
            return str(last_num+1)
        
    # if still not found, we'll look for just this format:
    for l in lines:
        l = no_punctuation(l).strip()
        if re.fullmatch(r'\d+ \d+', l) or re.fullmatch(r'\d+ \d+ \d+', l): # two or three numbers, separated by a space
            return l.split(' ')[-1].strip() # return the last
        
    # didn't find it
    return 'unknown'
    

# Read PDFs to text
def get_lines_pages(INPUT_DIRECTORY_PATH):
    """
    Returns a list where each item is (line_text, true_page_num, file_name, file_page_num
    """
    
    files = [f for f in sorted(os.listdir(INPUT_DIRECTORY_PATH)) if f.endswith('.pdf')]

    lines_with_pages = []
    last_num = 0
    for file in tqdm(files, total=len(files), desc="Reading PDFs..."):
       reader = PdfReader(os.path.join(INPUT_DIRECTORY_PATH, file))

       for file_page_num,page in enumerate(reader.pages):
           page_text = page.extract_text()
           curr_page_num = get_page_number(page_text, last_num)

           lines_with_pages.extend( [(line, curr_page_num, file, file_page_num+1) for line in page_text.split('\n')] )

           if curr_page_num.isdigit():
              last_num = int(curr_page_num)

    return lines_with_pages


## HELPERS FOR WORD SEARCH
def line_starts_with_speaker_name(line):
    if ':' not in line:
        return False
    
    before_colon = line[:line.find(':')]
    if len(before_colon.split()) <= 4:
        return before_colon
    return False

def guess_speaker(lines, i, current_witness, current_examiner):
    # searches previous lines to guess who spoke the current line
    lines_to_search = lines[i-30:i+1]
    for l in reversed(lines_to_search):
        starting_speaker_name = line_starts_with_speaker_name(l.strip())
        if is_answer(l):
            return current_witness
        elif starts_question(l, 'current_examiner'):
            return current_examiner
        elif starting_speaker_name:
            return starting_speaker_name
    return 'unknown'



def word_search(lines_with_pages, search_terms, DEFAULT_EXAMINER_KEY):
    lines = [l for l,_,_,_ in lines_with_pages]

    results_totals = defaultdict(int)
    results_df = 'Search term,True page number,File name,Within-file page number,Speaker\n'

    current_witness = ''
    current_witness_side = ''
    current_examination = ''
    current_examiner = ''

    for i,(currline,true_page,filename,file_page) in enumerate(lines_with_pages):
  
        # keep track of these so we can guess the speaker of the word
        if line_is_witness_identifier(lines, i):
            current_witness = clean_simple_line(currline)
            current_witness_side = who_presents_this_witness(lines, i)

        elif line_is_examination_identifier(lines, i):
            current_examiner = ''
            current_examination = clean_simple_line(currline)

        elif line_is_examiner_identifier(currline):
            current_examiner = clean_examiner_name(currline)

        for term in search_terms:
            if f' {term} ' in currline: # term surrounded with spaces, so it's not just part of another word
                results_totals[term] += 1

                if current_examiner == '': # we may have missed this before, and have to guess now
                    current_examiner = guess_examiner(current_witness_side, current_examination, DEFAULT_EXAMINER_KEY) 
                speaker = guess_speaker([l for l,_,_,_ in lines_with_pages], i, current_witness, current_examiner)

                results_df += f'{term},{true_page},{filename},{file_page},{speaker}\n'

    print(f'Finished searching transcript, saving output.')
    return dict(results_totals), results_df


def write_output(results_totals, results_df, INPUT_DIRECTORY_PATH, unique_id):
    """Results are written to two CSV files, one with the total occurrences of each term and one with the individual search results"""

    totals_path = os.path.join(INPUT_DIRECTORY_PATH, f'word_search_totals_{unique_id}.csv')
    totals_text = 'Term,Count,'
    for term,count in results_totals.items():
        totals_text += f'{term},{count}\n'
    with open(totals_path, 'w') as file:
        file.write(totals_text)
    
    df_path = os.path.join(INPUT_DIRECTORY_PATH, f'word_search_results_{unique_id}.csv')
    with open(df_path, 'w') as file:
        file.write(results_df)


# RUN SCRIPT!
if __name__ == "__main__":
    start_time = datetime.now()

    INPUT_DIRECTORY_PATH, search_terms = parse_inputs()
    lines_with_pages = get_lines_pages(INPUT_DIRECTORY_PATH)
    DEFAULT_EXAMINER_KEY = get_default_examiners([l for l,_,_,_ in lines_with_pages])
    
    results_totals, results_df = word_search(lines_with_pages, search_terms, DEFAULT_EXAMINER_KEY)

    unique_id = get_unique_id([l for l,_,_,_ in lines_with_pages])
    write_output(results_totals, results_df, INPUT_DIRECTORY_PATH, unique_id)

    end_time = datetime.now()
    elapsed_minutes = (end_time - start_time).total_seconds() / 60
    print(f"Script took {elapsed_minutes:.2f} minutes")
# %pip install pypdf
# %pip install tqdm
# %pip install transformers
# %pip install torch
# %pip install openai
# %pip install joblib

import os, re, argparse
from datetime import datetime
from pypdf import PdfReader
from tqdm import tqdm
from openai import OpenAI
from joblib import Parallel, delayed
from collections import defaultdict

############################### PROCESS COMMAND LINE ARGUMENTS ############################

parser = argparse.ArgumentParser(description='Transcript yes/no analysis.')
parser.add_argument('path', type=str, help='Path to the input directory of transcript files.')
parser.add_argument('-o', '--openai_key', dest='openai_key', help='OpenAI API key (alternative: save to a file called key.txt in this directory!)')
parser.add_argument('--thorough', dest='thorough_bool', action='store_true', help='If flagged, this script will query GPT more times--a more thorough but slower verison.')
args = parser.parse_args()

INPUT_DIRECTORY_PATH = args.path
THOROUGH_BOOL = args.thorough_bool
if args.openai_key:
    OPENAI_KEY = args.openai_key
else: 
    with open('key.txt','r') as f: 
        OPENAI_KEY = f.read()
print(OPENAI_KEY)

print(f'Accessing input files at: {INPUT_DIRECTORY_PATH}')
print(f'Thorough version?: {THOROUGH_BOOL}')
print(f'API KEY: {OPENAI_KEY}')

############################### DATA LOADING AND PROCESSING ###############################

files = [f for f in sorted(os.listdir(INPUT_DIRECTORY_PATH))]

# Read all the PDFs into a huge string, and then split into a big list of lines
entire_transcript = ""
print('Processing PDFs to text...')
for file in tqdm(files, total=len(files)):
  reader = PdfReader(os.path.join(INPUT_DIRECTORY_PATH, file))
  for page in reader.pages:
    entire_transcript += page.extract_text() + '\n'
print('finished!\n')

# separate into lines, and filter out the ones that are just line numbers, e.g. "24 "
lines = entire_transcript.split('\n')
lines = [line for line in lines if not re.match(r'^[\d\s]*$', line)]



############################### TRANSCRIPT ANALYSIS #######################################

# HELPER FUNCTIONS
def delete_numbers_whitespace(text):
    return re.sub(r'[\d\t\n]', '', text).strip()

def clean_simple_line(line):
    # removes punctuation/numbers/non-letters
    return re.sub(r'[^a-zA-Z\s]', '', line).upper().strip() 

def line_is_witness_identifier(lines, i):
    line = clean_simple_line(lines[i])
    words = line.split(' ')
    return len(words) < 6 and i < len(lines)-1 and ' as a witness' in lines[i+1].lower()

def who_presents_this_witness(lines, witness_line_i): 
    for j in range(witness_line_i+1, witness_line_i+5): # scan the next few lines for keywords
        if 'people' in lines[j].lower():
            return 'people'
        if 'defense' in lines[j].lower() or 'defendant' in lines[j].lower():
            return 'defense'
    return 'unknown'

def line_is_examiner_identifier(line):
    # each examination begins with a line like "By Mr. Smith:"  
    line = re.sub(r'\d+', '', line).strip() # eliminate leading numbers + whitespace. we don't want clean_simple_line because we want to keep colon if there is one
    return len(line.split(' ')) < 6 and line[0:2].lower() == 'by' and line.strip()[-1] == ':'

def clean_examiner_name(examiner_line):
    if '.' in examiner_line and ':' in examiner_line:
        name_substr = examiner_line[examiner_line.find('.'):examiner_line.find(':')]
        return clean_simple_line(name_substr)
    
    name_followed_by_colon = [w for w in examiner_line.split(' ') if ':' in w][0]
    return clean_simple_line(name_followed_by_colon)

def line_is_examination_identifier(lines, i):
    line = clean_simple_line(lines[i])
    return len(line.split()) < 4 and 'EXAMINATION' in line and ('CROSS' in line or 'DIRECT' in line) and i < len(lines)-1 and ( 
        line_is_examiner_identifier(lines[i+1]) or lines[i+1].startswith('Q.') or lines[i+1].startswith('A.')
        )

def is_answer(line):
    return  re.sub(r'[^a-zA-Z. ]', '', line).strip().startswith('A. ') # or line.strip().startswith('THE WITNESS:')

def starts_question(text, current_examiner):
    return any(item in text for item in ['Q. ', 'Q . ', 'Q• ', 'Q • ', current_examiner+':']) # and '?' in text

def guess_previous_question(lines, i):
    # if the previous question was not read in properly with 'Q.', then we want to parse what the question was when we hit an answer
    possible_question = lines[i-1]
    for prevline in reversed(lines[i-11:i-2]): # check previous 10 lines for question, stop when we hit punctuation
        if prevline.strip().endswith(('.','!','?')) or any(item in prevline for item in ['A. ', 'A . ']):
            break
        possible_question = prevline + possible_question
    return delete_numbers_whitespace(possible_question) 

def is_yes_no_answer(lines,i,current_examiner):
    # querying chatGPT for yes/no questions is very time consuming, so we only want to do it if we cannot tell from the answer itself
    answer = lines[i]
    for nextline in lines[i+1:i+10]: # check next lines and add continuance of answer if necessary
        if nextline.strip().endswith(('.','!','?')) or is_answer(nextline) or starts_question(nextline, current_examiner):
            break
        answer += nextline

    answer_split = re.sub(r'[^A-Za-z ]', '', answer).upper().strip().split(' ')
    if any(item in answer_split for item in ['YES', 'YEAH', 'YEP', 'NO', 'NOPE', 'UHHUH', 'UHUH', 'UMHUM', 'UMUM']) or 'NOT' in answer_split[0:3]:
        if len(answer_split) < 8:
            return 'yes'
        return 'maybe'
    if len(answer_split) < 5:
        return 'maybe'
    return 'no'

def is_yes_no(question, OPENAI_KEY):
    # returns true if the question is a yes/no question. queries GPT to do so!
    client = OpenAI(api_key=OPENAI_KEY)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at identifying yes/no questions, and at analyzing courtroom transcripts."},
            {"role": "user", "content": f"Is the following question a yes or no question? Respond with 'yes' or 'no':\n\nQuestion: {question}"}
        ]
    )
    return 'yes' == completion.choices[0].message.content.strip().lower()


# there are some instances where the 'examiner identification' line isn't read properly by the pdf reader
# for these, we need a default guess for who the examiner is.
# so, we'll find the first direct examination for each side (people/defense) and save who the examiner is -- this is a good guess

DEFAULT_EXAMINER_KEY = {'people': '', 'defense': ''}
found = {'people': False, 'defense': False}
for i in range(len(lines)):
    if line_is_witness_identifier(lines, i):
        side = who_presents_this_witness(lines, i)
        if side != 'unknown' and not found[side]:
            # search the next 200 lines for a direct exam, if found one then get the examiner ID
            direct_exam_found = True
            for j,line in enumerate(lines[i:i+200]):
                if line_is_examination_identifier(lines, i+j) and 'DIRECT' in line:
                    direct_exam_found = True
                if direct_exam_found and line_is_examiner_identifier(line):
                    DEFAULT_EXAMINER_KEY[side] = clean_examiner_name(line)
                    found[side] = True
                    break
    if found['people'] and found['defense']: 
        break
    
print('Default examiner default guesses: ', DEFAULT_EXAMINER_KEY, '\nIf these look incorrect, please stop and revise.')

def guess_examiner(witness_side, current_examination):
    print('Examiner not found, guessing from previous records (this message should be rare).')
    if 'DIRECT' in current_examination.upper():
        return DEFAULT_EXAMINER_KEY[witness_side]
    elif 'CROSS' in current_examination.upper():
        other_side = [i for i in DEFAULT_EXAMINER_KEY.keys() if i != witness_side][0]
        return DEFAULT_EXAMINER_KEY[other_side]
    return 'error: unknown examiner'


#### LOOP THROUGH AND ANALYZE TRANSCRIPT

# split this process into ranges of lines corresponding to each witness (to parallelize for speed)
def process_one_range(range_of_lines, lines):
    # initialize variables to be used as we loop
    current_witness = ''
    current_witness_side = ''
    current_examination = ''
    current_examiner = ''
    active_question = ''
    gpt_query_idxs = []
    local_name_to_stats = defaultdict(lambda: defaultdict(lambda: {'total_questions': 0, 'yes_no_questions': 0})) # use default dict so we don't have to check if key already exists

    for i in range_of_lines:
        line = lines[i]

        if line_is_witness_identifier(lines, i):
            current_witness = clean_simple_line(line)
            current_witness_side = who_presents_this_witness(lines, i)
            active_question = '' # just in case we get carried away

        elif line_is_examination_identifier(lines, i):
            current_examiner = ''
            current_examination = clean_simple_line(line)
            active_question = ''

        elif line_is_examiner_identifier(line):
            current_examiner = clean_examiner_name(line)
            active_question = ''

        elif starts_question(line, current_examiner): # when we eventually hit an answer, I want the active_question to contain everything since the last question started
            active_question = line # start adding to active_question

        elif is_answer(line):
            # we may need to guess necessary preceding info if there was an error in pdf reading
            if current_examiner == '': 
                current_examiner = guess_examiner(current_witness_side, current_examination) 
            if active_question == '': 
                active_question = guess_previous_question(lines, i) 
                
            if '?' in active_question: # to rule out things like "Q. Good morning."
                local_name_to_stats[current_witness][current_examiner]['total_questions'] += 1

                yes_no = is_yes_no_answer(lines, i, current_examiner)

                if yes_no == 'yes': 
                    local_name_to_stats[current_witness][current_examiner]['yes_no_questions'] += 1

                # if we're being maximally thorough, or we're not but the yes_no function returned "maybe" -- query GPT 
                elif THOROUGH_BOOL or (not THOROUGH_BOOL and yes_no=='maybe'): 
                    gpt_query_idxs.append(i)
                    local_name_to_stats[current_witness][current_examiner]['yes_no_questions'] += is_yes_no(active_question, OPENAI_KEY) # 1 if true, 0 if false

            active_question = '' # reset

        elif active_question:
            active_question += line # if we started a question, add this line. resets at every answer or special identifying line

    return dict(local_name_to_stats), gpt_query_idxs


def merge_dicts(list_of_dicts):
    main = defaultdict(lambda: defaultdict(lambda: {'total_questions': 0, 'yes_no_questions': 0}))
    for d in list_of_dicts:
        for witness, subdict in d.items():
            for examiner, values in subdict.items():
                main[witness][examiner]['total_questions'] += values['total_questions']
                main[witness][examiner]['yes_no_questions'] += values['yes_no_questions']
    return {k:dict(v) for k,v in main.items()}


#### PROCESS ENTIRE TRANSCRIPT (in parallel)
# find witness IDs to break the transcript into chunks, to hand each chunk to a separate core
witness_breaks = [i for i in range(len(lines)) if line_is_witness_identifier(lines, i)] 
witness_ranges = [range(0, witness_breaks[i]) if i == 0 else range(witness_breaks[i-1], witness_breaks[i]) for i in range(len(witness_breaks))]

results = Parallel(n_jobs=-1)(delayed(process_one_range)(r, lines) for r in witness_ranges)

name_to_stats = merge_dicts([r[0] for r in results])
all_gpt_query_idxs = [idx for sublist in [r[1] for r in results] for idx in sublist]

print(f'Finished transcript, saving output.')


############################### OUTPUT TXT FILE ###########################################

output_csv_text = 'Witness,Examiner,Total questions,Yes/No Questions,Yes/No Percentage\n'

for name,values in name_to_stats.items():
    for examiner, stats in values.items():
        output_csv_text += f'{name},{examiner},{stats["yes_no_questions"]},{stats["total_questions"]},'
        try:
            percentage = round(stats['yes_no_questions'] / stats['total_questions'] * 100, 2)
        except:
             percentage = 'N/A'
        output_csv_text += f'{percentage}\n'
    output_csv_text += '\n'

def get_unique_id(lines):
    for l in lines[0:30]:
        if 'NO. ' in l: # case number
            return 'case-' + l.split('NO. ')[1].strip()
    return datetime.now().strftime('date-%Y-%m-%d_%H-%M')

thorough_tag = 'thorough' if THOROUGH_BOOL  else 'nonthorough'
output_path = os.path.join(INPUT_DIRECTORY_PATH, f'yesno_analysis_{get_unique_id(lines)}_{thorough_tag}.csv')

with open(output_path, 'w') as file: # CHANGE FILENAME TO UNIQUE ID
    file.write(output_csv_text)
"""
This script analyses a transcript to quantify how many yes/no questions each witness is asked. 
It uses a question classification model someone else wrote.
It can be run with: python yesno.py /path/to/RT/directory
"""
# %pip install -r requirements.txt

import os, re, argparse
from datetime import datetime
from pypdf import PdfReader
from tqdm import tqdm
from collections import defaultdict
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# all code is now factored into functions, which are all called at the bottom of this script

############################### PROCESS COMMAND LINE ARGUMENTS ############################

def parse_input_path():
    parser = argparse.ArgumentParser(description='Transcript yes/no analysis.')
    parser.add_argument('path', type=str, nargs='?', default='./dev/example_transcripts', help='Path to the input directory of transcript files.')
    args = parser.parse_args()
    if not os.path.isdir(args.path):
        raise ValueError(f"The input directory '{args.path}' does not exist or is not a directory.")
    print(f'Running program on files at: {args.path}')
    return args.path


############################### DATA LOADING AND PROCESSING ###############################

def get_lines(INPUT_DIRECTORY_PATH):

    files = [f for f in sorted(os.listdir(INPUT_DIRECTORY_PATH)) if f.endswith('.pdf')]

    # Read all the PDFs into a huge string, and then split into a big list of lines
    entire_transcript = ""
    for file in tqdm(files, total=len(files), desc="Processing PDFs to text..."):
        reader = PdfReader(os.path.join(INPUT_DIRECTORY_PATH, file))
        for page in reader.pages:
            entire_transcript += page.extract_text() + '\n'

    # Separate into lines, and filter out the ones that are just line numbers, e.g. "24 "
    lines = entire_transcript.split('\n')
    lines = [line for line in lines if not re.match(r'^[\d\s]*$', line)]

    return lines


############################### LOAD QUESTION CLASSIFIER ###################################

classifier = None
def init_classifier():
    # load question classification model from local. Or, if local doesn't exist, download from HuggingFace and save to local
    local_model_path = './model_local'
    model_name = 'PrimeQA/tydi-boolean_question_classifier-xlmr_large-20221117'

    try: 
        # try to load local model
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        print(f"Loaded model from {local_model_path}")
    except: 
        # If loading locally fails, download and save the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Save the model locally
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)
        print(f"Downloaded and saved model to {local_model_path}")

    global classifier
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


############################### ANALYSIS HELPER FUNCTIONS  ################################

def remove_whitespace(text):
    return re.sub(r'[\t\n]', ' ', text)

def only_letters_numbers_normal_punctuation(text):
    out = re.sub(r'[^a-zA-Z0-9().,?!\-"\':;/ ]', '', text)
    if out.startswith('.'): out = out[1:].strip()
    return out

def clean_simple_line(line):
    # removes punctuation/numbers/non-letters
    return re.sub(r'[^a-zA-Z\s]', '', line).upper().strip() 

def clean_question(question):
    clean =  re.sub(r'[\t\n\s+]', ' ', question) # tabs, newlines, and extra spaces
    clean = re.sub(r'^\d+\s*', '', clean) # leading number and whitespace (line number)
    clean = re.sub(r'["]|Q |Q. |Q . |Q• |Q • |Q- |', '', clean)  # Question marker
    return clean.upper().strip()

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
    starts_a = re.sub(r'[^a-zA-Z. ]', '', line).strip().startswith('A. ') or re.sub(r' ', '', line).strip().startswith('.A.')
    not_time = not re.sub(r'[^a-zA-Z\.]', '', line).startswith('A.M.')
    return starts_a and not_time
    # return re.sub(r'[^a-zA-Z. ]', '', line).strip().startswith('A. ') and not re.sub(r'[^a-zA-Z\.]', '', line).startswith('A.M.') # or line.strip().startswith('THE WITNESS:')

def starts_question(text, current_examiner):
    return any(item in text for item in ['Q. ', 'Q . ', 'Q• ', 'Q • ', 'Q- ', current_examiner+':']) # and '?' in text 

def get_previous_question(lines, i, current_examiner):
    # if the previous question was not read in properly with 'Q.', then we want to parse what the question was when we hit an answer
    possible_question = lines[i-1]
    for j,prevline in enumerate(reversed(lines[i-11:i-1])): # check previous 10 lines for question, stop when we hit punctuation
        prevline = remove_whitespace(prevline)
        if starts_question(possible_question, current_examiner):
            break
        if prevline.strip().endswith(('.','!','?', ')')) or is_answer(prevline) or line_is_witness_identifier(lines, i-2-j) or line_is_examiner_identifier(prevline) or line_is_examination_identifier(lines, i-2-j):
            break
        possible_question = prevline + possible_question
    return only_letters_numbers_normal_punctuation(possible_question)

def is_yes_no_answer(lines, i, current_examiner):
    # querying the model is more time-consuming, so we only want to do it if we cannot tell from the answer itself
    answer = lines[i]
    for nextline in lines[i+1:i+10]: # check next lines and add continuance of answer if necessary
        if nextline.strip().endswith(('.','!','?')) or is_answer(nextline) or starts_question(nextline, current_examiner):
            break
        answer += nextline
    answer_split = re.sub(r'[^A-Za-z ]', '', answer).upper().strip().split(' ')
    if any(item in answer_split for item in ['YES', 'YEAH', 'YEP', 'NO', 'NOPE', 'UHHUH', 'UHUH', 'UMHUM', 'UMUM']) or 'NOT' in answer_split[0:3]:
        if len(answer_split) < 8:
            return True
    return False

def is_yes_no(question):
    # queries question classification model whether the question is a yes/no question or not, returns boolean
    result = classifier(question)[0]['label']
    if not result in ['LABEL_0', 'LABEL_1']:
        return 'ERROR: unexpected classification result'
    return result == 'LABEL_0' # model returns 'LABEL_0' for yes/no questions and 'LABEL_1' for other questions

#### for interruptions
def within_answer(lines, i, current_examiner, DEFAULT_EXAMINER_KEY):
    # is this line part of an answer? useful for identifying interruptions
    for line in reversed(lines[i-50:i+1]): # loop through previous 20 lines
        if is_answer(line) or 'THE WITNESS:' in line:
            return True
        if starts_question(line, current_examiner) or 'THE COURT:' in line or any([name in line for name in DEFAULT_EXAMINER_KEY.values()]):
            return False
    return False ## assuming answers aren't usually longer than this many lines


def who_says_next_line(lines,i, current_examiner):
    # useful for seeing who interrupts
    if lines[i+1].strip().startswith('Q') or starts_question(lines[i+1], current_examiner):
        return current_examiner
    if 'THE COURT' in lines[i+1]:
        return 'COURT'
    if ':' in lines[i+1]:
        return clean_examiner_name(lines[i+1])
    return None

# DEFAULT EXAMINER GUESSES
# there are some instances where the 'examiner identification' line isn't read properly by the pdf reader
# for these, we need a default guess for who the examiner is.
# so, we'll find the first direct examination for each side (people/defense) and save who the examiner is -- this is a good guess

def get_default_examiners(lines):
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
    return DEFAULT_EXAMINER_KEY

def guess_examiner(witness_side, current_examination, DEFAULT_EXAMINER_KEY):
    print('Examiner not found, guessing from previous records (this message should be rare).')
    if 'DIRECT' in current_examination.upper():
        return DEFAULT_EXAMINER_KEY[witness_side]
    elif 'CROSS' in current_examination.upper():
        other_side = [i for i in DEFAULT_EXAMINER_KEY.keys() if i != witness_side][0]
        return DEFAULT_EXAMINER_KEY[other_side]
    return 'error: unknown examiner'


###############################  TRANSCRIPT ANALYSIS  #####################################

# loop through transcript to identify questions, and save the ones we need to classify as yes/no questions or not
def analyze_transcript(lines, DEFAULT_EXAMINER_KEY):

    current_witness = ''
    current_witness_side = ''
    current_examination = ''
    current_examiner = ''
    name_to_stats = defaultdict(lambda: defaultdict(lambda: {'total_questions': 0, 'yes_no_questions': 0, 'interruptions': 0})) # use default dict so we don't have to check if key already exists
    questions_to_query = [] # to parallelize later

    for i,line in enumerate(lines):

        if line_is_witness_identifier(lines, i):
            current_witness = clean_simple_line(line)
            current_witness_side = who_presents_this_witness(lines, i)

        elif line_is_examination_identifier(lines, i):
            current_examiner = ''
            current_examination = clean_simple_line(line)

        elif line_is_examiner_identifier(line):
            current_examiner = clean_examiner_name(line)

        elif is_answer(line):

            if current_examiner == '': # we may have missed this before, and have to guess now
                current_examiner = guess_examiner(current_witness_side, current_examination, DEFAULT_EXAMINER_KEY) 

            question = get_previous_question(lines, i, current_examiner) 
                
            if '?' in question: # to rule out things like "Q. Good morning."
                name_to_stats[current_witness][current_examiner]['total_questions'] += 1
                
                if is_yes_no_answer(lines, i, current_examiner): # this function catches answers that are easy to see are yes/no answers, so we don't have to waste time querying the model
                    name_to_stats[current_witness][current_examiner]['yes_no_questions'] += 1
                else:
                    # not able to identify it as yes/no, add this question (and identifying information) to the pile of questions to query later
                    questions_to_query.append((clean_question(question), current_witness, current_examiner))

        # identify an interruption
        if line.strip().endswith('--') and within_answer(lines, i, current_examiner, DEFAULT_EXAMINER_KEY):
            next_speaker = who_says_next_line(lines, i, current_examiner)
            if next_speaker:
                name_to_stats[current_witness][next_speaker]['interruptions'] += 1


    # these fields aren't relevant for the court (just interruptions)
    for witness,stats in name_to_stats.items():
        if 'COURT' in stats.keys():
            stats['COURT']['total_questions'] = None
            stats['COURT']['yes_no_questions'] = None
                    

    print(f'Finished reading transcript, querying model with questions.')
    # execute question classification (in parallel)
    with ThreadPoolExecutor() as executor:
        classifier_results = list(executor.map(is_yes_no, [q for q,_,_ in questions_to_query]))

    # add the results of these queries to our stats
    for (_,witness,examiner),result in zip(questions_to_query, classifier_results):
        name_to_stats[witness][examiner]['yes_no_questions'] += result

    print(f'Finished analyzing transcript, saving output.')
    return name_to_stats


############################### OUTPUT TXT FILE ###########################################

def get_unique_id(lines):
        datetag = datetime.now().strftime('date-%Y-%m-%d_%H-%M')
        for l in lines[0:30]:
            if 'NO. ' in l: # case number
                return f"case-{l.split('NO. ')[1].strip()}_{datetag}"
        return datetag

def write_output(name_to_stats, INPUT_DIRECTORY_PATH, unique_id):
    output_csv_text = 'Witness,Examiner,Yes/No Questions,Total questions,Yes/No Percentage,Interruptions\n'

    for name,values in name_to_stats.items():
        for examiner, stats in values.items():
            output_csv_text += f'{name},{examiner},{stats["yes_no_questions"]},{stats["total_questions"]},'
            try:
                percentage = round(stats['yes_no_questions'] / stats['total_questions'] * 100, 2)
            except:
                percentage = 'N/A'
            output_csv_text += f'{percentage},'
            output_csv_text += f'{stats["interruptions"]}'
        output_csv_text += '\n'

    output_path = os.path.join(INPUT_DIRECTORY_PATH, f'yesno_analysis_{unique_id}.csv')

    with open(output_path, 'w') as file:
        file.write(output_csv_text)


############################### RUN ALL ###################################################

if __name__ == "__main__":
    start_time = datetime.now()
            
    INPUT_DIRECTORY_PATH = parse_input_path()
    lines = get_lines(INPUT_DIRECTORY_PATH)
    init_classifier()
    DEFAULT_EXAMINER_KEY = get_default_examiners(lines)

    name_to_stats = analyze_transcript(lines, DEFAULT_EXAMINER_KEY)

    unique_id = get_unique_id(lines)
    write_output(name_to_stats, INPUT_DIRECTORY_PATH, unique_id)

    end_time = datetime.now()
    elapsed_minutes = (end_time - start_time).total_seconds() / 60
    print(f"Script took {elapsed_minutes:.2f} minutes")
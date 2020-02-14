import sys
import argparse
import os
import json
import re
import spacy
import html

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  # replace newlines and tabs with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
        modComm = re.sub(r"\t{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
        modComm = modComm.replace(u'\xa0', ' ') # \xa0 sticks around for some reason as a unicode space
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|http|www).*\s", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub("\s\s+", " ",modComm)
        modComm = modComm.lstrip() # remove leading and trailing spaces to avoid /_SP during lemmization
    
    # Get the doc for modComm
    doc = nlp(modComm)
    
    # punct_chars are the characters spacy uses to demarcate sentence endings 
    punct_chars = sentencizer.punct_chars
    
    # And we store the tagged sentence in the list_for_pos array, which we then join
    # to turn back into a string
    list_for_pos = []
    
    i = 1
    for token in doc:
        if token.text in punct_chars or i == len(doc): # If it's a sentence end via period, or the last word, with no actual punctuation 
            list_for_pos.append(token.lemma_ + '/' + token.tag_ + '\n') 
        else:
            list_for_pos.append(token.lemma_ + '/' + token.tag_)
        i += 1
        
    modComm = " ".join(list_for_pos)
    modComm = re.sub(r"\n ", "\n", modComm) # Remove space between \n and next word
    return modComm


def main(args):
    ###### delete this when the time comes. this is replacement for argparse `
    ID = args.ID
    MAX = args.max
    output = args.output
    ############
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            
            # TODO: select appropriate args.max lines
            start_line = ID % len(data) # what line to start reading 
            for i in range(MAX):
                element = start_line + i
                if element >= len(data): # if you go over the # of lines
                    element = element % len(data) # loop back to beginning 
                j = json.loads(data[element])
                j['cat'] = file # add the category of alt/center/right/left
                modComm = preproc1(j['body']) # get the edited comment
                j['body'] = modComm 
                allOutput.append(j)

            
    fout = open(output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)

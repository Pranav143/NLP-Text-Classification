import numpy as np
import argparse
import json
import pandas as pd

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    ''' 
    return_vec = np.zeros(173)
    # First things first, find the number of \n tokens since that tells us the number of sentences, and also, manipulating the string later on removes the \n tokens
    # We need this for a later feature of number of sentences
    num_sentences = comment.count('\n')
 
    # (1) Extract features that rely on capitalization.
    # Note that the pos tags are capitalized, so we need to ensure we only check
    # the actual string. Thus, we split each token by "/" and check the first element
    split_list = comment.split()
    capital_count = sum(1 for c in split_list if c.split("/")[0].isupper() and len(c.split("/")[0]) >= 3)
    return_vec[0] += capital_count
    # Now, a lot of acronyms etc were not turned lower case after lemmization. We deal with them here
    lowercase_list = [c.split("/")[0].lower() + "/" + c.split("/")[1] if "/" in c else c for c in split_list]
    comment = " ".join(lowercase_list)
    
    """Exctracting features that do not rely on iteration"""
    
    # (5) Number of coordinating conjunctions
    num_cc = comment.count('CC') # POS tag CC represents coordinating conjunctions
    return_vec[4] += num_cc
    
    # (6) Number of past tense verbs 
    num_verb_past = comment.count('VBD')
    return_vec[5] += num_verb_past
    
    # (7) Number of future tense verbs have to also count number of occurences of "going+to+VB"
    # This also requires iteration, however, specifically a sliding window
    num_verb_future = comment.count("'ll") + comment.count("will") + comment.count("gonna") 
    for i in range(len(lowercase_list) - 2):
        if lowercase_list[i].split("/")[0] == "going" and lowercase_list[i+1].split("/")[0] == "to" and lowercase_list[i+2].split("/")[1] == "VB":
            num_verb_future += 1
    return_vec[6] += num_verb_future
    
    # (8) Number of commas
    num_commas = comment.count(",")
    return_vec[7] += num_commas / 2 # we divide by 2 since the comma pos tag is also a comma 
    
    # (15) Average length of sentences. Simply, number tokens/number sentences
    if num_sentences != 0:
        average_sentence_length = len(lowercase_list) / num_sentences
        return_vec[14] += average_sentence_length
    else:
        return_vec[14] += 0 
    
    # (17) Number of sentences. This was calculated first, so we just use that here
    return_vec[16] += num_sentences
    """Exctrating features that rely on iteration"""
    
    num_first_pnouns = 0
    num_second_pnouns = 0
    num_third_pnouns = 0
    num_verb_future = 0
    num_mult_char = 0
    num_common_nouns = 0
    num_proper_nouns = 0
    num_adverbs = 0
    num_wh_words = 0
    num_slang = 0
    len_nonpunct_token = 0
    num_nonpunct_tokens = 0
    
    # (18) - (20) and (24) - (26) All in one pass
    aoa_count = 0
    aoa_word_count = 0
    img_count = 0
    img_word_count = 0
    fam_count = 0
    fam_word_count = 0
    
    vmean_count = 0
    vmean_word_count = 0
    amean_count = 0
    amean_word_count = 0
    dmean_count = 0
    dmean_word_count = 0
    
    # Lists to store values so that we can calculate stddev
    aoa_l = []
    img_l = []
    fam_l = []
    
    vmean_l = []
    amean_l = []
    dmean_l = []
    
    for token in lowercase_list:
        if "/" not in token: # This is prolly an error of a word since it has no tag
            continue
        word = token.split("/")[0]
        tag = token.split("/")[1]
        # (2) Number of 1st person pronouns
        num_first_pnouns += 1 if word in FIRST_PERSON_PRONOUNS else 0

        # (3) Number of 2nd person pronouns
        num_second_pnouns +=  1 if word in SECOND_PERSON_PRONOUNS else 0
    
        # (4) Number of 3rd person pronounds
        num_third_pnouns += 1 if word in THIRD_PERSON_PRONOUNS else 0
           
        # (9) Number of multi-character punctuation tokens 
        num_mult_char += 1 if tag == ":" and len(word) >= 2 else 0
    
        # (10) Number of common nouns. Can't use comment.count() since NN counts might conflict with NNS count and vice versa
        num_common_nouns += 1 if tag in ("NN" ,"NNS") else 0
    
        # (11) Number of proper nouns
        num_proper_nouns += 1 if tag in ("NNP" , "NNPS") else 0
        
        # (12) Number of adverbs 
        num_adverbs += 1 if tag in ("RB", "RBR" , "RBS") else 0
        
        # (13) Number of wh-words
        num_wh_words += 1 if tag in ("WP$" ,"WDT" , "WP" , "WRB") else 0
        
        # (14) Number of slang acronyms 
        num_slang += 1 if word in SLANG else 0
    
        # (16) Average token length excluding punctuation tokens
        len_nonpunct_token += len(word) if tag not in ("#" , "$" , "." , "," , ":" , "(" , ")" ,'"' , "'" , "-LRB-" , "-RRB-") else 0
        num_nonpunct_tokens += 1 if tag not in ("#" , "$" , "." , "," , ":" , "(" , ")" ,'"' , "'" , "-LRB-" , "-RRB-") else 0

        # All the AoA, IMG, FAM, VMEAN, AMEAN, DMEAN stuff 
        aoa = aoa_dict.get(word)
        img = img_dict.get(word)
        fam = fam_dict.get(word)
        
        vmean = vmean_dict.get(word)
        amean = amean_dict.get(word)
        dmean = dmean_dict.get(word)
        
        if aoa != None:
            aoa_count += aoa
            aoa_word_count += 1
            aoa_l.append(aoa)
        if img != None:
            img_count += img
            img_word_count += 1
            img_l.append(img)
        if fam != None:
            fam_count += fam
            fam_word_count += 1
            fam_l.append(fam)
        if vmean != None:
            vmean_count += vmean
            vmean_word_count += 1
            vmean_l.append(vmean)
        if amean != None:
            amean_count += amean
            amean_word_count += 1
            amean_l.append(amean)
        if dmean != None:
            dmean_count += dmean
            dmean_word_count += 1
            dmean_l.append(dmean)

    if aoa_word_count == 0:
        avg_aoa = 0
        return_vec[17] += avg_aoa
    else:
        avg_aoa = aoa_count / aoa_word_count 
        return_vec[17] += avg_aoa
    
    if img_word_count == 0:
        avg_img = 0
        return_vec[18] += avg_img
    else:
        avg_img = img_count / img_word_count 
        return_vec[18] += avg_img
    
    if fam_word_count == 0:
        avg_fam = 0
        return_vec[19] += avg_fam
    else:
        avg_fam = fam_count / fam_word_count 
        return_vec[19] += avg_fam
    
    if vmean_word_count == 0:
        avg_vmean = 0
        return_vec[23] += avg_vmean
    else:
        avg_vmean = vmean_count / vmean_word_count 
        return_vec[23] += avg_vmean
    
    if amean_word_count == 0:
        avg_amean = 0
        return_vec[24] += avg_amean
    else:
        avg_amean = amean_count / amean_word_count 
        return_vec[24] += avg_amean
    
    if dmean_word_count == 0:
        avg_dmean = 0
        return_vec[25] += avg_dmean
    else:
        avg_dmean = dmean_count / dmean_word_count 
        return_vec[25] += avg_dmean
    
    # (21) - (23) and (27)-(29): All the stddev stuff 
    aoa_std = np.std(aoa_l) if len(aoa_l) != 0 else 0
    img_std = np.std(img_l) if len(img_l) != 0 else 0
    fam_std = np.std(fam_l) if len(fam_l) != 0 else 0
    
    vmean_std = np.std(vmean_l) if len(vmean_l) != 0 else 0
    amean_std = np.std(amean_l) if len(amean_l) != 0 else 0
    dmean_std = np.std(dmean_l) if len(dmean_l) != 0 else 0
    
    return_vec[20] += aoa_std 
    return_vec[21] += img_std 
    return_vec[22] += fam_std 
    return_vec[26] += vmean_std 
    return_vec[27] += amean_std 
    return_vec[28] += dmean_std 
    
    # All the features depending on loop iteration 
    return_vec[1] += num_first_pnouns
    return_vec[2] += num_second_pnouns
    return_vec[3] += num_third_pnouns
    return_vec[8] += num_mult_char
    return_vec[9] += num_common_nouns
    return_vec[10] += num_proper_nouns
    return_vec[11] += num_adverbs
    return_vec[12] += num_wh_words
    return_vec[13] += num_slang   
    
    if num_nonpunct_tokens == 0:
        average_nonpunct_length = 0 # error check for divide by 0 
    else:
        average_nonpunct_length = len_nonpunct_token/num_nonpunct_tokens
    return_vec[15] += average_nonpunct_length
    return return_vec 


def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.
    AKA INDEX 29 TO 172
    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    if comment_class == "Alt":
        line_num = alt_dict.get(comment_id)
        features = alt_npy[line_num]
    elif comment_class == "Center":
        line_num = center_dict.get(comment_id)
        features = center_npy[line_num]
    elif comment_class == "Right":
        line_num = right_dict.get(comment_id)
        features = right_npy[line_num]
    elif comment_class == "Left":
        line_num = left_dict.get(comment_id)
        features = left_npy[line_num]
    
    # We need to combine features with feats. We do so by turning the 144 feature 
    # vector into length 173 by adding 29 0s at the start. Then, we add it to feats
    combined = np.add(np.concatenate((np.zeros(29), features)), feats)
    return combined


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    output = args.output
    for i in range(len(data)):
        j = data[i]
        extract1_feats = extract1(j['body'])
        extract2_feats = extract2(extract1_feats, j['cat'], j['id'])
        if j['cat'] == "Alt":
            category = np.array([3])
        elif j['cat'] == "Right":
            category = np.array([2])
        elif j['cat'] == "Left":
            category = np.array([0])
        elif j['cat'] == "Center":
            category = np.array([1])
        combined = np.concatenate((extract2_feats, category))
        feats[i] = combined 
    np.savez_compressed(output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        
    
    # Wordlists
    bristolhooly = pd.read_csv(args.p + "../Wordlists/BristolNorms+GilhoolyLogie.csv")
    warriner = pd.read_csv(args.p + "../Wordlists/Ratings_Warriner_et_al.csv")
    
    # We make dictionaries from bristolhooly and warriner 
    aoa_dict = dict(zip(bristolhooly['WORD'],bristolhooly['AoA (100-700)']))
    img_dict = dict(zip(bristolhooly['WORD'],bristolhooly['IMG']))
    fam_dict = dict(zip(bristolhooly['WORD'],bristolhooly['FAM']))
    
    vmean_dict = dict(zip(warriner['Word'],warriner['V.Mean.Sum']))
    amean_dict = dict(zip(warriner['Word'],warriner['A.Mean.Sum']))
    dmean_dict = dict(zip(warriner['Word'],warriner['D.Mean.Sum']))
    
    # We also load the NPY files for the extract2 functions below. Noting that we use the 
    # mmap_mode of 'r' to open it as read only and save memory 
    alt_npy = np.load(args.p + "/feats/Alt_feats.dat.npy", mmap_mode= 'r')
    center_npy = np.load(args.p + "/feats/Center_feats.dat.npy", mmap_mode= 'r')
    right_npy = np.load(args.p + "/feats/Right_feats.dat.npy", mmap_mode= 'r')
    left_npy = np.load(args.p + "/feats/Left_feats.dat.npy", mmap_mode= 'r')
    
    # Also need to load in the ID files, and turn them into dictionaries for fast lookup of lines
    alt_id = pd.read_csv(args.p + "/feats/Alt_IDs.txt", names=["key"])
    alt_dict = dict(zip(alt_id.key, np.arange(len(alt_id))))
    
    center_id = pd.read_csv(args.p +"/feats/Center_IDs.txt", names=["key"])
    center_dict = dict(zip(center_id.key, np.arange(len(center_id))))
    
    left_id = pd.read_csv(args.p + "/feats/Left_IDs.txt", names=["key"])
    left_dict = dict(zip(left_id.key, np.arange(len(left_id))))
    
    right_id = pd.read_csv(args.p + "/feats/Right_IDs.txt", names=["key"])
    right_dict = dict(zip(right_id.key, np.arange(len(right_id))))

    main(args)


from sklearn.preprocessing import LabelEncoder
import numpy as np
# read_conll #
# Reads input data in the format of CoNLL. Each line is a word and each column is a feature. Sentences can be read
# from up to down. Output is a list of sentences where each sentence is a list of dictionaries (words).
# --input--
# path: path to the file in CoNLL format
# --output--
# sentences: list of sentences or list of list of dictionaries (words)
def read_conll(path):
    
    sentences = []
    sentence = []

    with open(path,'r') as f:
    
        for line in f:
        
            columns = line.split('\t')
        
            if len(columns) <= 1:
                #New sentence
                sentences.append(sentence)
                sentence = []
                continue
    
            word = dict()
            word['id'] = int(columns[0])
            word['lemma'] = columns[2]
            word['pos'] = columns[4]
            word['head'] = int(columns[8])
            word['deprel'] = columns[10]
            word['fillpred'] = columns[12]
            word['pred'] = columns[13].strip()
            word['apreds'] = [ x.strip() for x in columns[14:]]
        
            sentence.append(word)
            
    return sentences

# list_pos_tags #
# Reads the types of pos tags contained in the sentences. Returns both a list of types and a sklearn LabelEncoder.
# --input--
# sentences: list of sentences where each sentence is a list of dictionaries (words)
# --output--
# pos_tags: list of strings (tags)
# pos_tag_encoder: sklearn LabelEncoder on pos_tags

def list_pos_tags(sentences):
    
    pos_tags = set()
    pos_tag_encoder = LabelEncoder()
    
    for sentence in sentences:
        for word in sentence:
            pos_tags.add(word['pos'])
    
    pos_tags = list(pos_tags)
    pos_tag_encoder.fit(pos_tags)
    
    return pos_tags,pos_tag_encoder

# list_args #
# Reads the types of args contained in the sentences. Returns both a list of types and a sklearn LabelEncoder.
# --input--
# sentences: list of sentences where each sentence is a list of dictionaries (words)
# --output--
# args: list of strings (args)
# args_encoder: sklearn LabelEncoder on args
def list_args(sentences):
    
    args = set()
    args_encoder = LabelEncoder()
    
    for sentence in sentences:
        for word in sentence:
            for arg in word['apreds']:
                args.add(arg)
    
    args = list(args)
    args_encoder.fit(args)
    
    return args,args_encoder

# load_embeddings #
# Loads the embedding from a file. File should have a word for each line followed by a list of values
# (the embedding).
# --input--
# path_embeddings: path to the file containing the embeddings
# --output--
# embeddings: np array of the embeddings
# embedding_dict: dictionary labels to indexes ( of the embeddings )
# hidden_size: size of the embeddings
def load_embeddings(path_embeddings):

    embeddings = []
    embedding_dict = dict()

    with open(path_embeddings,'r') as f:

        for ind,embedding_pair in enumerate(f):

            label, embedding = embedding_pair.split(' ',1)
            
            embedding = np.array(embedding.split(),dtype=float)
            embeddings.append(embedding)
            
            label = label.strip()
            embedding_dict[label] = ind

    hidden_size = len(embedding)

    return np.array(embeddings),embedding_dict,hidden_size

# generate_feature_vector #
# Transform a single word into a feature vector concatenating the embedding of the word and the pos tag
# --input--
# word: dictionary containg features of the word
# embedding_dict: dictionary labels to embeddings indexes
# pos_tag_encoder: a label encoder for the pos tags
# --output--
# word_vector: vector of features, embedding + pos tag
def generate_feature_vector(word,embedding_dict,pos_tag_encoder):
    
    pos = word['pos']
    lemma = word['lemma']
    
    if lemma in embedding_dict:
        embed = embedding_dict[lemma]
    else:
        embed = embedding_dict['unk']
        
    pos_value = pos_tag_encoder.transform([pos])[0]
    
    return [embed,pos_value]

# generate_inputs #
# Transform input sentences in a suitable format for the network. If a sentence contains more than one predicate
# then the sentence is repeated as many times as the number of predicates. Senteces are predicate-centered using
# the left_words and right_words values (see report). It returns an additional vector where for each sentence it 
# contains the index of the predicate taken into account. 
# --input--
# sentences: list of sentences where each sentence is a list of dictionaries (words)
# embedding_dict: dictionary labels to embeddings indexes
# pos_tag_encoder: a label encoder for the pos tags
# window_span: list of how many words to pick on the left and on the right of the predicate (respect to the index)
# --output--
# out_sentences: list of sentences where each sentence is a list of features vectors (words)
# out_pred_inds: list of indexes of predicates, one for each sentence
def generate_inputs(sentences,embedding_dict,pos_tag_encoder,window_span):
    
    out_sentences = []
    out_pred_inds = []
    
    left_words = window_span[0]
    right_words = window_span[1]

    for sentence in sentences:
        
        # Retrieve all predicates
        preds = [w for w in sentence if w['fillpred'] == 'Y']
        
        if len(preds) == 0:
            # No predicates in the sentence. Ignored
            continue
        
        # Parse the whole sentence
        parsed_sentence = []
        
        for word in sentence:
            
            word_vector = generate_feature_vector(word,embedding_dict,pos_tag_encoder)
            
            parsed_sentence.append(word_vector)
            
        # Generate predicate-centered sentences
        for pred in preds:
            
            pred_index = pred['id'] - 1
            
            left_context = parsed_sentence[max(0, pred_index - left_words): pred_index ]
            right_context = parsed_sentence[pred_index + 1: pred_index + right_words + 1 ]
            
            out_sentence = left_context + [parsed_sentence[pred_index]] + right_context

            out_sentences.append(out_sentence)
            out_pred_inds.append(len(left_context))              
        
    return out_sentences,out_pred_inds

# generate_labels #
# Generates the labels for each pair word-predicate. If a sentence has more predicates then we first generate
# the vector of labels for the first preidcate then for the second and so on. The function looks only at labels
# close to the predicates according to the window_span parameter. It returns an additional list of
# false negatives values, one for each predicate.
# --input--
# sentences: list of sentences where each sentence is a list of dictionaries (words)
# args_encoder: a label encoder for the roles
# window_span: list of how many words to pick on the left and on the right of the predicate (respect to the index)
# --output--
# out_labels: list of sentences' labels
# missed: list of how many args will not be classified.
def generate_labels(sentences,args_encoder,window_span):
    
    out_labels = []
    missed = []

    left_words = window_span[0]
    right_words = window_span[1]
    
    for sentence in sentences:

        apreds = np.array([w['apreds'] for w in sentence])        
        shape = apreds.shape
        
        if shape[1] == 0:
            # No predicates in the sentence. Ignored
            continue
          
        # Computing the labels
        apreds = np.reshape(apreds,-1)
        apreds = args_encoder.transform(apreds)
        apreds = np.reshape(apreds,shape)
        
        # Transpose, one row for each predicate
        apreds = np.transpose(apreds)
             
        preds_indexes = [w['id'] - 1 for w in sentence if w['fillpred'] == 'Y']
        
        # Windowing
        for pred_index,pred_row in zip(preds_indexes,apreds):
            
            out_pred_labels = []
            out_pred_labels += list(pred_row[max(0, pred_index-left_words): pred_index ])
            out_pred_labels += list(pred_row[pred_index : pred_index + right_words + 1 ])
            
            out_labels.append(out_pred_labels)
            
            # Missed args (outside the window)
            missed_left = len([w for w in pred_row[0: max(0, pred_index-left_words)] if w != args_encoder.transform(['_'])[0]])
            missed_right = len([w for w in pred_row[pred_index + right_words + 1:]  if w != args_encoder.transform(['_'])[0]])
            
            missed.append(missed_left+missed_right)
            
    return out_labels,missed

# generate_labels_pos #
# Generates the labels for each pair word-predicate and the labels for the pos tags. If a sentence has more
# predicates then we first generate the vector of labels for the first preidcate then for the second and so on.
# The function looks only at labels close to the predicates according to the window_span parameter.
# It returns an additional list of false negatives values, one for each predicate
# --input--
# sentences: list of sentences where each sentence is a list of dictionaries (words)
# args_encoder: a label encoder for the roles
# pos_encoder: a label encoder for the pos tags
# window_span: list of how many words to pick on the left and on the right of the predicate (respect to the index)
# --output--
# out_labels: list of sentences' labels
# missed: list of how many args will not be classified.
def generate_labels_pos(sentences,args_encoder,pos_tag_encoder,window_span):
    
    out_labels = []
    missed = []

    left_words = window_span[0]
    right_words = window_span[1]

    for sentence in sentences:

        apreds = np.array([w['apreds'] for w in sentence])        
        shape = apreds.shape
        
        if shape[1] == 0:
            # No predicates in the sentence. Ignored
            continue
        
        # Computing the pos tags labels
        pos_tags = np.array([w['pos'] for w in sentence])
        pos_tags = pos_tag_encoder.transform(pos_tags)
        
        # Computing the word-predicate labels
        apreds = np.reshape(apreds,-1)
        apreds = args_encoder.transform(apreds)
        apreds = np.reshape(apreds,shape)
        
        # Transpose, one row for each predicate
        apreds = np.transpose(apreds)
        
        preds_indexes = [w['id'] - 1 for w in sentence if w['fillpred'] == 'Y']
        
        # Windowing
        for pred_index,pred_row in zip(preds_indexes,apreds):
            
            out_pred_labels = []
            out_pred_labels += list(pred_row[max(0, pred_index-left_words): pred_index ])
            out_pred_labels += list(pred_row[pred_index : pred_index + right_words + 1 ])
            
            out_pos_labels = []
            out_pos_labels += list(pos_tags[max(0, pred_index-left_words): pred_index ])
            out_pos_labels += list(pos_tags[pred_index : pred_index + right_words + 1 ])
            
            out_labels_stack = list(np.stack([out_pred_labels,out_pos_labels],axis=1))
            
            out_labels.append(out_labels_stack)
            
            # Missed args (outside the window)
            missed_left = len([w for w in pred_row[0: max(0, pred_index-left_words)] if w != args_encoder.transform(['_'])[0]])
            missed_right = len([w for w in pred_row[pred_index + right_words + 1:]  if w != args_encoder.transform(['_'])[0]])
            
            missed.append(missed_left+missed_right)
            
    return out_labels,missed

# pad #
# Given a list of lists of elements, it pads each internal list to max_length using pad_token.
# If max_length is not specified then the lists are padded to the maximum length found in the data.
# If the pad_token is not specified then a 0 token with the right shape is generated.
# --input--
# in_lists: list of list of items
# pad_token: optional, token used to pad
# length: optional, maximum length used to pad the lists
# --output--
# padded_lists: list of lists of items (padded)
# original_lengths: list of original lengths of the internal lists
def pad(in_lists,pad_token = None,max_length = None):

    padded_lists = []
    original_lengths = []
    
    if max_length is None:
        max_length = max([len(lis) for lis in in_lists])
        
    if pad_token is None:
        pad_token_shape = np.array(in_lists[0][0]).shape
        if len(pad_token_shape) == 0:
            pad_token = 0
        else:
            pad_token = np.zeros(pad_token_shape[0])
    
    for in_list in in_lists:
        
        padded_list = in_list[:]
        
        original_lengths.append(len(in_list))

        diff = max_length - len(in_list)

        for i in range(diff):
            padded_list.append(pad_token)
            
        padded_lists.append(padded_list)

    return padded_lists,original_lengths

# compute_scores #
# Computes the precision, recall and F1 score given the predictions and the labels.
# --input--
# predictions: list of predictions, int values
# labels: list of labels, int values
# null_code: code associated to the null label (no-classification)
# missed_labels: optional list, false negatives not taken into account by the predictions.
# --output--
# precision: #roles correctly assigned/ #roles assigned
# recall: #roles correctly assigned/ total #roles
# f1_score : 2*(precision*recall)/(precision + recall)
def compute_scores(predictions,labels,null_code,missed_labels = None):
    
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    if missed_labels is not None:
        fn += sum(missed_labels)
    
    for pred,labl in zip(predictions,labels):
        
        if labl != null_code:
            
            if pred == labl:
                tp += 1
            else:
                fn += 1
        else:
            
            if pred != null_code:
                fp += 1
                
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
        
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)   
    
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall)/(precision + recall)

    return precision,recall,f1_score

# add_semantic_info_conll #
# Modifying the attribute 'lemma' appending the babelnetid. The structure of 'disambiguated_words_file' is 
# sentence_# TAB word_# TAB babelnetid. 
# --input--
# sentence: list of list of dictionaries (words)
# disambiguated_words_file: file containing the senses for each ambiguous word
# --output--
# None
def add_semantic_info_conll(sentences,disambiguated_word_file):
    
    with open(disambiguated_word_file,'r') as f:
        for line in f:
            sent_number, word_number,sense = line.split('\t')
            word = sentences[int(sent_number)][int(word_number)]
            word['lemma'] = word['lemma'] + '_' + sense.strip()


# augment_conll_data #
# The function adds the 'tag' attribute to each word in the sentences and parse the pos tag.
# In terms of tags, either 'wf' or 'instance' is chosen as value for the tag argument (compatibility with 
# the second homework). A word is said to be ambiguous if it has a match in the lemma_to_senses dictionary
# otherwise it is unambiguous. In terms of pos tags, the function uses a dictionary in order to generate
# the pos tag known by the net of the second homework (universal pos tags)
# --input--
# sentences: list of list of dicitonaries (words)
# dict_lemmas_to_senses: dictionary lemmas to possible senses
# dict_pos_mapping: dictionary penn pos tag to universal pos tag
# --output--
# None
def augment_conll_data(sentences,dict_lemmas_to_senses,dict_pos_mapping):
    
    for sentence in sentences:
        for word in sentence:
            
            lemma = word['lemma']
            
            if lemma in dict_lemmas_to_senses:
                word['tag'] = 'instance'
            else:
                word['tag'] = 'wf'
            
            pos = word['pos']
            word['pos'] = dict_pos_mapping[pos]

# add_labels #
# Given sentences, appends to each word the role-prediction in the same format as CoNLL. Predictions are inserted
# in the attribute 'papreds' (predicted apreds)
# --input--
# sentences: list of list of dictionaries ( words )
# roles: list of predictions
# window_span: list of how many words to pick on the left and on the right of the predicate (respect to the index)
# args_encoder: a label encoder for the roles
# --output--
# None 
def add_labels(sentences,roles,window_span,args_encoder):
    
    left_words = window_span[0]
    right_words = window_span[1]
    null_code = args_encoder.transform(['_'])[0]
    
    # Used to scan the roles list
    index_roles = 0
    
    for sentence in sentences:
        
        preds = [w for w in sentence if w['fillpred'] == 'Y']
        
        if len(preds) == 0:
            for word in sentence:
                word['papreds'] = []
            # No predicates in the sentence. Ignored
            continue
        
        # Will contain one list per predicate, each list will be len(sentence) long
        # Shape will be (predicates,words)
        sentence_labels = []
        
        for pred in preds:
            
            pred_index = pred['id'] - 1
            
            # How many words on the left and right of the predicate
            n_left = min(left_words,pred_index)
            n_right = min(right_words,len(sentence) - pred['id'])
            
            # Window size (window without padding)
            window_size = n_left + n_right + 1
            
            # Retrieving the roles
            pred_labels = roles[ index_roles : index_roles + window_size]
            index_roles += window_size
            
            # Indexes of the window
            left_index = pred_index - n_left
            right_index = pred_index + n_right
            
            # Computing the necessary padding
            left_pad = left_index
            right_pad = len(sentence) - 1 - right_index
            
            # Padding
            pred_labels = np.pad(pred_labels,[left_pad,right_pad],'constant',constant_values=null_code)
            
            sentence_labels.append(pred_labels)
        
        # Transpose in order to get all predicate labels for one word at time
        # Shape is (words,predicates)
        sentence_labels = np.transpose(sentence_labels)
        
        # Adding the labels
        for word,labels in zip(sentence,sentence_labels):
            word['papreds'] = list(args_encoder.inverse_transform(labels))

# write_labels_conll #
# Creates a file with the labeled sentences in the CoNLL format. Predicted labels are stored in the 'papreds'.
# --input--
# input_path: path to the input file, where are the raw sentences
# output_file: name of the file where to place the output
# sentences: list of list of dictionaries (words)
def write_labels_conll(input_path,output_file,sentences):
    
    write_file = open(output_file,'w')
    with open(input_path,'r') as f:
        
        sentence_index = 0
        word_index = 0
        
        for line in f:
            
            
            lin = line.strip()
            
            # New sentence
            if lin == '':
                
                sentence_index +=1
                word_index = 0
                write_file.write('\n')
                
                continue
                
            labels_to_write = sentences[sentence_index][word_index]['papreds']
            labels_to_write = '\t'.join(labels_to_write)
            
            lin = lin + '\t' + labels_to_write + '\n'
            
            write_file.write(lin)

            word_index += 1
            
    write_file.close()
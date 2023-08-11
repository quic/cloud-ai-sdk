from transformers import BertTokenizer
import numpy as np
import os
 
 
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
 
def save_packed_encodings(encodings, path, sl=None):
    if sl:
        print('padding')
        pad_len = sl - encodings['input_ids'].shape[1]
        for attr in ('input_ids', 'position_ids', 'token_type_ids'):
            encodings[attr] = np.pad(encodings[attr], [[0, 0], [0, pad_len]])
        encodings['attention_mask'] = np.pad(encodings['attention_mask'],
                                           [[0, 0], [0, pad_len], [0, pad_len]])
        print('padding done')
    os.makedirs(path, exist_ok=True)
    encodings['input_ids'].tofile(os.path.join(path, 'input_ids.raw'))
    encodings['attention_mask'].astype(np.bool).tofile(os.path.join(path, 'input_mask.raw'))
    encodings['token_type_ids'].tofile(os.path.join(path, 'segment_ids.raw'))
    encodings['position_ids'].tofile(os.path.join(path, 'input_position_ids.raw'))
    print('Packed encodings saved at:', path)
 
def pack(*encodings):
    new_encodings = {}
    for attr in ('input_ids', 'token_type_ids'):
        new_encodings[attr] = np.concatenate([x[attr] for x in encodings], axis=1)
    
    # create 2D attention mask
    seq_lengths = [x['input_ids'].shape[1] for x in encodings]
    mask = np.array([[i for i in range(len(seq_lengths)) \
                       for _ in range(seq_lengths[i]) ]])
    new_encodings['attention_mask'] = 1 * np.equal(mask, mask.transpose())
    new_encodings['attention_mask'] = new_encodings['attention_mask'][np.newaxis, :]
    
    # add new input (position_ids)
    new_encodings['position_ids'] = np.concatenate(
        [np.arange(x['input_ids'].shape[1], dtype=np.int64) for x in encodings])
    new_encodings['position_ids'] = new_encodings['position_ids'][np.newaxis, :]
 
    return new_encodings
 
def encode_string_packed(context, question, path='inputFiles/packed', sl=384):
    encodings = tokenizer.encode_plus(question, context, return_tensors='np')
 
    # Older versions of encode_plus return a dictionary instead of BatchEncoding
    if type(encodings) == dict:
        for attr in ('input_ids', 'attention_mask', 'token_type_ids'):
            encodings[attr] = np.array([encodings[attr]])
 
    packed_encodings = pack(encodings, encodings)
    save_packed_encodings(packed_encodings, path, sl=sl)
 
# Ques-Context pair 2
univ_context = 'The Panthers used the San Jose State practice facility and stayed at the San Jose Marriott. The Broncos practiced at Stanford University and stayed at the Santa Clara Marriott.'
univ_question = "At what university's facility did the Panthers practice?"
 
#Here are few other queries to try:
#univ_question = "At what university's facility did the Broncos practice?"
#univ_question = "Where did the Panthers stay?"
#univ_question = "Where did the Broncos stay?"
 
encode_string_packed(univ_context, univ_question)


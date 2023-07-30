# Simple CPP Example for Bert-base-cased model on AIC-100 

This project demonstrates using Bert-based-cased model from hugging face, using C++ Qaic APIs.

## To build and use it.
```bash
   mkdir build
   cd build
   cmake <path/to/root-cmake>
   make
```

Bert-base-cased model from hugging face, is based on a vocabulary file
(vocab.txt), which which needs to be downloaded from hugging-face website.

## To use the example, the user needs to :
- download the hugging face bert-base-cased model. (Refer Jupyter notebooks for NLP models). 
- Replace the QPC path used in the main.cpp with the actual QPC path.
- Replace the names of the input/output buffers as used to compile bert model into QPC
  ```
  for example:
     ("input_ids", "attention_mask") for input buffers
     ("logits")  for output  buffers
  ```
- build using above build steps
- run the executable `simple-bert-inference-example`

## The example has the following helper classes.

### VocabularyHelper :
   This class parses the vocab.txt, and stores the index of every
   string token in the vocab.txt file. The index of the words in
   this file is used in the input and output feeded to the model
   while running the inference.


### Tokenizer :
   This class, is very basic and trivial parser of input sentence
   feeded to the bert model. It uses space as delimeter to parse
   the sentence.  It does not cater special handling for special
   characters and symbols used in sentence. 
   Ideally, in C++ the user can use, for example, the 
   sentencePiece library provided as in https://github.com/google/sentencepiece 


### QBufferWrapper:
   This is a helper class to ensure that the memory allocated
   for QBuffers used in Qaic APIs is automatically  released.
   Helper functions are provided for this class
   
   `createBuffer` : create the wrapper from a QBuffer class
   
   `qBufferToString` : create a string for printing with QBuffer data


### Helper Functions to convert few data structures to string for printing:
```cpp
[[nodiscard]] std::string to_string(const qaic::rt::BufferMapping& bufMap)<br>
[[nodiscard]] std::string to_string(const qaic::rt::BufferMappings& allBufferMappings)<br>
[[nodiscard]] std::string to_string(const std::vector<int64_t> & tokenVec)<br>
```

### Processing the intput and output for inference:
   The input buffer for bert inference in this example is an array of bytes
   representing the indexes for each sentence word ( in the vocabulary file ).

   For example:
   
   If the compiled QPC has the sequence = 128 and the input type is int64_t
   then the size of input buffers must be <br>
   128 * 8 <br>
   128 [max num tokens in input] * 8 [size of each index in vocabulary file]<br>

   If the input sentence has 10 words, then the first 10*8 bytes in the
   input buffer must be populated with the indexes of the sentence words
   in the vocabulary file. Rest of the bytes must be zero initialized.

   Bert Model uses attention_mask as an input to model. The attention_mask
   input buffer can be populated with 1 for initial 10 words and rest of bytes
   can be zero initialized.

   The output buffer for bert inference in this example is an array
   of logit values (corresponding to each symbol/word in the vocabulary)
   for each input token.

   For example:
   
   If the compiled QPC has the sequence value = 128 and the output format
   is float (4 bytes). Then the QBuffer for output must be<br>
   128 * 4 * 289960 <br>
   128 [max num tokens in input] * 4 [size of each logit value] * 289960 [Vocabular Size]<br>
  
   For getting the predicted output sentence, the logit values for the
   [MASK] token must be extracted from the output buffer. Then the index for the
   maximum logit value can be used to get the predicted output word.
   
   For example:
   
   If the [MASK] token is at 3rd word index in sentence, then the corresponding
   logit values shall be present in the following bytes in the output buffer<br>
   289960*3*4  to 289960*4*4 bytes position.<br>
   These 2899960 float values are the logits for the corresponding logits for
   each symbol/word in the vocabulary.
   We find the index for maximum logit value to get the index of prediceted
   word. Then we find the word in the vocabulary.

# REFERENCE NOTES

## VARIABLES

- block_size = 8          -> how many characters the model looks at as context
- batch_size = 4          -> how many samples processed at once during training
- vocab_size = 80         -> total unique characters in the dataset
- device                  -> uses GPU if available, otherwise CPU
- text                    -> raw text content loaded from wizard_of_oz.txt
- chars                   -> sorted list of all unique characters in the text
- data                    -> the full text converted to numbers (tensor)
- train_data              -> first 80% of data, used for training
- val_data                -> last 20% of data, used for validation
- x                       -> input sequences (context)
- y                       -> target sequences (what comes next after x)
- token_embedding_table   -> shows the probability of which character comes next for each character(80x80)
- max_new_tokens          -> no. of characters to generate  

- *Logits*
- B                       -> Batch Size  
- T                       -> Time Dimension (Block Size)
- C                       -> Vocab Size

## FUNCTIONS

- stoi            -> maps each character to a number (e.g. 'a'     -> 45)
- itos            -> maps each number back to a character (e.g. 45     -> 'a')
- encode()        -> converts a string into a list of numbers
- decode()        -> converts a list of numbers back into characters
- get_batch()     -> picks random chunks from train/val data and returns input -target pairs
- forward_pass()  -> defines how input data moves through your neural network to produce an output
- generate()      -> generates the tokens (next character to come)
- estimate_loss() -> to get loss at different iterations

## CLASS

- bigramlanguagemodel()   -> neural network that predicts the next character based on the current one

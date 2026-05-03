# BPE Coding from scratch
# Byte Pair Encoding builds a vocabulary by starting with individual characters and repeatedly merging the most frequent pair of symbols. 
# That's it. HuggingFace's tokenizer does this under the hood — you're going to implement the core of it yourself.
# BPE solves the out-of-vocabulary problem. Instead of a fixed word vocabulary where "unhappiness" might be unknown, 
# BPE can represent it as "un", "happiness" or "un", "happy", "ness" — subwords it has seen before.

# architecture

# text from https://www.reedbeta.com/blog/programmers-intro-to-unicode/
text = "The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
# print(type(text)) # string right now 
text = text.split()
# print(type(text)) # converting it to list 
# main part 
temp_text = "Hello I am vraj patel and my sister is abc patel" # temp text to try out examples :)
# split every word into characters and mark end with </w>

initial_ls = []

for word in text:
    temp_list = []
    curr_word_len = len(word)
    cnt = 0
    for char in word:
        if cnt == curr_word_len - 1:
            str = char + '</w>'
            temp_list.append(str)
        else:
            temp_list.append(char)
        cnt += 1
    initial_ls.append(temp_list)

# our output glimpse
# [['H', 'e', 'l', 'l', 'o</w>'], ['I</w>'], ['a', 'm</w>'], ['v', 'r', 'a', 'j</w>'], ['p', 'a', 't', 'e', 'l</w>'], ['a', 'n', 'd</w>'], ['m', 'y</w>'], ['s', 'i', 's', 't', 'e', 'r</w>'], ['i', 's</w>'], ['a', 'b', 'c</w>'], ['p', 'a', 't', 'e', 'l</w>']]
# ab = [['H', 'e', 'l', 'l', 'o</w>'], ['I</w>'], ['a', 'm</w>'], ['v', 'r', 'a', 'j</w>'], ['p', 'a', 't', 'e', 'l</w>'], ['a', 'n', 'd</w>'], ['m', 'y</w>'], ['s', 'i', 's', 't', 'e', 'r</w>'], ['i', 's</w>'], ['a', 'b', 'c</w>'], ['p', 'a', 't', 'e', 'l</w>']]

def get_pair_counts(sequences):
    pair_counts = {}
    for sequence in sequences: 
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i+1])
            if pair in pair_counts:
                pair_counts[pair] += 1
            else:
                pair_counts[pair] = 1
    return pair_counts


pair_ls = get_pair_counts(initial_ls)


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



# print("Top pair:", top_pair[0], top_pair[1])
# print(new_ls)

# # output :
# rojects\Deep Learning Projects\Transfer Learning for nlp>python -u "c:\My Projects\Deep Learning Projects\Transfer Learning for nlp\BPE tokenizer from scratch.py"
# Top pair: i n
# [['T', 'h', 'e</w>'], ['v', 'e', 'r', 'y</w>'], ['n', 'a', 'm', 'e</w>'], ['s', 't', 'r', 'i', 'k', 'e', 's</w>'], ['f', 'e', 'a', 'r</w>'], ['a', 'n', 'd</w>'], ['a', 'w', 'e</w>'], ['in', 't', 'o</w>'], ['t', 'h', 'e</w>'], ['h', 'e', 'a', 'r', 't', 's</w>'], ['o', 'f</w>'], ['p', 'r', 'o', 'g', 'r', 'a', 'm', 'm', 'e', 'r', 's</w>'], ['w', 'o', 'r', 'l', 'd', 'w', 'i', 'd', 'e', '.</w>'], ['W', 'e</w>'], ['a', 'l', 'l</w>'], ['k', 'n', 'o', 'w</w>'], ['w', 'e</w>'], ['o', 'u', 'g', 'h', 't</w>'], ['t', 'o</w>'], ['“', 's', 'u', 'p', 'p', 'o', 'r', 't</w>'], ['U', 'n', 'i', 'c', 'o', 'd', 'e', '”</w>'], ['i', 'n</w>'], ['o', 'u', 'r</w>'], ['s', 'o', 'f', 't', 'w', 'a', 'r', 'e</w>'], ['(', 'w', 'h', 'a', 't', 'e', 'v', 'e', 'r</w>'], ['t', 'h', 'a', 't</w>'], ['m', 'e', 'a', 'n', 's', '—', 'l', 'i', 'k', 'e</w>'], ['u', 's', 'in', 'g</w>'], ['w', 'c', 'h', 'a', 'r', '_', 't</w>'], ['f', 'o', 'r</w>'], ['a', 'l', 'l</w>'], ['t', 'h', 'e</w>'], ['s', 't', 'r', 'in', 'g', 's', ',</w>'], ['r', 'i', 'g', 'h', 't', '?', ')', '.</w>'], ['B', 'u', 't</w>'], ['U', 'n', 'i', 'c', 'o', 'd', 'e</w>'], ['c', 'a', 'n</w>'], ['b', 'e</w>'], ['a', 'b', 's', 't', 'r', 'u', 's', 'e', ',</w>'], ['a', 'n', 'd</w>'], ['d', 'i', 'v', 'in', 'g</w>'], ['in', 't', 'o</w>'], ['t', 'h', 'e</w>'], ['t', 'h', 'o', 'u', 's', 'a', 'n', 'd', '-', 'p', 'a', 'g', 'e</w>'], ['U', 'n', 'i', 'c', 'o', 'd', 'e</w>'], ['S', 't', 'a', 'n', 'd', 'a', 'r', 'd</w>'], ['p', 'l', 'u', 's</w>'], ['i', 't', 's</w>'], ['d', 'o', 'z', 'e', 'n', 's</w>'], ['o', 'f</w>'], ['s', 'u', 'p', 'p', 'l', 'e', 'm', 'e', 'n', 't', 'a', 'r', 'y</w>'], ['a', 'n', 'n', 'e', 'x', 'e', 's', ',</w>'], ['r', 'e', 'p', 'o', 'r', 't', 's', ',</w>'], ['a', 'n', 'd</w>'], ['n', 'o', 't', 'e', 's</w>'], ['c', 'a', 'n</w>'], ['b', 'e</w>'], ['m', 'o', 'r', 'e</w>'], ['t', 'h', 'a', 'n</w>'], ['a</w>'], ['l', 'i', 't', 't', 'l', 'e</w>'], ['in', 't', 'i', 'm', 'i', 'd', 'a', 't', 'in', 'g', '.</w>'], ['I</w>'], ['d', 'o', 'n', '’', 't</w>'], ['b', 'l', 'a', 'm', 'e</w>'], ['p', 'r', 'o', 'g', 'r', 'a', 'm', 'm', 'e', 'r', 's</w>'], ['f', 'o', 'r</w>'], ['s', 't', 'i', 'l', 'l</w>'], ['f', 'in', 'd', 'in', 'g</w>'], ['t', 'h', 'e</w>'], ['w', 'h', 'o', 'l', 'e</w>'], ['t', 'h', 'in', 'g</w>'], ['m', 'y', 's', 't', 'e', 'r', 'i', 'o', 'u', 's', ',</w>'], ['e', 'v', 'e', 'n</w>'], ['3', '0</w>'], ['y', 'e', 'a', 'r', 's</w>'], ['a', 'f', 't', 'e', 'r</w>'], ['U', 'n', 'i', 'c', 'o', 'd', 'e', '’', 's</w>'], ['in', 'c', 'e', 'p', 't', 'i', 'o', 'n', '.</w>']]


merge_rules = []
num_merges = 50

for _ in range(num_merges):
    # 1. get pair counts from current sequences
    pair_dict = get_pair_counts(initial_ls)
    # 2. find top pair
    output = dict(reversed(sorted(pair_dict.items(), key=lambda item: item[1])))
    top_pair = list(output.keys())[0]
    # 3. merge it
    new_ls = []
    for i in initial_ls:
        new_word = []
        idx = 0
        while idx < len(i):
            if idx < len(i) - 1 and i[idx] == top_pair[0] and i[idx + 1] == top_pair[1]:
                new_word.append(top_pair[0] + top_pair[1])   
                idx += 2
            else:
                new_word.append(i[idx])
                idx += 1
        new_ls.append(new_word)
    # 4. update your sequences
    initial_ls = new_ls
    # 5. store the top pair in merge_rules
    merge_rules.append(top_pair)

print(merge_rules)


# output  
[('i', 'n'), ('t', 'h'), ('a', 'n'), ('a', 'r'), ('s', 't'), ('s', ',</w>'), ('in', 'g</w>'), ('o', 'd'), ('c', 'od'), ('i', 'cod'), ('n', 'icod'), ('U', 'nicod'), ('o', 'u'), ('o', 'r'), ('m', 'e'), ('th', 'e</w>'), ('a', 'm'), ('a', 'n</w>'), ('t', 'e'), ('l', 'l</w>'), ('t', 'o</w>'), ('an', 'd</w>'), ('r', 'i'), ('v', 'e'), ('o', 'n'), ('t', 'i'), ('l', 'e</w>'), ('e', 'p'), ('i', 't'), ('p', 'l'), ('an', 'd'), ('b', 'e</w>'), ('c', 'an</w>'), ('Unicod', 'e</w>'), ('in', 'g'), ('st', 'r'), ('o', 'r</w>'), ('f', 'or</w>'), ('u', 's'), ('w', 'h'), ('Unicod', 'e'), ('u', 'p'), ('s', 'up'), ('g', 'h'), ('n', 'o'), ('a', 'll</w>'), ('i', 'd'), ('r', 's</w>'), ('me', 'rs</w>'), ('am', 'mers</w>')]

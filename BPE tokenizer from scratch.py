# BPE Coding from scratch
# Byte Pair Encoding builds a vocabulary by starting with individual characters and repeatedly merging the most frequent pair of symbols. 
# That's it. HuggingFace's tokenizer does this under the hood — you're going to implement the core of it yourself.
# BPE solves the out-of-vocabulary problem. Instead of a fixed word vocabulary where "unhappiness" might be unknown, 
# BPE can represent it as "un", "happiness" or "un", "happy", "ness" — subwords it has seen before.

# architecture
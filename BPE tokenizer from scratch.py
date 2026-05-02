# BPE Coding from scratch
# Byte Pair Encoding builds a vocabulary by starting with individual characters and repeatedly merging the most frequent pair of symbols. 
# That's it. HuggingFace's tokenizer does this under the hood — you're going to implement the core of it yourself.
# BPE solves the out-of-vocabulary problem. Instead of a fixed word vocabulary where "unhappiness" might be unknown, 
# BPE can represent it as "un", "happiness" or "un", "happy", "ness" — subwords it has seen before.

# architecture

# text from https://www.reedbeta.com/blog/programmers-intro-to-unicode/
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
temp = text.encode("utf-8")
print(temp)
print('---')
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
print('---')
print(text)
print("length:", len(text))
print('---')
print(tokens)
print("length:", len(tokens))


# now lets write the function that wil 
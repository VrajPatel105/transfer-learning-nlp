
**Your starting point**

Say your corpus is just two words: `"aabc"` and `"abbc"`, each appearing once.

First, split every word into characters and mark the end: `['a', 'a', 'b', 'c</w>']` and `['a', 'b', 'b', 'c</w>']`

Your initial vocab is just the set of all unique characters: `{a, b, c</w>}`

---

**Step 1 — Count all adjacent pairs**

Go through every sequence and count every pair of neighbors:

From `['a', 'a', 'b', 'c</w>']`: pairs are `(a,a)`, `(a,b)`, `(b,c</w>)`

From `['a', 'b', 'b', 'c</w>']`: pairs are `(a,b)`, `(b,b)`, `(b,c</w>)`

Pair counts: `(a,b): 2`, `(a,a): 1`, `(b,c</w>): 2`, `(b,b): 1`

Two pairs are tied at 2. Pick one — say `(a,b)`.

---

**Step 2 — Merge that pair in place**

Everywhere `a` is immediately followed by `b`, collapse them into `ab`. Your sequences become:

`['a', 'ab', 'c</w>']` and `['ab', 'b', 'c</w>']`

Add `ab` to your vocab: `{a, b, c</w>, ab}`

That's it. No new dict, no substitution variable. Just updated sequences and a growing vocab.

---

**Step 3 — Repeat**

Count pairs again on the new sequences. Keep going until you hit your target vocab size.

---

**The thing to hold onto**

Your vocab starts as individual characters — that's your floor, you can always represent anything. Every merge adds one new token. After 10 merges your vocab has base chars + 10 learned tokens. After 10,000 merges you have a real tokenizer vocab.

The sequences are your working memory. The vocab is what you're building. They update together every iteration.

---

Now code just this much first — not the full tokenizer:

Write a function `get_pair_counts(sequences)` that takes a list of token lists and returns a dict of pair frequencies. Get that working and printing correctly, then we go to the merge step.
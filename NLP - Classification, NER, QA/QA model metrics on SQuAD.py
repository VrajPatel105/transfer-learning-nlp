from datasets import load_dataset
import re
import string
from collections import Counter

dataset = load_dataset("squad")
# print(dataset)
# print(dataset['train'][0])
#DatasetDict({
#     train: Dataset({
#         features: ['id', 'title', 'context', 'question', 'answers'],
#         num_rows: 87599
#     })
#     validation: Dataset({
#         features: ['id', 'title', 'context', 'question', 'answers'],
#         num_rows: 10570
#     })
# })
# {'id': '5733be284776f41900661182', 'title': 'University_of_Notre_Dame', 
# 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.', 
# 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?', 
# 'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}

def normalize_answer(s):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(s.split())

    return s

def is_exact_match(prediction, ground_truth):
    
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    
    # normalize both prediction and ground truth
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    if len(prediction) == 0 or len(ground_truth) == 0:
        return 0
    # split both into tokens by whitespace and then find overlappint tokens
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    # precision
    precision = overlap / len(pred_tokens)
    # recall
    recall = overlap / len(truth_tokens)
    if precision + recall == 0:
        return 0
    # F1 
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def best_exact_match(prediction, ground_truth):
    return max(is_exact_match(prediction, gt) for gt in ground_truth)

def best_f1(prediction, ground_truth):
    return max(f1_score(prediction, gt) for gt in ground_truth)

def evaluate(predictions, ground_truths):
    em_scores = []
    f1_scores = []
    for pred, gts in zip(predictions, ground_truths):
        em_scores.append(best_exact_match(pred, gts))
        f1_scores.append(best_f1(pred, gts))
    return {
        'exact_match': sum(em_scores) / len(em_scores),
        'f1': sum(f1_scores) / len(f1_scores)
    }

predictions = ["Saint Bernadette Soubirous", "the Virgin Mary", "wrong answer"]
ground_truths = [
    ["Saint Bernadette Soubirous"],
    ["the Virgin Mary", "Virgin Mary"],
    ["Saint Bernadette Soubirous"]
]

print(evaluate(predictions,ground_truths))
# ```python
# ============================================================
# QA with SQuAD — Evaluation Metrics from Scratch
# ============================================================
# Task: Given a passage and a question, a QA model predicts
# the start and end token positions of the answer span.
# The model outputs start logits and end logits for every token.
# Answer = tokens[argmax(start_logits) : argmax(end_logits) + 1]
#
# SQuAD has multiple valid answers per question — we compare
# prediction against all of them and take the best score.
#
# NORMALIZATION (applied before both metrics):
# - lowercase
# - remove punctuation
# - remove articles (a, an, the)
# - fix whitespace
#
# EXACT MATCH (EM):
# - normalize both prediction and ground truth
# - return 1 if they match exactly, 0 otherwise
# - harsh metric — "Saint Bernadette" vs "Saint Bernadette Soubirous" = 0
#
# F1:
# - normalize both, split into tokens
# - count overlapping tokens using Counter intersection
# - precision = overlap / len(prediction tokens)
# - recall = overlap / len(ground truth tokens)
# - F1 = 2 * precision * recall / (precision + recall)
# - lenient metric — partial credit for partial matches
# - edge cases: empty string → 0, precision+recall==0 → 0
#
# EVALUATE:
# - loops over all prediction/ground truth pairs
# - for each pair, takes best EM and best F1 across all valid answers
# - returns average EM and average F1 across the dataset
#
# When a paper says "89.7 F1 on SQuAD" — this is exactly what
# they computed. You now know what that number means at the
# implementation level.
# ============================================================
```
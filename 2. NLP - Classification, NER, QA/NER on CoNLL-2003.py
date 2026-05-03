from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np 

dataset = load_dataset("tomaarsen/conll2003")

# print(dataset)
# print(dataset['train'][0])
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'document_id', 'sentence_id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
#         num_rows: 14041
#     })
#     validation: Dataset({
#         features: ['id', 'document_id', 'sentence_id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
#         num_rows: 3250
#     })
#     test: Dataset({
#         features: ['id', 'document_id', 'sentence_id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
#         num_rows: 3453
#     })
# })
# {'id': '0', 'document_id': 1, 'sentence_id': 0, 'tokens': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], 
#  'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7], 
#  'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0], 'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]}

print(dataset['train'].features['ner_tags'])
#List(ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']))
# so, my interpretation :
# - 0 = O (outside any entity)
# - 1 = B-PER, 2 = I-PER (person)
# - 3 = B-ORG, 4 = I-ORG (organization)
# - 5 = B-LOC, 6 = I-LOC (location)
# - 7 = B-MISC, 8 = I-MISC (miscellaneous)

# B = beginning of an entity, I = inside an entity. So "New York" would be `[B-LOC, I-LOC]` — two tokens, one entity.

# From the first example: "EU" = 3 (B-ORG), "German" = 7 (B-MISC), "British" = 7 (B-MISC). Makes sense.

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples['tokens'],
        truncation='longest_first',
        padding='max_length',
        max_length=128,
        is_split_into_words=True # input is already list of words not strings
    )

    labels = []

    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100) # special tokens [CLS], [SEP], padding
            elif word_id != previous_word_id:
                aligned.append(label[word_id]) # first subword gets the real label
            else:
                aligned.append(-100) # subseqnent subwords gets ignored
            previous_word_id=word_id
        labels.append(aligned)

    tokenized['labels'] = labels
    return tokenized

dataset = dataset.map(tokenize_and_align_labels, batched=True)

# print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'document_id', 'sentence_id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
#         num_rows: 14041
#     })
#     validation: Dataset({
#         features: ['id', 'document_id', 'sentence_id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
#         num_rows: 3250
#     })
#     test: Dataset({
#         features: ['id', 'document_id', 'sentence_id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
#         num_rows: 3453
#     })
# })

model = AutoModelForTokenClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=9 # those  O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
)

seqeval = evaluate.load("seqeval")

label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label, pred in zip(labels, predictions)]
    true_preds = [[label_names[p] for p, l in zip(pred, label) if l != -100] for label, pred in zip(labels, predictions)]
    return seqeval.compute(predictions=true_preds, references=true_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics
)

trainer.train()
# output
# [transformers] `logging_dir` is deprecated and will be removed in v5.2. Please set `TENSORBOARD_LOGGING_DIR` instead.
# {'eval_loss': '0.04694', 'eval_LOC': {'precision': 0.928941908713693, 'recall': 0.974959172563963, 'f1': 0.951394422310757, 'number': 1837}, 'eval_MISC': {'precision': 0.872960372960373, 'recall': 0.8123644251626898, 'f1': 0.8415730337078652, 'number': 922}, 'eval_ORG': {'precision': 0.8682719546742209, 'recall': 0.9142431021625652, 'f1': 0.8906647293861242, 'number': 1341}, 'eval_PER': {'precision': 0.9780821917808219, 'recall': 0.9706362153344209, 'f1': 0.9743449781659389, 'number': 1839}, 'eval_overall_precision': '0.9216', 'eval_overall_recall': '0.9347', 'eval_overall_f1': '0.9281', 'eval_overall_accuracy': '0.9865', 'eval_runtime': '9.248', 'eval_samples_per_second': '351.4', 'eval_steps_per_second': '11.03', 'epoch': '1'}
# Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.11s/it]
# {'loss': '0.1084', 'grad_norm': '1.903', 'learning_rate': '3.106e-05', 'epoch': '1.139'}                                                                                                                                    
# {'eval_loss': '0.04296', 'eval_LOC': {'precision': 0.9620390455531453, 'recall': 0.9657049537289059, 'f1': 0.9638685139907633, 'number': 1837}, 'eval_MISC': {'precision': 0.8746014877789585, 'recall': 0.8926247288503254, 'f1': 0.8835212023617821, 'number': 922}, 'eval_ORG': {'precision': 0.9031781226903178, 'recall': 0.9112602535421327, 'f1': 0.9072011878247959, 'number': 1341}, 'eval_PER': {'precision': 0.9752155172413793, 'recall': 0.9842305600870038, 'f1': 0.9797023004059541, 'number': 1839}, 'eval_overall_precision': '0.9391', 'eval_overall_recall': '0.9478', 'eval_overall_f1': '0.9434', 'eval_overall_accuracy': '0.9888', 'eval_runtime': '9.709', 'eval_samples_per_second': '334.7', 'eval_steps_per_second': '10.51', 'epoch': '2'}
# Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.22it/s]
# {'loss': '0.02483', 'grad_norm': '1.151', 'learning_rate': '1.207e-05', 'epoch': '2.278'}                                                                                                                                   
# {'eval_loss': '0.04306', 'eval_LOC': {'precision': 0.9560321715817695, 'recall': 0.9706042460533478, 'f1': 0.9632631010264722, 'number': 1837}, 'eval_MISC': {'precision': 0.8747323340471093, 'recall': 0.886117136659436, 'f1': 0.8803879310344829, 'number': 922}, 'eval_ORG': {'precision': 0.9073529411764706, 'recall': 0.9202087994034303, 'f1': 0.9137356534616808, 'number': 1341}, 'eval_PER': {'precision': 0.9783432593394694, 'recall': 0.9825992387166939, 'f1': 0.9804666304937603, 'number': 1839}, 'eval_overall_precision': '0.9392', 'eval_overall_recall': '0.9498', 'eval_overall_f1': '0.9445', 'eval_overall_accuracy': '0.9893', 'eval_runtime': '9.873', 'eval_samples_per_second': '329.2', 'eval_steps_per_second': '10.33', 'epoch': '3'}
# Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.15it/s]
# {'train_runtime': '384.9', 'train_samples_per_second': '109.4', 'train_steps_per_second': '3.422', 'train_loss': '0.05372', 'epoch': '3'}                                                                                   
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1317/1317 [06:24<00:00,  3.98it/s][transformers] There were missing keys in the checkpoint model loaded: ['bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias', 'bert.encoder.layer.0.attention.output.LayerNorm.weight', 'bert.encoder.layer.0.attention.output.LayerNorm.bias', 'bert.encoder.layer.0.output.LayerNorm.weight', 'bert.encoder.layer.0.output.LayerNorm.bias', 'bert.encoder.layer.1.attention.output.LayerNorm.weight', 'bert.encoder.layer.1.attention.output.LayerNorm.bias', 'bert.encoder.layer.1.output.LayerNorm.weight', 'bert.encoder.layer.1.output.LayerNorm.bias', 'bert.encoder.layer.2.attention.output.LayerNorm.weight', 'bert.encoder.layer.2.attention.output.LayerNorm.bias', 'bert.encoder.layer.2.output.LayerNorm.weight', 'bert.encoder.layer.2.output.LayerNorm.bias', 'bert.encoder.layer.3.attention.output.LayerNorm.weight', 'bert.encoder.layer.3.attention.output.LayerNorm.bias', 'bert.encoder.layer.3.output.LayerNorm.weight', 'bert.encoder.layer.3.output.LayerNorm.bias', 'bert.encoder.layer.4.attention.output.LayerNorm.weight', 'bert.encoder.layer.4.attention.output.LayerNorm.bias', 'bert.encoder.layer.4.output.LayerNorm.weight', 'bert.encoder.layer.4.output.LayerNorm.bias', 'bert.encoder.layer.5.attention.output.LayerNorm.weight', 'bert.encoder.layer.5.attention.output.LayerNorm.bias', 'bert.encoder.layer.5.output.LayerNorm.weight', 'bert.encoder.layer.5.output.LayerNorm.bias', 'bert.encoder.layer.6.attention.output.LayerNorm.weight', 'bert.encoder.layer.6.attention.output.LayerNorm.bias', 'bert.encoder.layer.6.output.LayerNorm.weight', 'bert.encoder.layer.6.output.LayerNorm.bias', 'bert.encoder.layer.7.attention.output.LayerNorm.weight', 'bert.encoder.layer.7.attention.output.LayerNorm.bias', 'bert.encoder.layer.7.output.LayerNorm.weight', 'bert.encoder.layer.7.output.LayerNorm.bias', 'bert.encoder.layer.8.attention.output.LayerNorm.weight', 'bert.encoder.layer.8.attention.output.LayerNorm.bias', 'bert.encoder.layer.8.output.LayerNorm.weight', 'bert.encoder.layer.8.output.LayerNorm.bias', 'bert.encoder.layer.9.attention.output.LayerNorm.weight', 'bert.encoder.layer.9.attention.output.LayerNorm.bias', 'bert.encoder.layer.9.output.LayerNorm.weight', 'bert.encoder.layer.9.output.LayerNorm.bias', 'bert.encoder.layer.10.attention.output.LayerNorm.weight', 'bert.encoder.layer.10.attention.output.LayerNorm.bias', 'bert.encoder.layer.10.output.LayerNorm.weight', 'bert.encoder.layer.10.output.LayerNorm.bias', 'bert.encoder.layer.11.attention.output.LayerNorm.weight', 'bert.encoder.layer.11.attention.output.LayerNorm.bias', 'bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias'].
# [transformers] There were unexpected keys in the checkpoint model loaded: ['bert.embeddings.LayerNorm.beta', 'bert.embeddings.LayerNorm.gamma', 'bert.encoder.layer.0.attention.output.LayerNorm.beta', 'bert.encoder.layer.0.attention.output.LayerNorm.gamma', 'bert.encoder.layer.0.output.LayerNorm.beta', 'bert.encoder.layer.0.output.LayerNorm.gamma', 'bert.encoder.layer.1.attention.output.LayerNorm.beta', 'bert.encoder.layer.1.attention.output.LayerNorm.gamma', 'bert.encoder.layer.1.output.LayerNorm.beta', 'bert.encoder.layer.1.output.LayerNorm.gamma', 'bert.encoder.layer.2.attention.output.LayerNorm.beta', 'bert.encoder.layer.2.attention.output.LayerNorm.gamma', 'bert.encoder.layer.2.output.LayerNorm.beta', 'bert.encoder.layer.2.output.LayerNorm.gamma', 'bert.encoder.layer.3.attention.output.LayerNorm.beta', 'bert.encoder.layer.3.attention.output.LayerNorm.gamma', 'bert.encoder.layer.3.output.LayerNorm.beta', 'bert.encoder.layer.3.output.LayerNorm.gamma', 'bert.encoder.layer.4.attention.output.LayerNorm.beta', 'bert.encoder.layer.4.attention.output.LayerNorm.gamma', 'bert.encoder.layer.4.output.LayerNorm.beta', 'bert.encoder.layer.4.output.LayerNorm.gamma', 'bert.encoder.layer.5.attention.output.LayerNorm.beta', 'bert.encoder.layer.5.attention.output.LayerNorm.gamma', 'bert.encoder.layer.5.output.LayerNorm.beta', 'bert.encoder.layer.5.output.LayerNorm.gamma', 'bert.encoder.layer.6.attention.output.LayerNorm.beta', 'bert.encoder.layer.6.attention.output.LayerNorm.gamma', 'bert.encoder.layer.6.output.LayerNorm.beta', 'bert.encoder.layer.6.output.LayerNorm.gamma', 'bert.encoder.layer.7.attention.output.LayerNorm.beta', 'bert.encoder.layer.7.attention.output.LayerNorm.gamma', 'bert.encoder.layer.7.output.LayerNorm.beta', 'bert.encoder.layer.7.output.LayerNorm.gamma', 'bert.encoder.layer.8.attention.output.LayerNorm.beta', 'bert.encoder.layer.8.attention.output.LayerNorm.gamma', 'bert.encoder.layer.8.output.LayerNorm.beta', 'bert.encoder.layer.8.output.LayerNorm.gamma', 'bert.encoder.layer.9.attention.output.LayerNorm.beta', 'bert.encoder.layer.9.attention.output.LayerNorm.gamma', 'bert.encoder.layer.9.output.LayerNorm.beta', 'bert.encoder.layer.9.output.LayerNorm.gamma', 'bert.encoder.layer.10.attention.output.LayerNorm.beta', 'bert.encoder.layer.10.attention.output.LayerNorm.gamma', 'bert.encoder.layer.10.output.LayerNorm.beta', 'bert.encoder.layer.10.output.LayerNorm.gamma', 'bert.encoder.layer.11.attention.output.LayerNorm.beta', 'bert.encoder.layer.11.attention.output.LayerNorm.gamma', 'bert.encoder.layer.11.output.LayerNorm.beta', 'bert.encoder.layer.11.output.LayerNorm.gamma'].
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1317/1317 [06:25<00:00,  3.42it/s]
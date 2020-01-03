import spacy
import os
from tqdm import *
import argparse
import time
from seqeval.metrics import f1_score, precision_score
import sys
from spacy.tokenizer import Tokenizer

def get_pos_examples(data_dir):
    file_path = os.path.join(data_dir, "{}.txt".format("dev"))
    examples= []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f.readlines():
            if line == "\n":
                if words:
                   examples.append([words, labels])
                   words = []
                   labels = []
            elif line.startswith("#"):
                pass
            else:
                line = line.strip("\n").split("\t")
                words.append(line[1])
                labels.append(line[3])
    
    if words:
        examples.append([words, labels])
    return examples

def get_onto_pos_examples(data_dir):
    file_path = os.path.join(data_dir, "{}.txt".format("dev"))
    examples= []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f.readlines():
            if line == "\n":
                if words:
                   examples.append([words, labels])
                   words = []
                   labels = []
            elif line.startswith("#"):
                pass
            else:
                line = line.strip("\n").split("\t")
                words.append(line[0])
                labels.append(line[1])
    
    if words:
        examples.append([words, labels])
    return examples

def get_onto_ner_examples(data_dir):
    file_path = os.path.join(data_dir, "{}.txt".format("dev"))
    examples= []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f.readlines():
            if line == "\n":
                if words:
                   examples.append([words, labels])
                   words = []
                   labels = []
            elif line.startswith("#"):
                pass
            else:
                line = line.strip("\n").split("\t")
                words.append(line[0])
                if "-" in line[-1]:
                    labels.append(line[-1].split("-")[1])
                else:
                    labels.append(line[-1])
    
    if words:
        examples.append([words, labels])
    return examples
  
def get_ner_examples(data_dir, mode="dev"):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append([words, labels])
                words = []
                labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1 :
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    labels.append("O")
        if words:
            examples.append([words, labels])
            words = []
            labels = []
    return examples

def evaluate_pos(args, model):
    if args.onto:
        pos_examples = get_onto_pos_examples(args.pos_data)
    else:
        pos_examples = get_pos_examples(args.pos_data)
    print(len(pos_examples))

    model.tokenizer = Tokenizer(model.vocab)
    pred_pos_labels = []
    true_labels = []
    total_words = []
    total_tokens = []
    # inferenece
    start = time.time()
    pos_label_list = [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", 
        "DET", "INTJ", "NOUN", "NUM", "PART",
        "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
        "VERB", "X", "SPACE"
    ]

    # Just use the label of the first sub-word
    total_pred_labels = []
    for exp in tqdm(pos_examples):

        words = exp[0]
        labels = exp[1]
        
        idxs = []
        text = ""
        for word in words:
            idxs += [len(text)]
            text += word + " "
        
        tokens = model(text)
        pred_pos_labels = []
        for tk in tokens:
            # if tk.idx in idxs:
            #     pred_pos_labels.append(tk.tag_)
            pred_pos_labels.append(tk.tag_)
        # for word in words:
        #     tokens = model(word)
        #     total_words.append(word)
        #     total_tokens.append(tokens[0].text)
        #     pred_label = tokens[0].pos_
        #     if pred_label not in pos_label_list: 
        #         pred_label = "X"
        #         print(pred_label)
        #     pred_pos_labels.append(pred_label)
        
        assert len(pred_pos_labels) == len(labels)
        total_pred_labels.extend(pred_pos_labels)
       
    end = time.time()
    for exp in pos_examples:
        true_labels.extend(exp[1])

    ## evaluate
    total = len(total_pred_labels)
    hit = 0
    for pred, true_label in zip(tqdm(total_pred_labels), tqdm(true_labels)):
        if pred == true_label:
            hit += 1
    
    print("pred", total_pred_labels[:50])
    print("true labels",true_labels[:50] )
    print("sents per second", total*1.0000000/(end - start))
    print("pos tag time cost", end - start)
    print("pos acc", hit*1.0000000 / total)

    # write the prediction for checking
    # with open("pred_pos.txt", "w+", encoding="utf-8") as f:
    #     for word, tok, pred, true in zip(total_words, total_tokens, pred_pos_labels, true_labels):
    #         line = word + "\t" + tok + "\t" + pred + "\t" + true
    #         f.write(line + "\n") 




def evaluate_ner(args, model):
    if args.onto:
        ner_examples = get_onto_ner_examples(args.ner_data)
    else:
        ner_examples = get_ner_examples(args.ner_data)

    total_pred_labels = []
    true_labels = []
    total_f1 = 0.0
    start = time.time()
    total_labels_clean = []
    all_ner_words = []
    model.tokenizer = Tokenizer(model.vocab)
    for exp in tqdm(ner_examples):

        words = exp[0]
        labels = exp[1]

        idxs = []
        text = ""
        for word in words:
            idxs += [len(text)]
            text += word + " "
        
        tokens = model(text)
        pred_ner_labels = []
        for tk in tokens:
            # if tk.idx in idxs:
            #     pred_label = tk.ent_type_
            #     if len(pred_label)> 0:
            #         pred_ner_labels.append(pred_label)
            #     else:
            #         pred_ner_labels.append("O")
            pred_label = tk.ent_type_
            if len(pred_label) > 0:
                pred_ner_labels.append(pred_label)
            else:
                pred_ner_labels.append("O")

        assert len(pred_ner_labels) == len(labels)

        
       
        total_labels_clean.append(labels)
        total_pred_labels.append(pred_ner_labels)
    end = time.time()

    ## evaluate
    print("pred", total_pred_labels[:20])
    print("true labels", total_labels_clean[:20])
    total = sum([len(x) for x in total_pred_labels])
    
    print("sents per second", total*1.0000000/(end - start))
    print("ner tag time cost", end - start)
    print("ner acc", precision_score(total_labels_clean, total_pred_labels))
    print("ner f1", f1_score(total_labels_clean, total_pred_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_data", type=str, default="")
    parser.add_argument("--ner_data", type=str, default="")
    parser.add_argument("--model_type", type=str, default="")
    parser.add_argument("--onto", action="store_true")

    args = parser.parse_args()

    nlp = spacy.load(args.model_type)
    
    if len(args.pos_data) > 0:
        evaluate_pos(args, nlp)

    if len(args.ner_data) > 0:
        evaluate_ner(args, nlp)
    # ner_examples = get_ner_examples(args.ner_data)

    # print(len(ner_examples))

    

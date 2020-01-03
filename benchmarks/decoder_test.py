from mtdnn_decoder import Decoder
from tqdm import *
import time
import argparse
import os

def make_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, 1)]

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
    
def get_ner_examples(data_dir):
    file_path = os.path.join(data_dir, "{}.txt".format("dev"))
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

def speed_test_pos(args, model):
    pos_examples = get_pos_examples(args.pos_data)
    all_text = []
    total_pred_labels_num = 0
    for exp in pos_examples:
        words = exp[0]
        total_pred_labels_num += len(words)
        text = " ".join(words)
        all_text.append(text)
    
    start = time.time()
    if args.batch_size == 1:
        for exp in tqdm(pos_examples):
            words = exp[0]
            labels = exp[1]
            text = " ".join(words)
            results = decoder.do_predict(text, "pos")
    else:
        l = len(all_text)
        for ndx in tqdm(range(0, l, args.batch_size)):
            text = all_text[ndx: min(ndx + args.batch_size,l)]
            results = decoder.do_predict(text, "pos", batch_size=args.batch_size)

    end = time.time()

    print("words per second", total_pred_labels_num*1.0000000/(end-start))
    print("pos tag time cost", end - start)


def speed_test_ner(args, model):
    ner_examples = get_ner_examples(args.ner_data)
    print(len(ner_examples))
    all_text = []
    total_pred_labels_num = 0
    for exp in ner_examples:
        words = exp[0]
        total_pred_labels_num += len(words)
        text = " ".join(words)
        all_text.append(text)

    start = time.time()
    if args.batch_size == 1:
        for exp in tqdm(ner_examples):
            words = exp[0]
            labels = exp[1]
            # total_pred_labels_num += len(words)
            text = " ".join(words)
            results = decoder.do_predict(text, "ner")
    else:
        l = len(all_text)
        for ndx in tqdm(range(0, l, args.batch_size)):
            text = all_text[ndx: min(ndx + args.batch_size,l)]
            results = decoder.do_predict(text, "ner", batch_size=args.batch_size)

    end = time.time()

    print("words per second", total_pred_labels_num*1.0000000/(end-start))
    print("ner tag time cost", end - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_data", type=str, default="")
    parser.add_argument("--ner_data", type=str, default="")
    parser.add_argument("--labels_dir", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    pos_labels_fn = os.path.join(args.labels_dir, "pos.txt")
    ner_labels_fn = os.path.join(args.labels_dir, "ner.txt")
    chunking_labels_fn = os.path.join(args.labels_dir, "chunking.txt")
    srl_labels_fn = os.path.join(args.labels_dir, "srl.txt")

    config = os.path.join(args.model_dir, "config.json")
    tokenizer_name = os.path.join(args.model_dir, "vocab.txt")
    model_path = os.path.join(args.model_dir, "pytorch_model.bin")

    decoder = Decoder(pos_labels_fn = pos_labels_fn,
                    ner_labels_fn = ner_labels_fn,
                    chunking_labels_fn = chunking_labels_fn,
                    srl_labels_fn = srl_labels_fn,
                    no_cuda = args.no_cuda
                    )

    decoder.setup_model(
        config = config,
        model_name_or_path = model_path,
        tokenizer_name = tokenizer_name,
        do_lower_case=True
    )

    if len(args.pos_data) > 0:
        speed_test_pos(args, decoder)
    
    if len(args.ner_data) > 0:
        speed_test_ner(args, decoder)



    
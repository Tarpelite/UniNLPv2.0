from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch

import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import copy 
import requests
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import *

from utils_mtdnn import MegaDataSet
from uninlp import AdamW, get_linear_schedule_with_warmup
from uninlp import WEIGHTS_NAME, BertConfig, MTDNNModel, BertTokenizer
from pudb import set_trace
set_trace()

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )),
    ())

MODEL_CLASSES = {
    "bert":(BertConfig, MTDNNModel, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, model, datasets, all_dataset_sampler, task_id=-1):

    args.train_batch_size = args.mini_batch_size * max(1, args.n_gpu)

    no_decay = ["bias", "LayerNorm.weight"]
    alpha_sets = ["alpha_list"]

    if mode == "joint":
        t_total = sum(len(x) for x in datasets) //args.gradient_accumulation_steps
    else:
        t_total = len(datasets) // args.gradient_accumulation_steps

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in (no_decay + alpha_sets))], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in alpha_sets)], 'lr':1e-1}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)


    logger.info("***** Running training *****")
    logger.info(" Num Epochs = %d", args.num_train_epochs)
    logger.info(" Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)

    step = 0
    for _ in train_iterator:
        train_dataloader = DataLoader(datasets, sampler=all_dataset_sampler) 
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        model.train()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids = batch[0]
            input_mask = batch[1]
            segment_ids = batch[2]
            label_ids = batch[3]
            task_id = batch[4]
            inputs = {"input_ids":input_ids, 
                      "attention_mask":input_mask,
                      "token_type_ids":segment_ids,
                      "labels":label_ids,
                      "task_id":task_id}

            outputs = model(**inputs)
            loss = outputs[0]
            if args.do_task_embedding:
                alpha = outputs[0]
                loss = outputs[1]

            elif args.do_alpha:
                loss = outputs[1]
            
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()

            print("loss", loss)
            
            if (step + 1) % 100 == 0:
                print("loss", loss.item())
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    return model


def evaluate(args, model, UniDataSet, task):
    
    _, dataset = UniDataSet.load_single_dataset(task, "dev")
    task_id = UniDataSet.task_map[task]
    label_list = UniDataSet.labels_list[task_id]

    if torch.cuda.device_count() > 0: 
        eval_batch_size = torch.cuda.device_count() * args.mini_batch_size
    else:
        eval_batch_size = args.mini_batch_size
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    logger.info(" *** Runing {} evaluation ***".format(task)) 
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "task_id":task_id}
            
            outputs = model(**inputs)

            if args.do_alpha:
                alpha = outputs[0]
                outputs = outputs[1:]

            logits = outputs[0]

        nb_eval_steps += 1
        if preds is None:
            # print("preds", logits.shape)
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch[3].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)
    
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(label_list)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    
    results = {}
    results["a"] = accuracy_score(out_label_list, preds_list)
    results["p"] = precision_score(out_label_list, preds_list)
    results["r"] = recall_score(out_label_list, preds_list)
    results["f"] = f1_score(out_label_list, preds_list)
    logger.info("*** {} Evaluate results ***".format(task))
    for key in sorted(results.keys()):
        logger.info("  %s = %s ", key, str(results[key]))
    
    # print(results)
    print("predict_sample")
    print("predict_list", preds_list[0])
    print("out_label_list", out_label_list[0])

    return results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default=None, type=str, required=True, 
                        help="Model type selected in the list:" + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset_dir", default=None, type=str, required=True)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--mini_batch_size", default=8, type=int)

    parser.add_argument("--recover_path", default="", type=str)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument("--do_alpha", action="store_true")
    parser.add_argument("--do_adapter", action="store_true")
    parser.add_argument("--num_adapter_layers", type=int, default=2)
    parser.add_argument("--do_task_embedding", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   -1, device, args.n_gpu, False , args.fp16)

    # Set seed
    set_seed(args)

    # Setup tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=2, 
                                          cache_dir=None,
                                          output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=None
                                                )
    
    
    # Setup dataLoader Machine
    UniDataSet = MegaDataSet(datasets_dir = args.dataset_dir,
                             max_seq_length = args.max_seq_length,
                             tokenizer = tokenizer,
                             mini_batch_size = args.mini_batch_size * max(1, args.n_gpu))

    model = model_class.from_pretrained(args.model_name_or_path, 
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config = config,
                                        labels_list=UniDataSet.labels_list,
                                        do_task_embedding=args.do_task_embedding,
                                        do_alpha=args.do_alpha,
                                        do_adapter = args.do_adapter,
                                        num_adapter_layers = args.num_adapter_layers
                                        )
    
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        if os.path.exists(args.recover_path) and len(args.recover_path) > 0:
            logger.info("Recover model from %s", args.output_dir)
            checkpoint = os.path.join(args.output_dir, "pytorch_model.bin")
            model = model_class.from_pretrained(checkpoint,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config = config,
                                            labels_list=UniDataSet.labels_list,
                                            do_task_embedding=args.do_task_embedding,
                                            do_alpha=args.do_alpha,
                                            do_adapter = args.do_adapter,
                                            num_adapter_layers = args.num_adapter_layers)
            model.to(args.device)
        

        all_train_datasets, all_dataset_sampler = UniDataSet.load_MTDNN_dataset((args.mini_batch_size * max(1, args.n_gpu)), debug=args.debug)
        model = train(args, model, all_train_datasets, all_dataset_sampler=all_dataset_sampler)
    
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    
    if args.do_eval:
        
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, 
                                                    do_lower_case=args.do_lower_case)
        checkpoint = os.path.join(args.output_dir, "pytorch_model.bin")

        model = model_class.from_pretrained(checkpoint,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config = config,
                                            labels_list=UniDataSet.labels_list,
                                            do_task_embedding=args.do_task_embedding,
                                            do_alpha=args.do_alpha,
                                            do_adapter = args.do_adapter,
                                            num_adapter_layers = args.num_adapter_layers)
        

        model.to(args.device)
        total_results = {}
        for task in UniDataSet.task_list:
            # dataset = UniDataSet.load_single_dataset(task, "dev")
            # task_id = UniDataSet.task_map[task]
            # label_list = UniDataSet.labels_list[task_id]
            results = evaluate(args, model, UniDataSet, task)
            if task == "POS":
                total_results["POS_ACC"] = results["a"]
            elif task == "NER":
                total_results["NER_F1"] = results["f"]
            elif task == "CHUNKING":
                total_results["CHUNKING_F1"] = results["f"]
            elif task == "SRL":
                total_results["SRL_F1"] = results["f"]
            elif task == "ONTO_POS":
                total_results["ONTO_POS_ACC"] = results["a"]
            elif task == "ONTO_NER":
                total_results["ONTO_NER_F1"] = results["f"]
        print(total_results)

if __name__ == "__main__":
    main()



    
    
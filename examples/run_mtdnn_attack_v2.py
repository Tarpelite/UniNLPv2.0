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

from utils_mtdnn_v2 import MegaDataSet, las_score
from uninlp import AdamW, get_linear_schedule_with_warmup
from uninlp import WEIGHTS_NAME, BertConfig, MTDNNModel, BertTokenizer, DeepBiAffineDecoderV2, MTDNNModelV2

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter
# from parallel import DataParallelModel

# from pudb import set_trace
# set_trace()

logger = logging.getLogger(__name__)
# writer = SummaryWriter()

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )),
    ())

MODEL_CLASSES = {
    "bert":(BertConfig, MTDNNModelV2, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def copy_model(src_model, target_model):
    src_model_dict = src_model.state_dict()
    target_model_dict = target_model.state_dict()
    
    src_model_dict = {k:v for k,v in src_model_dict.items() if k in target_model_dict}

    target_model_dict.update(src_model_dict)
    target_model.load_state_dict(target_model_dict)

    return target_model

def setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(args.seed)

def cleanup():
    dist.destroy_process_group()


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


def train(args, model, datasets, all_dataset_sampler, task_id=-1):

    args.train_batch_size = args.mini_batch_size * max(1, args.n_gpu)
    # train_sampler = all_dataset_sampler
    # train_dataloader  = DataLoader(datasets, sampler=train_sampler)
    no_decay = ["bias", "LayerNorm.weight"]
    alpha_sets = ["alpha_list"]

    t_total = sum(len(x) for x in datasets) //args.gradient_accumulation_steps

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
        # model = DDP(model, device_ids=list(range(args.n_gpu)))

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        


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
        if all_dataset_sampler == None:
            train_sampler = RandomSampler(datasets)
            train_dataloader = DataLoader(datasets, sampler=train_sampler, batch_size=args.train_batch_size)
        else:
            train_dataloader = DataLoader(datasets, sampler=all_dataset_sampler)
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        model.train()
        iter_bar = tqdm(train_dataloader, desc="Iter(loss=X.XXX)")
        for step, batch in enumerate(iter_bar):
            input_ids = batch[0].squeeze().long().to(args.device)
            input_mask = batch[1].squeeze().long().to(args.device)
            segment_ids = batch[2].squeeze().long().to(args.device)
            label_ids = batch[3].squeeze().long().to(args.device)
            head_ids = batch[4].squeeze().long().to(args.device)
            
            task_id = batch[5].squeeze().long().to(args.device)

            assert batch[5].max() == batch[5].min()
            task_id = batch[5].max().unsqueeze(0)
            inputs = {"input_ids":input_ids, 
                      "attention_mask":input_mask,
                      "token_type_ids":segment_ids,
                      "labels":label_ids,
                      "heads":head_ids,
                      "task_id":task_id}
            
            # ======================= Code for adversarial training ====================

            if isinstance(model, torch.nn.DataParallel):
                embeds_init = model.module.bert.embeddings.word_embeddings(input_ids)
            else:
                embeds_init = model.bert.embeddings.word_embeddings(input_ids)
            
            if args.adv_init_mag > 0:
                input_mask = input_mask.to(embeds_init)
                input_lengths = torch.sum(input_mask, 1)

                if args.norm_type == "l2":
                    delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                    dims = input_lengths * embeds_init.size(-1)
                    mag = args.adv_init_mag / torch.sqrt(dims)
                    delta = (delta*mag.view(-1, 1, 1)).detach()
                elif args.norm_type == "linf":
                    delta = torch.zeros_like(embeds_init).uniform_(-args.adv_init_mag, args.adv_init_mag) * input_mask.unsqueeze(2)
            else:
                delta = torch.zeros_like(embeds_init)
            
            for astep in range(args.adv_steps):
                # (0) forward
                delta.requires_grad_()
                inputs["inputs_embeds"] = delta + embeds_init
                inputs["input_ids"] = None
                
                outputs = model(**inputs)
                loss = outputs[0]

                # (1) backward
                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gredient_accumulation_steps
                
                loss = loss / args.adv_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scale_loss:
                        scale_loss.backward()
                else:
                    loss.backward()
                

                # (2) get gradient on delta
                delta_grad = delta.grad.clone().detach()

                # (3) update and clip
                if args.norm_type == "l2":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr*delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                        exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                        reweights = (args.adv_max_norm /delta_norm * exceed_mask \
                                     + (1 - exceed_mask)).view(-1, 1 ,1)
                        
                        delta = (delta * reweights).detach()
                elif args.norm_type == "linf":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
                else:

                    print("Norm type {} not specified.".format(args.norm_type))
                    exit()
                
                if isinstance(model, torch.nn.DataParallel):
                    embeds_init = model.module.bert.embeddings.word_embeddings(input_ids)
                else:
                    embeds_init = model.bert.embeddings.word_embeddings(input_ids)
                
            # ====================== End(2)  ===============================================
                    


            
            # if args.n_gpu>1:
            #     device_ids = list(range(args.n_gpu))
            #     outputs = data_parallel(model,inputs, device_ids)
            # else:
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
            # writer.add_scalar("Loss/train", loss.item().data, global_step)
            # global_step += 1

            if args.local_rank in [-1, 0]:
                iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
         
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                scheduler.step()
                optimizer.step()

                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                iert_bar.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    return model


def evaluate(args, model, UniDataSet, task):
    
    _, dataset, _ = UniDataSet.load_single_dataset(task, batch_size=args.mini_batch_size, mode="dev")
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
            if type(model.classifier_list[task_id]) == DeepBiAffineDecoderV2: # do parsing
                logits_arc = outputs[0]
                logits_label = outputs[1]
            else:
                logits = outputs[0]

        nb_eval_steps += 1
        if preds is None:
            # print("preds", logits.shape)
            if type(model.classifier_list[task_id]) == DeepBiAffineDecoderV2:
                preds_arc = logits_arc.detach().cpu().numpy()
                preds_label = logits_label.detach().cpu().numpy()
                out_head_ids = batch[4].detach().cpu().numpy()
                out_label_ids = batch[3].detach().cpu().numpy()
            else:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch[3].detach().cpu().numpy()
        else:
            if type(model.classifier_list[task_id]) == DeepBiAffineDecoderV2:
                preds_arc = np.append(preds_arc, logits_arc.detach().cpu().numpy(), axis=0)
                preds_label = np.append(preds_label, logits_label.detach().cpu().numpy(), axis=0)

                out_head_ids = np.append(out_head_ids, batch[4].detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)
    
    if type(model.classifier_list[task_id]) == DeepBiAffineDecoderV2:
        preds_arc = np.argmax(preds_arc, axis=2)
        preds_label = np.argmax(preds_label, axis=2)
    else:
        preds = np.argmax(preds, axis=2)
    
    label_map = {i: label for i, label in enumerate(label_list)}
    print(label_map)
    if type(model.classifier_list[task_id]) == DeepBiAffineDecoderV2:
        pad_token_label_id = -100
        out_head_list = [[] for _ in range(out_head_ids.shape[0])]
        preds_arc_list = [[] for _ in range(out_head_ids.shape[0])]

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_label_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_head_ids.shape[0]):
            for j in range(out_head_ids.shape[1]):
                if out_head_ids[i, j] != pad_token_label_id:
                    out_head_list[i].append(str(out_head_ids[i][j]))
                    preds_arc_list[i].append(str(preds_arc[i][j]))
    
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_label_list[i].append(label_map[preds_label[i][j]])
    
    else:
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != -100:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
        
        if task == "ONTO_NER" or task == "NER":
            for i in range(len(preds_list)):
                for j in range(len(preds_list[i])):
                    preds_list[i][j] =  preds_list[i][j].split("-")[-1]
            
            for i in range(len(out_label_list)):
                for j in range(len(out_label_list[i])):
                    out_label_list[i][j] = out_label_list[i][j].split("-")[-1]
    
    
    
    results = {}
    if type(model.classifier_list[task_id]) == DeepBiAffineDecoderV2:
        results["uas"] = accuracy_score(out_head_list, preds_arc_list)
        results["las"] = las_score(out_label_list, out_head_list, preds_label_list, preds_arc_list)
    else:
        results["a"] = accuracy_score(out_label_list, preds_list)
        results["p"] = precision_score(out_label_list, preds_list)
        results["r"] = recall_score(out_label_list, preds_list)
        results["f"] = f1_score(out_label_list, preds_list)
        logger.info("*** {} Evaluate results ***".format(task))
    for key in sorted(results.keys()):
        logger.info("  %s = %s ", key, str(results[key]))
    
    # print(results)
    if type(model.classifier_list[task_id]) == DeepBiAffineDecoderV2:
        print("sample results")
        print("preds head", preds_arc_list[0])
        print("true head", out_head_list[0])

        print("preds label", preds_label_list[0])
        print("true label", out_label_list[0])

    else:
        print("predict_sample")
        print("predict_list", preds_list[0])
        print("out_label_list", out_label_list[0])

    # # write the results to text
    # with open("results-v2.txt", "w+", encoding="utf-8") as f:
    #     for line in preds_list:
    #         line = " ".join(line) + "\n"
    #         f.write(line)

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
    
    parser.add_argument('--adv-lr', type=float, default=0)
    parser.add_argument('--adv-steps', type=int, default=1, help="should be at least 1")
    parser.add_argument('--adv-init-mag', type=float, default=0)
    parser.add_argument('--norm-type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv-max-norm', type=float, default=0, help="set to 0 to be unlimited")

    parser.add_argument("--do_alpha", action="store_true")
    parser.add_argument("--do_adapter", action="store_true")
    parser.add_argument("--do_ft", action="store_true")
    parser.add_argument("--num_adapter_layers", type=int, default=2)
    parser.add_argument("--do_task_embedding", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   -1, device, args.n_gpu, False , args.fp16)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set seed
    set_seed(args)

    # set multi-gpu
    # setup(args, 0, 4)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

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
                                        task_list = UniDataSet.task_list,
                                        do_task_embedding=args.do_task_embedding,
                                        do_alpha=args.do_alpha,
                                        do_adapter = args.do_adapter,
                                        num_adapter_layers = args.num_adapter_layers
                                        )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        if os.path.exists(args.recover_path) and len(args.recover_path) > 0:
            if args.local_rank != -1:
                assert "Not implement in distributed environment"
            logger.info("Recover model from %s", args.output_dir)
            checkpoint = os.path.join(args.output_dir, "pytorch_model.bin")
            model = model_class.from_pretrained(checkpoint,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config = config,
                                            labels_list=UniDataSet.labels_list,
                                            task_list=UniDataSet.task_list,
                                            do_task_embedding=args.do_task_embedding,
                                            do_alpha=args.do_alpha,
                                            do_adapter = args.do_adapter,
                                            num_adapter_layers = args.num_adapter_layers)
            model.to(args.device)
        
        all_train_datasets, all_dataset_sampler = UniDataSet.load_MTDNN_dataset((args.mini_batch_size * max(1, args.n_gpu)), debug=args.debug)
        model = train(args, model, all_train_datasets, all_dataset_sampler=all_dataset_sampler)
    
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if args.local_rank in [-1, 0]:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


    if args.local_rank not in [-1, 0]:
        # evaluate using single processor
        return

    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir,
                                                    do_lower_case=args.do_lower_case)
        if args.recover_path:
            checkpoint = args.recover_path
        else:
            checkpoint = os.path.join(args.output_dir, "pytorch_model.bin")

        model = model_class.from_pretrained(checkpoint,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config = config,
                                            labels_list=UniDataSet.labels_list,
                                            task_list = UniDataSet.task_list,
                                            do_task_embedding=args.do_task_embedding,
                                            do_alpha=args.do_alpha,
                                            do_adapter = args.do_adapter,
                                            num_adapter_layers = args.num_adapter_layers)

        # model = torch.load(checkpoint)
        

        model.to(args.device)
        total_results = {}
        # source_model_dict = copy.deepcoy(model.state_dict())
        for task in UniDataSet.task_list:
            # dataset = UniDataSet.load_single_dataset(task, "dev")
            # task_id = UniDataSet.task_map[task]
            # label_list = UniDataSet.labels_list[task_id]
            if args.do_ft:
                model = model_class.from_pretrained(checkpoint,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config = config,
                                            labels_list=UniDataSet.labels_list,
                                            task_list = UniDataSet.task_list,
                                            do_task_embedding=args.do_task_embedding,
                                            do_alpha=args.do_alpha,
                                            do_adapter = args.do_adapter,
                                            num_adapter_layers = args.num_adapter_layers)
                features,dataset, task_id = UniDataSet.load_single_dataset(task, max(1, args.n_gpu)*args.mini_batch_size, mode="train")
                model.to(args.device)
                model = train(args, model, dataset, all_dataset_sampler=None, task_id=task_id)

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
            elif task == "PARSING_UD":
                total_results["PARSING_UD_UAS"] = results["uas"]
                total_results["PARSING_UD_LAS"] = results["las"]
            elif task == "PARSING_PTB":
                total_results["PARSING_PTB_UAS"] = results["uas"]
                total_results["PARSING_PTB_LAS"] = results["las"]
            if args.do_ft:
                ft_model_path = os.path.join(args.output_dir, "{}-ft.bin".format(task))
                logger.info("Saving finetuned model to {}".format(ft_model_path))
                model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                torch.save(model_to_save.state_dict(), ft_model_path)

        print(total_results)

if __name__ == "__main__":
    main()



    
    

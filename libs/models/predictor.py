# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import logging
# from math import trunc

import os
import sys
sys.path.append(".")

from typing import List

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import torch
import random
import numpy as np
from io import open
import torch.nn as nn
from libs.models.model import Seq2Seq
from libs.models.model_data import Example, InputFeatures, ModelOption
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

current_dir = os.getcwd()

step_name = "Step5"
file_name = "Generate_Patch"

logger_name = "_".join([step_name, file_name])

logger = logging.getLogger(logger_name + ".predictor")

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    idx = 0
    with open(filename) as f1:
        for line1 in f1:
            examples.append(Example(idx=idx, source=line1.strip(), target=None))
            idx+=1
    return examples

start_context_token = '<context>'
end_context_token = '</context>'
trunc_token = '<trunc/>'

def convert_examples_to_features_for_buggy_blocks(examples: List[Example], tokenizer: RobertaTokenizer, args: ModelOption, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # Source
        # 512토큰 넘어가면 special token 추가
        max_source_token_length = args.max_source_length - 2

        origin_source_tokens = tokenizer.tokenize(example.source)

        start_context_index = origin_source_tokens.index(start_context_token)

        multi_chunk_tokens = []
        context_tokens = []

        if start_context_index > -1:
            multi_chunk_tokens = origin_source_tokens[:start_context_index]
            context_tokens = origin_source_tokens[start_context_index:]

        available_length = max_source_token_length

        source_tokens = multi_chunk_tokens[:max_source_token_length]
        available_length -= len(source_tokens)
        
        multi_chunks_length = len(multi_chunk_tokens)
        contexts_length = len(context_tokens)

        with_contexts = contexts_length > 0
        if (available_length > 0) and with_contexts:
            source_tokens += context_tokens[:available_length]

        if (multi_chunks_length + contexts_length) > max_source_token_length:
            source_tokens = source_tokens[:-1] + [trunc_token]

        # source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id] * padding_length
        source_mask+=[0] * padding_length
 
        #target
        if stage=="test" or stage=="predict":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
   
        if len(source_tokens) == args.max_source_length: # example_index < 5 or
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(InputFeatures(example_index, source_ids, target_ids, source_mask, target_mask))
    return features

def convert_examples_to_features(examples: List[Example], tokenizer: RobertaTokenizer, args: ModelOption, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
 
        #target
        if stage=="test" or stage=="predict":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length 
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(InputFeatures(example_index, source_ids, target_ids, source_mask, target_mask))

    return features

def set_seed(args: ModelOption):
    """set random seed."""
    seed = args.seed
    n_gpu = args.n_gpu

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        
def predict_model(args: ModelOption):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend = 'nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    
    # Set seed
    set_seed(args)

    logger.info("arguments\n{}".format(args.__dict__))
    
    # Set config
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    # Set tokenizer
    logger.info("config : {}".format(config))

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    if args.use_special_tokens:
        special_tokens_dict = {'additional_special_tokens': ['<bug>','</bug>','<omit>','</omit>', '<context>', '</context>', '<trunc/>', '<del/>']}
        num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    # Build model
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.use_special_tokens:
        ids_start = encoder.embeddings.word_embeddings.weight.shape[0]
        encoder.resize_token_embeddings(ids_start + num_added_tokens)

        logger.info("embeddings : {}".format(ids_start + num_added_tokens))

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                  beam_size=args.beam_size, max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        checkpoint = torch.load(args.load_model_path)
        model.load_state_dict(checkpoint)
        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # Multi-gpu training
        model = torch.nn.DataParallel(model)

    # Make dir if output_dir not exist
    output_dir = args.output_dir
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    # Predict
    if args.do_predict:
        input_path = os.path.join(args.input_dir, args.input_filename)

        logger.info("Test file: {}".format(input_path))

        eval_examples = read_examples(input_path)
        eval_features = convert_examples_to_features_for_buggy_blocks(eval_examples, tokenizer, args, stage="predict") if args.use_special_tokens and args.use_buggy_contexts else convert_examples_to_features(eval_examples, tokenizer, args, stage="predict")
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
        eval_data = TensorDataset(all_source_ids, all_source_mask)   

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        predictions=[]
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(b.to(device) for b in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                outputs = model(source_ids=source_ids, source_mask=source_mask)

                logger.info("Outputs length : {}".format(len(outputs)))
                # logger.info("Outputs\n{}".format(outputs))

                for output in outputs:
                    tokens = output[0].cpu().numpy()
                    tokens = list(tokens)
                    if 0 in tokens:
                        tokens = tokens[:tokens.index(0)] # tokens except padding tokens
                    text = tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
                    predictions.append(text)

        # Generate outputs
        output_path = os.path.join(output_dir, args.output_filename)

        with open(output_path, 'w') as prediction_file:
            for prediction in predictions:
                prediction_file.write(prediction + '\n')



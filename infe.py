# -*- coding: utf-8 -*-
'''
=================================================
@Author : Senbao Shi
@Date   : 2023/7/13
@Desc   : Inference
=================================================
'''



import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from transformers import AutoModel

from model import GEMELModel

import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import params
from utils import calc_r_at_1, check_dirs, set_seed, GEMELDataset, get_prefix_allowed_fn, load_prefix_tree, get_embed

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import json

def _test(args):
    test_ds = GEMELDataset(args.data_file['test'], args.tokenizer, **args.kwargs_ds)
    test_dl = DataLoader(test_ds, batch_size=args.eval_bs, collate_fn=test_ds.collate_fn, shuffle=False)

    print(f'\nload linear {args.best_ckpt}')
    checkpoint_file = os.path.join(args.ckpt_dir, args.best_ckpt)
    checkpoint = torch.load(checkpoint_file)
    args.model.linear.load_state_dict(checkpoint)

    r_at_1 = _eval(args, test_dl)


def _eval(args, dl):
    print('Test...')
    # record
    eval_steps = len(dl)
    predictions, targets = [], []
    # evaluate
    args.model.eval()
    results_file=dict()
    results_file['predictions'] = []
    with torch.no_grad():
        with tqdm(total=eval_steps) as pbar:
            for step, batch_data in enumerate(dl):
                pbar.set_description(f'eval steps: {step}')
                pbar.update(1)

                batch_pairs, batch_targets = batch_data

                termination_tokens=[args.tokenizer.eos_token_id]
                # add the ones already in the config
                termination_tokens.extend(args.model.lm.generation_config.eos_token_id if isinstance(args.model.lm.generation_config.eos_token_id, list) else [args.model.lm.generation_config.eos_token_id])

                termination_tokens=list(set(termination_tokens))  # remove duplicates

                features = {
                    "batch_pairs": batch_pairs,
                    "num_beams": args.num_beams,
                    "num_return_sequences": 1,
                    "max_new_tokens": args.max_new_tokens,
                    #"forced_bos_token_id": args.tokenizer.bos_token_id
                    "eos_token_id": termination_tokens,
                    "pad_token_id": args.tokenizer.eos_token_id,
                    #"early_stopping": True,
                }
                if args.use_prefix_tree:
                    features['prefix_allowed_tokens_fn'] = get_prefix_allowed_fn(args.trie, args.tokenizer)
                generated = args.model.generate(**features)
                batch_preds = args.tokenizer.batch_decode(generated, skip_special_tokens=True)
                predictions.extend(batch_preds)
                targets.extend(batch_targets)
                for i in range(len(batch_preds)):
                    input_text = ''.join([t for _, t in batch_pairs[i][-4:]])
                    results_file['predictions'].append({'input_text': input_text, 'target': batch_targets[i], 'pred': batch_preds[i]})
    r_at_1 = calc_r_at_1(predictions, targets)
    results_file['metrics'] = {'r_at_1': r_at_1}
    os.makedirs(os.path.dirname(args.inf_file), exist_ok=True)
    with open(args.inf_file, 'w') as f:
        json.dump(results_file, f, indent=4)
    return r_at_1


def _main(args):
    # 1.check dir
    check_dirs(dirs=[args.dataset_dir, args.log_dir, args.ckpt_dir])

    # 2.random seed
    set_seed(args.random_seed)

    # 3.model and tokenizer
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #  load the model in half-precision to accelerate generation and optimize memory consumption on GPU
    lm = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, cache_dir=args.cache_dir)
    # freeze large language model
    print('\nFreeze LLM\n')
    for param in lm.parameters():
        param.requires_grad = False

    args.dim_embedding = lm.config.hidden_size
    kwargs_model = {'dim_clip': args.dim_clip, 'dim_embedding': args.dim_embedding,
                    'visual_prefix_length': args.visual_prefix_length, 'device': args.device}
    args.model = GEMELModel(lm=lm, tokenizer=args.tokenizer, **kwargs_model).to(args.device)

    # model for calculating similarity
    args.train_embed = get_embed(args.ment_embed_file)
    args.roberta_tokenizer = AutoTokenizer.from_pretrained(args.simcse_model)
    args.roberta_model = AutoModel.from_pretrained(args.simcse_model, cache_dir=args.cache_dir).to(args.device)

    # 4.data
    # train dataset for calculating ICL similarity
    args.ICL_ds = GEMELDataset(args.data_file['train'], tokenizer=None, img_feat=args.img_feat)

    args.kwargs_ds = {'train_ds': args.ICL_ds, 'ICL_examples_num': args.ICL_examples_num, 'img_feat': args.img_feat, 'device': args.device,
                      'train_embed': args.train_embed, 'roberta_tokenizer': args.roberta_tokenizer, 'roberta_model': args.roberta_model,
                      'max_text_tokens': args.max_text_tokens}

    # prefix tree
    args.trie = load_prefix_tree(args.trie_file, bos_token_id=args.tokenizer.bos_token_id, eos_token_id=args.tokenizer.eos_token_id) if args.use_prefix_tree else None

    # 5.inference test
    if args.do_test: _test(args)


if __name__ == '__main__':
    args = params.get_args()
    print(args)

    _main(args)

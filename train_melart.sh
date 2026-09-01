#!/bin/bash

echo $HF_HUB_CACHE
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

python main.py --dataset melart --img_feat image_features.hdf5 --trie_file prefix_tree.pkl --ment_embed_file train_mentions_embeddings.pkl --model_name llama-3-8b --max_text_tokens 256 --do_eval_steps 1188 --random_seed 42

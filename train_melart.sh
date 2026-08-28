#!/bin/bash

echo $HF_HUB_CACHE

python main.py --dataset melart --img_feat image_features.hdf5 --trie_file prefix_tree.pkl --ment_embed_file train_mentions_embeddings.pkl --model_name llama-3-8b --random_seed 42

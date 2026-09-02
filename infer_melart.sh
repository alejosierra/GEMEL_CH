#!/bin/bash

echo $HF_HUB_CACHE
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

python infe.py --dataset melart --img_feat image_features.hdf5 --trie_file prefix_tree.pkl --ment_embed_file train_mentions_embeddings.pkl --model_name llama-3-8b --max_text_tokens 256 --random_seed 42 --best_ckpt llama-3-8b_melart_linear_4token_16examples.pkl --ckpt_dir ./checkpoint/melart/42/ --inf_file ./inference_results/melart/42/results.json

python infe.py --dataset melart --img_feat image_features.hdf5 --trie_file prefix_tree.pkl --ment_embed_file train_mentions_embeddings.pkl --model_name llama-3-8b --max_text_tokens 256 --random_seed 43 --best_ckpt llama-3-8b_melart_linear_4token_16examples.pkl --ckpt_dir ./checkpoint/melart/43/ --inf_file ./inference_results/melart/43/results.json

python infe.py --dataset melart --img_feat image_features.hdf5 --trie_file prefix_tree.pkl --ment_embed_file train_mentions_embeddings.pkl --model_name llama-3-8b --max_text_tokens 256 --random_seed 44 --best_ckpt llama-3-8b_melart_linear_4token_16examples.pkl --ckpt_dir ./checkpoint/melart/44/ --inf_file ./inference_results/melart/44/results.json

python infe.py --dataset melart --img_feat image_features.hdf5 --trie_file prefix_tree.pkl --ment_embed_file train_mentions_embeddings.pkl --model_name llama-3-8b --max_text_tokens 256 --random_seed 45 --best_ckpt llama-3-8b_melart_linear_4token_16examples.pkl --ckpt_dir ./checkpoint/melart/45/ --inf_file ./inference_results/melart/45/results.json

python infe.py --dataset melart --img_feat image_features.hdf5 --trie_file prefix_tree.pkl --ment_embed_file train_mentions_embeddings.pkl --model_name llama-3-8b --max_text_tokens 256 --random_seed 46 --best_ckpt llama-3-8b_melart_linear_4token_16examples.pkl --ckpt_dir ./checkpoint/melart/46/ --inf_file ./inference_results/melart/46/results.json

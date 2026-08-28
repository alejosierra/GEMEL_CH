#!/usr/bin/env bash

python adapt_ch_datasets.py \
	--orig_datset_train_path '../../artwork-search-engine/data/processed/dataset_1_1/final_annotations/llama3/train.json' \
	--orig_datset_val_path '../../artwork-search-engine/data/processed/dataset_1_1/final_annotations/llama3/val.json' \
	--orig_datset_test_path '../../artwork-search-engine/data/processed/dataset_1_1/final_annotations/llama3/test.json' \
	--orig_datset_depicted_entities_path '../../artwork-search-engine/data/processed/dataset_1_1/final_annotations/depicted_entities.json' \
	--ent_image_dir '../../artwork-search-engine/data/raw/images' \
	--mention_train_mentions_dir 'data/wikimusa/train.json' \
	--mention_val_mentions_dir 'data/wikimusa/dev.json' \
	--mention_test_mentions_dir 'data/wikimusa/test.json' \
	--mention_image_feats_file 'data/wikimusa/image_features.hdf5' \
	--ent_train_data_dir 'data/wikimusa/depicted_entities.json' \
	--ent_prefix_tree_file 'data/wikimusa/prefix_tree.pkl' \
	--lm_model 'llama-3-8b' \
	--simcse_model 'princeton-nlp/sup-simcse-roberta-large' \
	--mention_train_mentions_embed_file 'data/wikimusa/train_mentions_embeddings.pkl'

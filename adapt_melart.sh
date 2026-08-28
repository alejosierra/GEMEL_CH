#!/bin/bash

python adapt_ch_datasets.py \
	--orig_datset_train_path '../../artwork-search-engine/data/processed/melart/final_annotations/train.json' \
	--orig_datset_val_path '../../artwork-search-engine/data/processed/melart/final_annotations/val.json' \
	--orig_datset_test_path '../../artwork-search-engine/data/processed/melart/final_annotations/test.json' \
	--orig_datset_depicted_entities_path '../../artwork-search-engine/data/processed/melart/final_annotations/depicted_entities.json' \
	--ent_image_dir '../../artwork-search-engine/data/raw/melart/images' \
	--mention_train_mentions_dir 'data/melart/train.json' \
	--mention_val_mentions_dir 'data/melart/dev.json' \
	--mention_test_mentions_dir 'data/melart/test.json' \
	--mention_image_feats_file 'data/melart/image_features.hdf5' \
	--ent_train_data_dir 'data/melart/depicted_entities.json' \
	--ent_prefix_tree_file 'data/melart/prefix_tree.pkl' \
	--lm_model 'llama-3-8b' \
	--simcse_model 'princeton-nlp/sup-simcse-roberta-large' \
	--mention_train_mentions_embed_file 'data/melart/train_mentions_embeddings.pkl'

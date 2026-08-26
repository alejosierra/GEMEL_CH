
import argparse as ArgumentParser
import json
from tqdm import tqdm
from pathlib import Path

if __name__=='__main__':

    argparse = ArgumentParser.ArgumentParser(description='Adapt CH Datasets')
    argparse.add_argument('--orig_datset_train_path', type=str, default='../../artwork-search-engine/data/processed/dataset_1_1/final_annotations/llama3/train.json', help='Path to the original training dataset JSON file')
    argparse.add_argument('--orig_datset_val_path', type=str, default='../../artwork-search-engine/data/processed/dataset_1_1/final_annotations/llama3/val.json', help='Path to the original validation dataset JSON file')
    argparse.add_argument('--orig_datset_test_path', type=str, default='../../artwork-search-engine/data/processed/dataset_1_1/final_annotations/llama3/test.json', help='Path to the original test dataset JSON file')
    argparse.add_argument('--orig_datset_depicted_entities_path', type=str, default='../../artwork-search-engine/data/processed/dataset_1_1/final_annotations/depicted_entities.json', help='Path to the original depicted entities JSON file')
    argparse.add_argument('--ent_image_dir', type=str, default="../../artwork-search-engine/data/raw/images", help='Path to images directory')
    argparse.add_argument('--mention_train_mentions_dir', type=str, default="data/wikimusa/train.json", help='Path to save the adapted training mentions JSON file')
    argparse.add_argument('--mention_val_mentions_dir', type=str, default="data/wikimusa/dev.json", help='Path to save the adapted validation mentions JSON file')
    argparse.add_argument('--mention_test_mentions_dir', type=str, default="data/wikimusa/test.json", help='Path to save the adapted test mentions JSON file')
    argparse.add_argument('--ent_train_data_dir', type=str, default="data/wikimusa/depicted_entities.json", help='Path to save the adapted depicted entities JSON file')
    argparse.add_argument('--ent_prefix_tree_file', type=str, default="data/wikimusa/prefix_tree.pkl", help='Path to save the adapted depicted entities Prefix tree')
    argparse.add_argument('--ent_prefix_tree_id_first', action='store_true', help='Whether to use the entity ID as the first element in the prefix tree')

    args = argparse.parse_args()

    # Prepare mentions
    input_mentions_train = args.orig_datset_train_path
    input_mentions_val = args.orig_datset_val_path
    input_mentions_test = args.orig_datset_test_path

    depicted_entities_dir=args.orig_datset_depicted_entities_path
    ent_output_dir=args.ent_train_data_dir
    Path(ent_output_dir).parent.mkdir(parents=True, exist_ok=True)

    all_entities = {}
    for qid, depicted_entity in tqdm(json.load(open(depicted_entities_dir, 'r', encoding='utf-8')).items(), desc="Processing depicted entities"):
        """
            "Q211568": {
            "qid": "Q211568",
            "label": "fleur-de-lis",
            "description": "stylized iris flower used as a heraldic symbol",
            "types": {
                "Q3744866": "mobile charge"
            },
            "images": [
                "Fleur%20de%20lys%20%28or%29.svg"
            ],
            "full_text": "fleur-de-lis. stylized iris flower used as a heraldic symbol. mobile charge"
        },

        to 

        {
        "Q1": "Q211568. fleur-de-lis",
        "Q2": "Q2. Paris is the capital city of France."
        }
        """
        
        new_depicted_entity = {
            qid: f"{qid}. {depicted_entity.get('label', '')}" if args.ent_prefix_tree_id_first else f"{depicted_entity.get('label', '')}. {qid}"
        }
        all_entities.update(new_depicted_entity)
    with open(Path(ent_output_dir), 'w', encoding='utf-8') as f:
        json.dump(all_entities, f, ensure_ascii=False, indent=4)

    for split, input_mentions in zip(['train', 'val', 'test'], [input_mentions_train, input_mentions_val, input_mentions_test]):
        mentions_output_dir = getattr(args, f'mention_{split}_mentions_dir')
        Path(mentions_output_dir).parent.mkdir(parents=True, exist_ok=True)

        with open(input_mentions, 'r', encoding='utf-8') as f:
            orig_mentions = json.load(f)
        new_mentions = []
        for id, mention in tqdm(orig_mentions.items(), desc=f"Processing {split} mentions"):
            """
            "Q18891111": {
                "depicted_entities": [
                    "Q7307"
                ],
                "iconographic_depictions": [],
                "preiconographic_depictions": [
                    "Q7307"
                ],
                "article": "The artwork \"The Kiss\" represents a couple embracing in darkness, their faces merged into a single, featureless shape. The oil painting on canvas, created by Norwegian symbolist artist Edvard Munch in 1897, is characterized by long, slurpy brush strokes and depicts the unity of the lovers amidst a somber atmosphere.",
                "img": [
                    "Edvard%20Munch%20-%20The%20Kiss%20-%20Google%20Art%20Project.jpg"
                ]
            }

            to 

            {"text": "[START_ENT] Otto von Corvin [END_ENT]", "img_name": "20168.jpg", "mention": "Otto von Corvin", "golden": "Q75265", "target": "Otto von Corvin"}

            {
                "id": "Q18891111",
                "text": "The artwork \"The Kiss\" represents a couple embracing in darkness, their faces merged into a single, featureless shape. The oil painting on canvas, created by Norwegian symbolist artist Edvard Munch in 1897, is characterized by long, slurpy brush strokes and depicts the unity of the lovers amidst a somber atmosphere.",
                "golden": ["Q7307"],
                "img_name": "Edvard%20Munch%20-%20The%20Kiss%20-%20Google%20Art%20Project.jpg"
            }
            """

            golden_entities = [all_entities.get(qid, qid) for qid in mention.get("depicted_entities", [])]
            
            new_mention = {
                "id": id,
                "text": mention.get("article", ""),
                "golden": golden_entities,
                "img_name": mention.get("img", [""])[0] if mention.get("img") else ""
            }
            new_mentions.append(new_mention)

        with open(mentions_output_dir, 'w', encoding='utf-8') as f:
            for mention in new_mentions:
                f.write(json.dumps(mention, ensure_ascii=False) + "\n")

    
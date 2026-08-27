
import argparse as ArgumentParser
import json
import h5py
import torch
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

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
    argparse.add_argument('--mention_image_feats_file', type=str, default="data/wikimusa/image_features.hdf5", help='Path to save the adapted image features HDF5 file')
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

    ent_image_dir=Path(args.ent_image_dir)
    file_mappings={}
    assert ent_image_dir.exists() and ent_image_dir.is_dir()
    if (ent_image_dir / "file_name_mapping.json").exists():
        with open(ent_image_dir / "file_name_mapping.json", 'r', encoding='utf-8') as f:
            file_mappings = json.load(f)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda" if torch.cuda.is_available() else "cpu")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # change config to include hidden states for clip_model
    clip_model.config.vision_config.output_hidden_states = True

    with torch.no_grad(), h5py.File(args.mention_image_feats_file, 'w') as hdf5_file:

        all_images = dict() #. img_name to path

        for split, input_mentions in tqdm(zip(['train', 'val', 'test'], [input_mentions_train, input_mentions_val, input_mentions_test])):
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

                img_name = mention.get("img", [""])[0] if mention.get("img") else ""
                img_path = ent_image_dir / file_mappings.get(img_name, img_name)

                if not img_path.exists():
                    print(f"Image not found: {img_path}")
                    break

                all_images[img_name] = img_path

                new_mention = {
                    "id": id,
                    "text": mention.get("article", ""),
                    "golden": golden_entities,
                    "img_name": img_name
                }
                new_mentions.append(new_mention)

            with open(mentions_output_dir, 'w', encoding='utf-8') as f:
                for mention in new_mentions:
                    f.write(json.dumps(mention, ensure_ascii=False) + "\n")

        # Process images and save features to HDF5
        """

                # encode the image using clip for adding to the hdf5 file
                image = Image.open(img_path)
                image_input = clip_processor(images=image, return_tensors="pt")
                image_features = clip_model.vision_model(**image_input).last_hidden_state

                # get the CLS token representation
                cls_token_representation = image_features[:, 0, :].squeeze()

                # save the image features to the HDF5 file
                hdf5_file.create_dataset(img_name, data=cls_token_representation.detach().numpy())
        """

        #process them in batches
        batch_size = 32
        img_names = list(all_images.keys())
        for i in tqdm(range(0, len(img_names), batch_size), desc="Processing images in batches"):
            batch_img_names = img_names[i:i + batch_size]
            batch_images = [Image.open(all_images[img_name]) for img_name in batch_img_names]
            image_inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
            image_features = clip_model.vision_model(**image_inputs).last_hidden_state

            # get the CLS token representation for each image in the batch
            cls_token_representations = image_features[:, 0, :] # (batch_size, hidden_size)

            # save the image features to the HDF5 file
            for j, img_name in enumerate(batch_img_names):
                hdf5_file.create_dataset(img_name, data=cls_token_representations[j].to("cpu").detach().numpy())


    
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import json
import argparse
from PIL import Image
from disc_loss import DiscriminativeLoss
from embed import Scorer


os.makedirs('./captions', exist_ok=True)
os.makedirs('./uid2acc', exist_ok=True)

def main(caps_filename):

    dataset = json.load(open('/home/manugaur/detect-describe-discriminate/dataset.json', 'r'))
    
    loss = DiscriminativeLoss() # SR objective
    scorer = Scorer() #SigLIP 
    
    #Load predicted caps
    caps_path = f"./captions/{caps_filename}.json"
    caps = json.load(open(caps_path, "r"))

    print(f"| Self-Retrieval Evaluation on D3 Benchmark")

    rand_50_scores = []
    rand_25_scores = []

    uid2recall_1 = {}

    for uid, images in tqdm(dataset.items(), total = len(dataset)):

        img_feats = torch.cat([scorer.get_feat(image=Image.open(images['left']).convert("RGB")), scorer.get_feat(image=Image.open(images['right']).convert("RGB"))])
        text_feats = torch.cat([scorer.get_feat(text=caps[uid]['left']), scorer.get_feat(text=caps[uid]['right'])])
        
        rand_50, rand_25, recall_1 = loss(text_feats, img_feats)
        rand_50_scores.append(rand_50)
        rand_25_scores.append(rand_25)
        uid2recall_1[uid] = [_.item() for _ in recall_1]

    with open(f"./uid2acc/{caps_path.split('/')[-1].split('.')[0]}.pkl", "wb") as f:
        pickle.dump(uid2recall_1, f)

    print(f"| Scores : {round(np.array(rand_25_scores).mean()*100,2)}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass the name of the json file in ./captions as argument for SR eval")
    parser.add_argument('caps_filename', type=str, help="caption json filename")
    args = parser.parse_args()
    main(args.caps_filename)

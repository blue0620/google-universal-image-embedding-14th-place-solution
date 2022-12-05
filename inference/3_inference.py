import sys
import os
import math
import random
import re
import warnings
import imghdr
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import normalize
import json
IMAGE_SIZE = 336
BATCH_SIZE = 128
SEED = 1213

strategy = tf.distribute.get_strategy()
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)



def create_model_for_inference(weights_path: str):
    # Model loading.
    model = tf.saved_model.load(weights_path)
    print(model.summary())
    embedding_fn = model.signatures["serving_default"]
    return embedding_fn



def load_image_tensor(image_path):
    tensor = tf.convert_to_tensor(np.array(Image.open(image_path).convert("RGB").resize((IMAGE_SIZE,IMAGE_SIZE))))
    #tensor = tf.image.resize(tensor, size=(IMAGE_SIZE,IMAGE_SIZE))
    expanded_tensor = tf.expand_dims(tensor, axis=0)
    return expanded_tensor


def create_batch(files):
    images = []
    for f in files:
        images.append(load_image_tensor(f))
    return tf.concat(images, axis=0)


def extract_global_features(image_root_dir):
    image_paths = []
    for root, dirs, files in os.walk(image_root_dir):
        for file in files:
            if os.path.isfile(os.path.join(root, file)) and imghdr.what(os.path.join(root, file))!=None:
                image_paths.append(os.path.join(root, file))
    num_embeddings = len(image_paths)

    ids = []
    for path in image_paths:
        ids.append(path.split('/')[-1][:-4])
    
    emb = np.zeros((num_embeddings, 64))
    image_paths = np.array(image_paths)
    
    #chunk_size = 512
    chunk_size = BATCH_SIZE
    
    n_chunks = len(image_paths) // chunk_size
    if len(image_paths) % chunk_size != 0:
        n_chunks += 1
    #model = create_model_for_inference("finalmodel")
    with strategy.scope():
        model = tf.saved_model.load("embedding_norm_model")
        embedding_fn = model.signatures["serving_default"]
        for i in tqdm(range(n_chunks)):
            files = image_paths[i * chunk_size:(i + 1) * chunk_size]
            batch = create_batch(files)
            embedding_tensor = embedding_fn(batch)["embedding_norm"]
            embedding_tensor = normalize(embedding_tensor,axis=1)
            emb[i * chunk_size:(i + 1) * chunk_size] += embedding_tensor
        np.savez('savedresult', image_paths=image_paths, emb=emb)
        with open("output.jsonl","w") as wf:
            for eachemb,imgpath in zip(emb,image_paths):
                imgname=os.path.basename(imgpath)
                jsonobj={}
                jsonobj["feature"]=",".join([str(num) for num in eachemb.tolist()])
                jsontext = json.dumps(jsonobj)
                wf.write(imgname+"\t"+jsontext+"\n")


extract_global_features("/data/img-data/")

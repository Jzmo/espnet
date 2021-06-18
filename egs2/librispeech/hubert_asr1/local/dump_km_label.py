# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np

import joblib
import torch
import tqdm
import pdb

from sklearn_km import (MfccFeatureReader, get_path_iterator)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def dump_pseudo_label_mfcc(km_path, task, sample_rate, nj):
    apply_kmeans = ApplyKmeans(km_path)
    reader = MfccFeatureReader(sample_rate)
    generator, num = get_path_iterator(f"{task}/wav.scp")
    iterator = generator()
    
    if nj > 1:
        feats = joblib.Parallel(n_jobs=nj)(
            joblib.delayed(reader.get_feats)(path)
                           for utt_id, path in
                           tqdm.tqdm(iterator, total=num))

        p_labs = joblib.Parallel(n_jobs=nj)(
            joblib.delayed(apply_kmeans)(feat)
            for feat in
            tqdm.tqdm(feats, total=num))
        iterator = generator()
        utt_ids = [utt_id for utt_id, _ in iterator]
    else:
        utt_ids, p_labs = [], []
        for utt_id, path in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path)
            p_lab = apply_kmeans(feat).tolist()
            p_labs.append(p_lab)
            utt_ids.append(utt_id)
    return utt_ids, p_labs

def dump_label(km_path, recog_set, sample_rate, nj):
    if recog_set:
        for task in recog_set:
            logger.info("Dumping pseudo labeling for: %s", task)
            utt_ids, p_labs = dump_pseudo_label_mfcc(f"{km_path}/km.gz", task, sample_rate, nj)
            with open(f"{task}/ptext", "w") as f:
                for utt_id, p_lab in zip(utt_ids, p_labs):
                    f.write(utt_id + " " + " ".join(map(str, p_lab)) + "\n")
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--km-path", type=str)
    parser.add_argument("--recog-set", default=None,
                        nargs='+', help='folders contain wav.scp for recog')
    parser.add_argument("--nj", default=1, type=int)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    logging.info(str(args))

    dump_label(**vars(args))

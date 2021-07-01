import logging
import os
import sys
from random import sample
import warnings
    
import numpy as np
import math

import soundfile as sf
import torch
import torchaudio
import tqdm

from sklearn.cluster import MiniBatchKMeans
from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder
import fairseq

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")

class MfccFeatureReader(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def read_audio(self, path):
        ref_len = sf.info(path).frames
        wav, sr = sf.read(path)
        assert sr == self.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path):
        x = self.read_audio(path)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.view(1, -1)

            mfccs = torchaudio.compliance.kaldi.mfcc(
                waveform=x,
                sample_frequency=self.sample_rate,
                use_energy=False,
            )  # (time, freq)
            mfccs = mfccs.transpose(0, 1)  # (freq, time)
            deltas = torchaudio.functional.compute_deltas(mfccs)
            ddeltas = torchaudio.functional.compute_deltas(deltas)
            concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
            concat = concat.transpose(0, 1).contiguous()  # (freq, time)
            return concat


def get_path_iterator(wav, portion=0.1):
    with open(wav, "r") as f:
        #root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        lines = sample(lines,int(portion*len(lines)))
        def iterate():
            for line in lines:
                utt_id, path = line.split(" ")
                yield utt_id, f"{path}"        

        return iterate, len(lines)

class HubertFeatureReader(object):
    def __init__(self, hubert_url, hubert_dir_path, layer, max_chunk=1600000):
        print(hubert_url, hubert_dir_path)
        e = FairseqHubertEncoder(0, hubert_url, hubert_dir_path).cuda()
        self.model = e.encoders.eval()
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            #x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
            return torch.cat(feat, 1).squeeze(0)
    
    
def gen_mfcc_feature(root, sample_rate, nj, portion):
    reader = MfccFeatureReader(sample_rate)
    generator, num = get_path_iterator(f"{root}/wav.scp", portion)
    iterator = generator()

    if nj > 1:
        feats = joblib.Parallel(n_jobs=nj)(
            joblib.delayed(reader.get_feats)(path)
                           for utt_id, path in
                           tqdm.tqdm(iterator, total=num))
    else:
        feats = []
        for utt_id, path in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path)
            feats.append(feat.cpu().numpy())           
        np.random.shuffle(feat)
    logger.info("finished successfully")
    return np.vstack(feats)

def gen_hubert_feature(root, sample_rate, portion, url, dir, layer):

    reader = HubertFeatureReader(url, dir, layer)
    generator, num = get_path_iterator(f"{root}/wav.scp", portion)
    iterator = generator()
    feats = []
    for utt_id, path in tqdm.tqdm(iterator, total=num):
        feat = reader.get_feats(path)
        feats.append(feat.cpu().numpy())           
    np.random.shuffle(feat)
    logger.info("finished successfully")
    return np.vstack(feats)

def load_feature(root, sample_rate, nj, portion, feature, hurl, hdir):
    # generate mfcc feature
    if feature == "mfcc":
        feat = gen_mfcc_feature(root, sample_rate, nj, portion)
    elif "hubert" in feature:
        hlayer = int(feature.replace("hubert", ""))
        feat = gen_hubert_feature(root, sample_rate, portion, hurl, hdir, hlayer)
    # TODO, extract hubert feature
    return feat

def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def learn_kmeans(
    root,
    km_path,
    n_clusters,
    nj,
    seed,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
    sample_rate,
    portion,
    feature,
    hurl,
    hdir,
):
    np.random.seed(seed)
    feat=load_feature(
        root,
        sample_rate,
        nj,
        portion,
        feature,
        hurl,
        hdir,
    )
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, f"{km_path}/km.gz")

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished k-means training successfully")
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="folder contains wav.scp for training")
    parser.add_argument("--km-path", type=str)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--nj", default=1, type=int, help="only support mfcc")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max-iter", default=100, type=int)
    parser.add_argument("--batch-size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max-no-improvement", default=100, type=int)
    parser.add_argument("--n-init", default=20, type=int)
    parser.add_argument("--reassignment-ratio", default=0.0, type=float)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--portion", type=float, default=1.0)
    parser.add_argument("--feature", type=str, default="mfcc")
    parser.add_argument("--hurl", type=str, default="./")
    parser.add_argument("--hdir", type=str, default="./")
    
    args = parser.parse_args()
    logging.info(str(args))
    learn_kmeans(**vars(args))
    #    with warnings.filterwarnings("ignore"):

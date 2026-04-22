#!/usr/bin/env python3
"""WavJEPA 임베딩 벡터 기반 kNN 성능 측정 스크립트.

HuggingFace 리모트 코드를 런타임에 참조하지 않고,
로컬 safetensors + 내부 구현 클래스(local_wavjepa.py)로 동작합니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from local_wavjepa import load_local_wavjepa


# -------- I/O --------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="로컬 모델 디렉토리(config.json + *.safetensors)")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_label(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("label", "labels", "class", "target"):
        if key in data:
            v = data[key]
            return v[0] if isinstance(v, list) else v
    for v in data.values():
        if isinstance(v, (str, int)):
            return v
        if isinstance(v, list) and v and isinstance(v[0], (str, int)):
            return v[0]
    raise ValueError(f"라벨 필드를 찾을 수 없습니다: {json_path}")


def load_split(split_dir: Path):
    wavs = sorted(split_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"wav 파일이 없습니다: {split_dir}")

    pairs = []
    for w in wavs:
        j = w.with_suffix(".json")
        if not j.exists():
            raise FileNotFoundError(f"json 라벨 파일이 없습니다: {j}")
        pairs.append((w, read_label(j)))
    return pairs


# -------- Model --------
def get_embedding(model, extractor, wav_path: Path, device: str) -> np.ndarray:
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0).numpy()  # mono, (T,)

    inputs = extractor(
        wav,
        sampling_rate=sr,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        out = model(input_values)

    feat = out[0] if isinstance(out, tuple) else out
    if feat.ndim == 3:
        feat = feat.mean(dim=1)

    return feat.squeeze(0).detach().cpu().numpy()


def extract_split_embeddings(model, extractor, items, device: str):
    xs, ys = [], []
    for wav_path, label in tqdm(items, desc="embedding"):
        xs.append(get_embedding(model, extractor, wav_path, device))
        ys.append(label)
    return np.stack(xs), np.array(ys)


def main() -> None:
    args = parse_args()

    train_items = load_split(args.dataset_root / "train")
    test_items = load_split(args.dataset_root / "test")

    if not args.model.exists():
        raise FileNotFoundError(f"모델 디렉토리가 없습니다: {args.model}")

    model, extractor, load_info = load_local_wavjepa(args.model, args.device)
    print(f"model load info: missing={load_info['missing']}, unexpected={load_info['unexpected']}")

    x_train, y_train = extract_split_embeddings(model, extractor, train_items, args.device)
    x_test, y_test = extract_split_embeddings(model, extractor, test_items, args.device)

    clf = KNeighborsClassifier(n_neighbors=args.k, metric="cosine")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    print(f"test/acc: {accuracy_score(y_test, pred):.4f}")
    print(f"test/f1_macro: {f1_score(y_test, pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()

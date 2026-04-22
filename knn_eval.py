#!/usr/bin/env python3
"""WavJEPA 임베딩 벡터 기반 kNN 성능 측정 스크립트.

HuggingFace 인터페이스(AutoModel + AutoFeatureExtractor) 기준으로 동작합니다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


# -------- I/O --------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="로컬 HF 모델 디렉토리(코드+config+safetensors)")
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
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    if isinstance(out, tuple):
        feat = out[0]
    else:
        feat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

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

    # 로컬에 복사된 코드/가중치만 사용 (원격 HF 참조 금지)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    ).to(args.device).eval()
    extractor = AutoFeatureExtractor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )

    x_train, y_train = extract_split_embeddings(model, extractor, train_items, args.device)
    x_test, y_test = extract_split_embeddings(model, extractor, test_items, args.device)

    clf = KNeighborsClassifier(n_neighbors=args.k, metric="cosine")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    print(f"test/acc: {accuracy_score(y_test, pred):.4f}")
    print(f"test/f1_macro: {f1_score(y_test, pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""WavJEPA 임베딩 기반 kNN 평가 스크립트.

기본 경로는 HuggingFace가 아니라 WavJEPA GitHub 코드의 HEAR runtime 로딩 방식
(hear_configs/WavJEPA.py -> RuntimeJEPA)을 따라 로컬 ckpt를 읽습니다.

데이터셋 구조:
- dataset_root/train/*.wav + 같은 이름의 .json 라벨
- dataset_root/test/*.wav + 같은 이름의 .json 라벨
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


# -------- I/O --------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 기본: WavJEPA GitHub 로컬 코드 + 로컬 ckpt
    p.add_argument(
        "--backend",
        type=str,
        default="wavjepa",
        choices=["wavjepa", "hf"],
        help="임베딩 추출 백엔드. 기본은 wavjepa(로컬 ckpt).",
    )
    p.add_argument("--wavjepa_repo", type=Path, help="로컬 wavjepa GitHub 리포 경로")
    p.add_argument("--ckpt_path", type=Path, help="로컬 WavJEPA .ckpt 파일 경로")

    # 선택: 기존 HF 경로(호환 목적)
    p.add_argument("--model", type=str, help="HF repo id 또는 로컬 HF 모델 디렉토리 (backend=hf)")
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
def load_wavjepa_runtime(wavjepa_repo: Path, ckpt_path: Path, device: str):
    """WavJEPA GitHub HEAR 설정 코드를 통해 로컬 ckpt 로드.

    참조: hear_configs/WavJEPA.py 의 load_model(model_path)
    """
    if not wavjepa_repo:
        raise ValueError("backend=wavjepa 인 경우 --wavjepa_repo 가 필요합니다.")
    if not ckpt_path:
        raise ValueError("backend=wavjepa 인 경우 --ckpt_path 가 필요합니다.")
    if not wavjepa_repo.exists():
        raise FileNotFoundError(f"wavjepa repo 경로가 없습니다: {wavjepa_repo}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt 파일이 없습니다: {ckpt_path}")

    repo_str = str(wavjepa_repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    wavjepa_cfg = importlib.import_module("hear_configs.WavJEPA")
    model = wavjepa_cfg.load_model(str(ckpt_path))

    model_device = torch.device(device)
    model = model.to(model_device)
    model.eval()

    # RuntimeJEPA의 FeatureExtractor.forward가 .cuda()를 강제하는 경우를 우회.
    if hasattr(model, "feature_extractor") and hasattr(model.feature_extractor, "_wav2feature"):
        def _safe_feature_forward(x: torch.Tensor, _m=model):
            feat = _m.feature_extractor._wav2feature(x)
            return feat.to(next(_m.model.parameters()).device)

        model.feature_extractor.forward = _safe_feature_forward

    return model


def load_hf_model(model_id_or_dir: str, device: str):
    from transformers import AutoFeatureExtractor, AutoModel

    model = AutoModel.from_pretrained(model_id_or_dir, trust_remote_code=True).to(device).eval()
    extractor = AutoFeatureExtractor.from_pretrained(model_id_or_dir, trust_remote_code=True)
    return model, extractor


def get_embedding(backend_obj: Any, backend: str, wav_path: Path, device: str) -> np.ndarray:
    wav, sr = torchaudio.load(wav_path)

    if backend == "wavjepa":
        # RuntimeJEPA 입력은 [B, T] 또는 [B, C, T] 를 허용.
        wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        audio = wav.to(device)
        with torch.no_grad():
            emb = backend_obj.get_scene_embeddings(audio)
        return emb.squeeze(0).detach().cpu().numpy()

    if backend == "hf":
        model, extractor = backend_obj
        wav = wav.mean(dim=0).numpy()
        inputs = extractor(wav, sampling_rate=sr, return_tensors="pt")
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

    raise ValueError(f"지원하지 않는 backend: {backend}")


def extract_split_embeddings(backend_obj: Any, backend: str, items, device: str):
    xs, ys = [], []
    for wav_path, label in tqdm(items, desc="embedding"):
        xs.append(get_embedding(backend_obj, backend, wav_path, device))
        ys.append(label)
    return np.stack(xs), np.array(ys)


def main() -> None:
    args = parse_args()

    train_items = load_split(args.dataset_root / "train")
    test_items = load_split(args.dataset_root / "test")

    if args.backend == "wavjepa":
        backend_obj = load_wavjepa_runtime(args.wavjepa_repo, args.ckpt_path, args.device)
    elif args.backend == "hf":
        if not args.model:
            raise ValueError("backend=hf 인 경우 --model 이 필요합니다.")
        backend_obj = load_hf_model(args.model, args.device)
    else:
        raise ValueError(f"지원하지 않는 backend: {args.backend}")

    x_train, y_train = extract_split_embeddings(backend_obj, args.backend, train_items, args.device)
    x_test, y_test = extract_split_embeddings(backend_obj, args.backend, test_items, args.device)

    clf = KNeighborsClassifier(n_neighbors=args.k, metric="cosine")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    print(f"test/acc: {accuracy_score(y_test, pred):.4f}")
    print(f"test/f1_macro: {f1_score(y_test, pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()

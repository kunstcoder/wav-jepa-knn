"""Local WavJEPA implementation copied/simplified from HF custom code.

목표:
- HF remote code를 런타임에 불러오지 않음
- 로컬 config.json + safetensors 기반으로 임베딩 추출
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import BatchFeature, SequenceFeatureExtractor


def normalize(audio: torch.Tensor) -> torch.Tensor:
    mean = audio.mean(dim=(-2, -1), keepdim=True)
    std = audio.std(dim=(-2, -1), keepdim=True)
    return (audio - mean) / (std + 1e-5)


def calculate_padding_mask(pad_frames, total_frames, sr, output_steps, process_seconds, device, batch_size):
    total_chunks = int((total_frames / sr) / process_seconds)
    total_output_steps = output_steps * total_chunks
    mask = torch.zeros((batch_size, total_output_steps), dtype=torch.bool, device=device)
    output_sr = int(output_steps / process_seconds)
    pad_seconds = pad_frames / sr
    pad_steps = int(pad_seconds * output_sr)
    if pad_steps > 0:
        mask[..., total_output_steps - pad_steps :] = True
    return mask, total_output_steps - pad_steps


class WavJEPAFeatureExtractor(SequenceFeatureExtractor):
    feature_extractor_type = "wavjepa-base"

    def __init__(self, feature_size=1, sampling_rate=16000, padding_value=0.0, **kwargs):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

    def _normalize_audio(self, audio_data: torch.Tensor, target_dbfs=-14.0):
        rms = torch.sqrt(torch.mean(audio_data**2))
        if rms == 0:
            return audio_data
        current_dbfs = 20 * torch.log10(rms)
        gain_db = target_dbfs - current_dbfs
        gain_linear = 10 ** (gain_db / 20)
        return audio_data * gain_linear

    def _extract_features(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        audio = torch.as_tensor(audio)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim == 2 and audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.ndim == 2 and audio.shape[0] == 1:
            return self._normalize_audio(audio)
        raise ValueError(f"지원하지 않는 오디오 shape: {tuple(audio.shape)}")

    def __call__(self, raw_speech, sampling_rate=None, return_tensors=None, **kwargs) -> BatchFeature:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"sampling_rate는 {self.sampling_rate} 이어야 합니다: {sampling_rate}")
        if isinstance(raw_speech, torch.Tensor):
            raw_speech = raw_speech.detach().cpu().numpy()
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim <= 2:
            raw_speech = [raw_speech]
        features = [self._extract_features(w) for w in raw_speech]
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        return BatchFeature({"input_values": features})


class ConvFeatureExtractor(nn.Module):
    def __init__(self, conv_layers_spec: list[tuple[int, int, int]], in_channels: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        layers: list[nn.Module] = []
        in_d = in_channels
        for i, (dim, kernel, stride) in enumerate(conv_layers_spec):
            block = [nn.Conv1d(in_d, dim, kernel, stride=stride, bias=False), nn.Dropout(p=dropout)]
            if i == 0:
                block.append(nn.GroupNorm(dim, dim, affine=True))
            block.append(nn.GELU())
            layers.append(nn.Sequential(*block))
            in_d = dim
        self.cnn = nn.Sequential(*layers)
        self.embedding_dim = conv_layers_spec[-1][0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x).transpose(1, 2)

    def total_patches(self, time: int) -> int:
        x = torch.zeros((1, self.in_channels, time), device=next(self.parameters()).device)
        return self.cnn(x).shape[-1]


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class WavJEPA(nn.Module):
    sample_rate = 16000
    process_audio_seconds = 2.01

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        conv_spec = [tuple(x) for x in cfg.get("conv_layers_spec", [[384, 8, 8], [384, 8, 8]])]
        self.extract_audio = ConvFeatureExtractor(conv_spec, in_channels=int(cfg.get("in_channels", 1)), dropout=float(cfg.get("conv_dropout", 0.0)))
        self.feature_norms = nn.LayerNorm(self.extract_audio.embedding_dim)

        d_model = int(cfg.get("encoder_d_model", cfg.get("hidden_size", 768)))
        nhead = int(cfg.get("encoder_nhead", 12))
        enc_layers = int(cfg.get("encoder_num_layers", 12))
        ff = int(cfg.get("encoder_dim_feedforward", d_model * 4))

        self.post_extraction_mapper = nn.Linear(self.extract_audio.embedding_dim, d_model) if self.extract_audio.embedding_dim != d_model else None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=float(cfg.get("encoder_dropout", 0.0)),
            activation=nn.GELU(),
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)

        self.total_patches = int(cfg.get("total_patches", 200))
        self.target_length = int(self.sample_rate * self.process_audio_seconds)
        self.output_steps = self.extract_audio.total_patches(self.target_length)

        pos = np.arange(self.total_patches, dtype=np.float64)
        pe = get_1d_sincos_pos_embed_from_grid(d_model, pos)
        self.pos_encoding_encoder = nn.Parameter(torch.from_numpy(pe).float().unsqueeze(0), requires_grad=False)

    @torch.inference_mode()
    def _get_segment_representation(self, audio: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        local_features = self.extract_audio(audio)
        local_features = self.feature_norms(local_features)
        if self.post_extraction_mapper is not None:
            local_features = self.post_extraction_mapper(local_features)
        local_features = local_features + self.pos_encoding_encoder[:, : local_features.shape[1], :]
        return self.encoder(local_features, src_key_padding_mask=padding_mask)

    @torch.inference_mode()
    def get_audio_representation(self, audio: torch.Tensor):
        if audio.ndim != 3:
            raise ValueError("audio input tensor must be 3D with shape (B, C, T)")

        batch_size = audio.shape[0]
        pad_frames = self.target_length - (audio.shape[-1] % self.target_length)
        if pad_frames > 0 and pad_frames < self.target_length:
            audio = torch.nn.functional.pad(audio, (0, pad_frames), mode="constant")
        else:
            pad_frames = 0

        padding_mask, cut_off = calculate_padding_mask(
            pad_frames=pad_frames,
            total_frames=audio.shape[-1],
            sr=self.sample_rate,
            output_steps=self.output_steps,
            process_seconds=self.target_length // self.sample_rate,
            device=audio.device,
            batch_size=batch_size,
        )

        embeddings = []
        mask_idx = 0
        for i in range(audio.shape[-1] // self.target_length):
            chunk = audio[..., i * self.target_length : (i + 1) * self.target_length]
            mask = padding_mask[..., mask_idx : mask_idx + self.output_steps]
            embedding = self._get_segment_representation(normalize(chunk), mask)
            mask_idx += self.output_steps
            embeddings.append(embedding)

        x = torch.hstack(embeddings)
        return x[:, :cut_off, :], None


class LocalWavJEPAInference(nn.Module):
    def __init__(self, model: WavJEPA):
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor):
        return self.model.get_audio_representation(input_values)


def _find_weight_files(model_dir: Path) -> list[Path]:
    index_json = model_dir / "model.safetensors.index.json"
    if index_json.exists():
        index_data = json.loads(index_json.read_text(encoding="utf-8"))
        weight_map = index_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        shards = [model_dir / name for name in shard_names]
        missing = [p for p in shards if not p.exists()]
        if missing:
            raise FileNotFoundError(f"index에 명시된 shard 파일이 없습니다: {missing[:3]}")
        return shards

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"safetensors 파일이 없습니다: {model_dir}")
    return files


def _load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for file in _find_weight_files(model_dir):
        merged.update(load_file(str(file)))
    return merged


def _strip_common_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ["model.", "module.", "wavjepa.", "backbone.", "jepa."]
    out = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _extract_arch_cfg(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    for key in ("wavjepa_config", "model_config", "model_args", "config"):
        if key in raw_cfg and isinstance(raw_cfg[key], dict):
            return raw_cfg[key]
    return raw_cfg


def load_local_wavjepa(model_dir: Path, device: str):
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json 파일이 없습니다: {config_path}")

    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    arch_cfg = _extract_arch_cfg(raw_cfg)

    model = WavJEPA(arch_cfg)
    raw_state_dict = _load_state_dict(model_dir)
    state_dict = _strip_common_prefixes(raw_state_dict)

    # startswith('encoder') 대신, key 내 포함 여부로 더 유연하게 판별
    if not any("encoder" in k for k in state_dict.keys()):
        preview = list(state_dict.keys())[:10]
        raise RuntimeError(
            "WavJEPA 형태의 가중치로 보이지 않습니다. "
            f"state_dict key 예시: {preview}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # 완전 불일치 상황을 더 명확히 안내
    if len(missing) > 200 and len(unexpected) > 200:
        raise RuntimeError(
            "config/safetensors 구조 불일치가 큽니다. "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    model = model.to(device).eval()
    extractor = WavJEPAFeatureExtractor(sampling_rate=16000)
    info = {
        "missing": len(missing),
        "unexpected": len(unexpected),
        "num_tensors": len(state_dict),
    }
    return LocalWavJEPAInference(model), extractor, info

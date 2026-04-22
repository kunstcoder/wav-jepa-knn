# TODO - WavJEPA kNN 평가 코드

- [x] 데이터셋 폴더 구조(train/test, wav+json)에 맞는 로더 구현
- [x] HuggingFace 인터페이스(`AutoModel` + `AutoFeatureExtractor`) 기반 임베딩 추출 구현
- [x] 실행 시 원격 HF 참조 없이 로컬 모델 디렉토리만 사용하도록 정리
- [x] 임베딩 벡터 기반 cosine kNN 분류 및 Acc/F1 측정 구현
- [x] 최소 코드 형태로 단일 스크립트 작성 (`knn_eval.py`)
- [x] 실행 방법 문서화 (`README.md`)
- [x] `requirements.txt` 추가
- [x] `.gitignore` 추가
- [x] `CODEX_RULES.md` 추가

## 실행 예시

```bash
python knn_eval.py \
  --dataset_root /path/to/your_dataset \
  --model /path/to/local_wavjepa_model \
  --k 20
```

## 메모

- `--model`은 HF 리포 ID가 아니라 **로컬 디렉토리 경로**를 기대합니다.
- 로컬 디렉토리에 `config.json`, (remote code를 복사한)`model.py`, `feature_extractor.py`, `model.safetensors` 등이 필요합니다.

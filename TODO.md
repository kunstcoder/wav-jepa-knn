# TODO - WavJEPA kNN 평가 코드

- [x] 데이터셋 폴더 구조(train/test, wav+json)에 맞는 로더 구현
- [x] 임베딩 벡터 기반 cosine kNN 분류 및 Acc/F1 측정 구현
- [x] 최소 코드 형태로 단일 스크립트 작성 (`knn_eval.py`)
- [x] 실행 방법 문서화
- [x] `requirements.txt` 추가
- [x] `.gitignore` 추가
- [x] `CODEX_RULES.md` 추가

## 현재 기본 동작

- [x] **기본 백엔드 `wavjepa`**: 로컬 `wavjepa` GitHub 코드 + 로컬 `.ckpt` 로딩
- [x] **호환 백엔드 `hf`**: 기존 HuggingFace 추론 경로 유지

## 실행 예시

### 1) 권장: wavjepa github + local ckpt

```bash
python knn_eval.py \
  --dataset_root /path/to/your_dataset \
  --backend wavjepa \
  --wavjepa_repo /path/to/wavjepa \
  --ckpt_path /path/to/model.ckpt \
  --k 20
```

### 2) 호환: HuggingFace

```bash
python knn_eval.py \
  --dataset_root /path/to/your_dataset \
  --backend hf \
  --model labhamlet/wavjepa-base \
  --k 20
```

## 메모

- `wavjepa` 경로는 `hear_configs.WavJEPA.load_model(...)` 흐름을 따라 로컬 ckpt를 불러옵니다.
- 환경/버전에 따라 원본 리포 코드와 체크포인트 키 네이밍이 다를 수 있어, 필요 시 로딩 매핑 로직 수정이 필요할 수 있습니다.

# TODO - WavJEPA kNN 평가 코드

- [x] 데이터셋 폴더 구조(train/test, wav+json)에 맞는 로더 구현
- [x] WavJEPA HuggingFace 추론 인터페이스(`AutoModel` + `AutoFeatureExtractor`) 기반 임베딩 추출 구현
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
  --model labhamlet/wavjepa-base \
  --k 20
```

## 메모

- 현재 버전은 HuggingFace 로딩 경로를 기준으로 동작합니다.
- 로컬 체크포인트를 사용하려면 HF 형식으로 로드 가능한 디렉토리/모델 구성이 필요합니다.

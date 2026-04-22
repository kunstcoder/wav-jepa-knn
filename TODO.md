# TODO - WavJEPA kNN 평가 코드

- [x] 데이터셋 폴더 구조(train/test, wav+json)에 맞는 로더 구현
- [x] WavJEPA 공식 추론 인터페이스(`AutoModel` + `AutoFeatureExtractor`) 기준으로 임베딩 추출 구현
- [x] 임베딩 벡터 기반 cosine kNN 분류 및 Acc/F1 측정 구현
- [x] 최소 코드 형태로 단일 스크립트 작성 (`knn_eval.py`)
- [x] 실행 방법 문서화
- [x] 이전 버전의 불확실한 fallback 로직 제거(동작 확실성 개선)

## 실행 예시

```bash
python knn_eval.py \
  --dataset_root /path/to/your_dataset \
  --model labhamlet/wavjepa-base \
  --k 20
```

## 메모

- 이번 버전은 WavJEPA README의 추론 패턴과 동일하게 `extractor(audio) -> model(**inputs)` 흐름을 사용.
- 즉, 임베딩 추출 경로를 단순화해 "짧지만 실제 인터페이스와 맞는" 코드로 정리함.
- 학습 ckpt가 HuggingFace 형식이 아니라면(예: Lightning raw ckpt), 먼저 HF 로딩 가능 형태로 변환/내보내기가 필요할 수 있음.

---

## 추가 작업 (요청 반영)

- [x] `requirements.txt` 추가
- [x] `.gitignore` 추가
- [x] `CODEX_RULES.md` 추가

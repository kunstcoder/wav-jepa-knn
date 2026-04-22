# TODO - WavJEPA kNN 평가 코드

- [x] 데이터셋 폴더 구조(train/test, wav+json)에 맞는 로더 구현
- [x] HF remote code를 로드하지 않는 로컬 클래스 기반 로더 구현 (`local_wavjepa.py`)
- [x] 로컬 `config.json` + `*.safetensors` 로딩 구현
- [x] safetensors shard index(`model.safetensors.index.json`) 및 key prefix 정규화 지원
- [x] 임베딩 벡터 기반 cosine kNN 분류 및 Acc/F1 측정 구현
- [x] 최소 코드 형태로 단일 스크립트 작성 (`knn_eval.py`)
- [x] 실행 방법 문서화 (`README.md`)
- [x] `requirements.txt` 갱신 (`safetensors` 추가)

## 실행 예시

```bash
python knn_eval.py \
  --dataset_root /path/to/your_dataset \
  --model /path/to/local_wavjepa_model \
  --k 20
```

## 메모

- `knn_eval.py`는 `local_wavjepa.load_local_wavjepa()`를 통해 내부 구현 클래스로 모델을 구성합니다.
- safetensors 키 구조가 다르면 `missing/unexpected`가 크게 발생할 수 있습니다.

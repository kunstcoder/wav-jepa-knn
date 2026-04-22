# Codex Rules (Repo Local)

이 저장소에서 Codex가 작업할 때 지켜야 할 기본 규칙입니다.

## 목표
- `knn_eval.py` 중심의 최소 코드 유지
- 데이터셋 구조(`train/`, `test/` + `wav/json`) 가정 유지
- 변경 시 `TODO.md` 업데이트

## 코딩 규칙
- 불필요한 추상화/복잡한 fallback 추가 금지
- 에러 메시지는 사용자가 원인 파악 가능하도록 명확히 작성
- 새 의존성 추가 시 `requirements.txt` 갱신

## 작업 규칙
- 변경 파일이 생기면 실행 방법/주의사항을 `TODO.md`에 반영
- 커밋 메시지는 변경 목적이 드러나게 작성

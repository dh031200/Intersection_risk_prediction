210812 - BJ edits
- trt_yolo_cv_bj1.py: 기존 try_yolo_cv.py 파일에서 프레임별 객체 정보를 json형태로 출력하는 함수 추가 (영상출력부 주석처리함)
- logs_by_frame 디렉토리: try_yolo_cv_bj1.py에서 추출된 프레임별 객체 정보 json파일을 저장
- 프레임별 객체 정보 json 형식 예제 {'1': ['tmp_id': 0, 'min_x':111, 'min_y': 145, 'max_x':168, 'max_y':187, class': 1}, 'tmp_id': 1, 'min_x':457, 'min_y': 101, 'max_x':523, 'max_y':142, class': 1}, ...]}
- load_json.py: 프레임별 객체정보 json 파일 읽어오는 코드 

210813 - DH edits
- grid_array.py: 확인용 시각화코드 삭제
- src 디렉토리: 소스코드에 필요한 이미지들 저장 (seg_map_v2_short.png)

210825 - BJ edits
- func_grid_matching.py: 객체 트래킹 후 그리드에 맵핑하는 작업을 수행하는 함수
- ---> 현재 트래킹 까지완료 

210826 - BJ edits
- ---> 현재 트래킹 후 객체 속도 추출 함수 까지 작성 완료 (K, FPS, SAMPLING_RATE 등 조정필요)
- ---> To do: 트래킹 성능 고도화 (객체별로 사람->사람 트래킹) (완)
- ---> To do: 각 객체 그리드 맵핑 (완)

210827 - BJ eidts
- 트래킹 성능 고도화 (class 별로 추적 가능) 완료
- 각 객체를 그리드에 맵핑 완료
- ---> To do: 트래킹 누적 후 경로 분포 추정을 통한 경로 예측 알고리즘 개발

210830 - BJ edits
- 트래킹 성능 고도화 기능 수정 (class 별로 최소 거리 파라미터 수정 가능하도록)

210901 - BJ edits
- 각 객체의 그리드 맵핑을 위한 그리드 함수 설계 완료
- 블루시그널 측의 그리드 맵핑 방식을 채택하지 않고, 독자적인 그리드 생성

210902 - BJ edits
- 객체가 그리드에 잘 맵핑 되는지 테스트 완료 (잘 맵핑 됨)
- 그리드 단위의 트래킹 함수 작성 중

210903 - BJ edits
- 그리드 단위의 트래킹 함수 구현 완료
- ---> To do: 객체 행동 정보 추출 함수 구현 예정

210906 - BJ edits
- 그리드 단위에서 객체 행동 정보 추출 함수 구현 완료 (테스트 예정)

210910 - BJ edits
- 그리드 단위에서 객체 행동 정보 추출이 가능하도록 데이터 구조 설계 완료
- 그리드 단위 객체 행동 정보 추출 함수 구현 완료 (테스트 완료) 
- ---> To do: 나비박스 설치 후 진행 예정 (9월 27일 설치)

210930 - BJ edits
- 객체 트래킹 함수 작성 완료
- 그리드 매칭 함수 작성 완료
- 위험도 식별 (임시알고리즘) 함수 일부 완료 후 탑재
- 위험도 식별 sign 메세지 전송 함수 작성
- 통합 테스트를 위한 일부 함수 통합 작업 완료
- ---> To do: 위험도 식별 알고리즘  (카메라 설치 이후)

211005 - BJ edits
- 스피커 경고 시간을 고려하여 time leg (5초) 발생시키도록 구현 완료
- ---> To do: 위험도 식별 알고리즘  (카메라 설치 이후)

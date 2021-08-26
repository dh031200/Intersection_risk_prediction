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
- ---> 현재 트래킹 까지완료 ---> 현재 트래킹 후 객체 속도 추출 함수 까지 작성 완료 (K, FPS, SAMPLING_RATE 등 조정필요)
- ---> To do: 트래킹 성능 고도화 (객체별로 사람->사람 트래킹)
- ---> To do: 각 객체 그리드 맵핑

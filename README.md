210812 - BJ edits
- trt_yolo_cv_bj1.py: 기존 try_yolo_cv.py 파일에서 프레임별 객체 정보를 json형태로 출력하는 함수 추가 (영상출력부 주석처리함)
- logs_by_frame 디렉토리: try_yolo_cv_bj1.py에서 추출된 프레임별 객체 정보 json파일을 저장
- 프레임별 객체 정보 json 형식 예제 {'1': ['tmp_id': 0, 'min_x':111, 'min_y': 145, 'max_x':168, 'max_y':187, class': 1}, 'tmp_id': 1, 'min_x':457, 'min_y': 101, 'max_x':523, 'max_y':142, class': 1}, ...]}
- load_json.py: 프레임별 객체정보 json 파일 읽어오는 코드 

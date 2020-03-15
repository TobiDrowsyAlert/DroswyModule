# DroswyModule
얼굴 좌표를 감지하여 졸음 징후를 파악하고 졸음 단계를 계산하는 모듈입니다.
중요 모듈로는 현재
landmark_process
sleep_step_calculation
이 있습니다.

## landmark_process
얼굴 좌표 처리를 담당하는 모듈입니다.
눈, 입 종횡비, 얼굴 각도 계산, 운전자 인식 여부 기능 등 얼굴 좌표를 통해 값을 얻는 기능을 수행합니다.

## sleep_step_calcaultion
졸음 단계를 계산 모듈입니다.
졸음 징후를 습득하여 졸음 단계를 변화시키고, 어떤 징후가 졸음 징후를 변화시켰는지 관리하는 기능을 수행합니다.

from keras.models import load_model
import cv2
import numpy as np

# 모델 로드
model = load_model("keras_Model.h5", compile=False)

# 레이블 로드
class_names = open("labels.txt", "r", encoding='utf-8').readlines()

# 카메라에 따라 0 또는 1로 설정
camera = cv2.VideoCapture(0)

while True:
    # 카메라에서 이미지 가져오기
    ret, image = camera.read()
    if not ret:
        continue  # 이미지가 제대로 읽히지 않았으면 다음 루프로 넘어감

    # 원시 이미지 크기 조정 (224 높이, 224 너비)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지 배열 정규화
    image = (image / 255.0).astype(np.float32)

    # 이미지를 모델 입력 형태로 재구성
    image = np.expand_dims(image, axis=0)

    # 모델 예측
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # 예측 및 신뢰도 점수 출력
    print("클래스:", class_name.strip(), end="")
    print("\n정확도:", str(np.round(confidence_score * 100))[:-2], "%")

    # 키보드 입력 대기
    keyboard_input = cv2.waitKey(1)

    # 키보드 입력이 27(ESC)일 때 종료
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

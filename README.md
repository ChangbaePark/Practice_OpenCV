# Practice_OpenCV
## Languages 
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">

## 프로그램에 대한 설명
- 이 프로그램은 웹캠을 이용합니다.
- 4가지 Class를 식별합니다.(사람, 마우스, 스마트폰, 키보드)

## 파일 설명
### 1. camKeras.py
- camKeras.py 파일은 Keras를 사용하여 웹캠으로부터 영상을 받아 실시간으로 객체를 인식하는 파이썬 스크립트입니다. 이 스크립트는 모델을 로드하고, 웹캠에서 영상을 입력받아, 실시간으로 객체를 인식하고 결과를 화면에 표시합니다.

### 2. camKeras_pyqt.py
- camKeras_pyqt.py는 camKeras.py의 기능을 기반으로 하되, PyQt를 사용하여 GUI를 제공합니다. 이 스크립트는 사용자에게 보다 친근하고 직관적인 인터페이스를 제공하여 객체 인식 결과를 시각화합니다. 사용자는 GUI를 통해 인식된 객체 정보를 더욱 효과적으로 볼 수 있습니다.

### 3. keras_model.h5
- keras_model.h5 파일은 훈련된 Keras 모델을 포함하고 있습니다. 이 모델은 camKeras.py 및 camKeras_pyqt.py에서 로드되어 실시간 객체 인식에 사용됩니다. 모델은 다음 객체들을 인식할 수 있도록 훈련되었습니다: 사람, 마우스, 스마트폰, 키보드

### 4. labels.txt
- labels.txt 파일은 모델이 인식할 수 있는 객체의 카테고리를 정의합니다. 각 라인에는 하나의 객체 카테고리가 지정되어 있으며, 모델의 예측 결과와 연동되어 사용됩니다

- 0: 사람 1: 마우스 2: 스마트폰 3: 키보드

## 프로그램 사용 방법
### 1. 가상환경 생성
    conda create -n camKeras python=3.9
- VScode 터미널 또는 cmd에서 해당 명령어 입력
- 해당 가상환경을 생성하기 위해서는 사전에 ANACONDA라는 프로그램을 Install 필요

### 2. 가상환경과 VSCode를 연결
- CTRL+SHIFT+P -> Select Python interpreter (새로 만든 가상환경 이름이 보이지 않으면 VSCode를 종료 후 재시작)

### 3. 새 터미널 확인 
- 가상환경이 적용되었는지 확인

### 4. Package 설치
     pip install -r requirements.txt 
- 해당 명령어를 사용하여 패키지 설치

### 5. camKeras_pyqt.py 파일 실행
- VSCode로 camKeras_pyqt.py 오픈후 실행

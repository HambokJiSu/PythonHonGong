import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # 최근접 이웃 알고리즘
from sklearn.model_selection import train_test_split

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# np.column_stack : 배열을 나란히 연결
fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))  # np.concatenate 배열 연결, 파라미터를 튜플 형태로 전달해야함

# 기본적으로 25%를 테스트 세트로 때어냄
# stratify=타깃 데이터, 타깃 데이터 비율에 맞게 샘플 배율을 조정해줌 (훈련 데이터가 적거나 특정 클래스의 샘플 개수가 적을 때 특히 유용)
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target) # 1.0

kn.predict([[25, 150]]) # [0.]

mean = np.mean(train_input, axis=0) # np.mean : 평균 계산, axis=0 : 행을 따라 각 열의 통계 값을 계산
std = np.std(train_input, axis=0) # np.std : 표준 편차 계산, axis=0 : 행을 따라 각 열의 통계 값을 계산

train_scaled = (train_input - mean) / std # train_input 배열 값 값에 mean만큼씩 빼고 std만큼씩 나누기
kn.fit(train_scaled, train_target)  # 스케일 변경된 모델 기준으로 훈련

new = ([25, 150] - mean) / std  #   테스트 건도 표준점수 연산 처리
distances, indexes = kn.kneighbors([new])   # 재훈련된 내용 기준 이웃값 재산출

plt.scatter(train_scaled[:, 0], train_scaled[:, 1]) # 훈련내용
plt.scatter(new[0], new[1], marker="^") # 테스트 내용
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1]) # 테스트 내용의 이웃값

plt.xlabel("length")
plt.ylabel("weight")
plt.show()
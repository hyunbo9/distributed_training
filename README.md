# distibuted_train

# setting

<img width="980" alt="스크린샷 2019-12-04 오후 3 40 11" src="https://user-images.githubusercontent.com/26562858/70119143-fc38fa80-16ac-11ea-9b42-c9381d9725dc.png" style="max-width:100%;">

# result

<img width="981" alt="스크린샷 2019-12-04 오후 3 40 00" src="https://user-images.githubusercontent.com/26562858/70119145-00fdae80-16ad-11ea-8088-15fe09c56cd1.png" style="max-width:100%;">


gpu 8개인 경우가 gpu 4대인 경우보다 학습 속도가 느린 이유 :

모델의 복잡성 대비 사용한 gpu가 너무 고성능이라서 gradient를 계산하는 속도는 빠른데 계산된 gradient를 공유 하는 시간이 오래걸리는 것으로 추정.

더 복잡한 모델에서 배치 사이즈를 늘려서 실험하면 gpu 8개 일때가 gpu 4대일 때보다 성능이 좋게 나올 거 같음.

결론 : 모델 사이즈에 맞게 gpu 개수를 설정해야 함.

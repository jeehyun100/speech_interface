# Dvector 숙제

## Requirement
  - torch                1.5.0
  - librosa              0.7.2
  - tensorflow           1.15.0
  - tensorboardX         2.0
  - librosa              0.7.2
  - scikit-learn         0.22.2.post1 
  - scipy                1.4.1
  
## How to Augmentation

   - python dvector_preprocess.py에서 main()에서 순서대로 주석을 풀면서 실행
   - 5초 씩 자르기 : split_5sec()
   - Augmentation : audio_aug()
   - txt만들기 : preprocess_ai_class_data(filepath)
   
## How to Train

   - python PJ4_trainer.py
   
## How to test

   - python model_test.py
   
## The result
!acc(./img/acc.png)
!loss(./img/loss.png)


## 경진대회 개요

경진대회 링크: https://dacon.io/competitions/official/235687/data/

* **주제**

  > **비식별화된 에러와 퀄리티 로그 및 수치 데이터를 분석**하여 시스템 품질 변화로 **사용자에게 불편을 야기하는 요인을 진단**한다. 

  EDA를 통해 에러와 퀄리티 데이터에 있는 변수가 무엇인지 파악하는 과정이 중요하다. 그 과정에서 나타나는 특징들을 사용자 관점으로 해석하려는 노력이 필요할 것으로 보인다.  

* **배경**

  > 다양한 장비/ 서비스에서 일어나는 시스템 데이터를 통해 사용자의 불편을 예지하기 위해 '시스템 데이터'와 '사용자 불편 발생 데이터'를 분석하여 불편을 느낀 사용자와 불편요인을 찾고자 한다.

* **목적**

  > 1. 데이터를 통해 사용자가 불편을 느끼는 원인 분석
  >
  > 2. 사용자 관점의 데이터 분석 능력이 뛰어난 인재 발굴

* **주최/주관**

  > 주최: LG AI Research
  >
  > 주관: 데이콘

* **사용 데이터**

  - 학습 데이터 (user_id : 10000 ~ 24999, 15000명)

    train_err_data.csv : 시스템에 발생한 에러 로그

    train_quality_data.csv : 시스템 퀄리티 로그

    train_problem_data.csv : 사용자 불만 및 불만이 접수된 시간

  

  - 테스트 데이터(user_id : 30000 ~ 44998, 14999명)

    test_err_data.csv : 시스템에 발생한 에러 로그

    test_quality_data.csv : 시스템 퀄리티 로그

    sample_submission.csv : 사용자 불만 확률(0~1) (제출용)

* 사용 모델

  생성한 변수들을 대상으로 *light GBM모델*을 구축하였다. 모델의 파라미터는 과적합을 최소로, 정확도는 최대가 될 수 있도록 조정하였다. 

  그 결과 test_set에 대해 정확도 *최대 83.2%*를 도출할 수 있었다. 

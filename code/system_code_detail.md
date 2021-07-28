# System.py 추가 설명

## 분석코드

*분석 코드는  github에서 열람 가능합니다*

system.py와 해당 설명 내용은 EDA 및 변수 생성 코드 중 작성자가 작성한 코드만 부분적으로 발췌한 것이다.

### Baseline

baseline에선 x를 user_id이 겪은 에러타입의 개수를 x로, 불만 수를 y로 두고 light GBM을 모델로 사용하였다. 

* x: user_id별 errtype개수 ( 행: user_id, 열: errtype별 개수)

  ![image-20210207211046537](C:\Users\seo\AppData\Roaming\Typora\typora-user-images\image-20210207211046537.png)

  본 조는 baseline에서 제공된 **x dataset에 파생변수**를 붙여나가는 식으로 정확도를 올리고자 하였다. **y는 그대로 user_id별 complain의 개수로 사용하였고 모델은 light GBM의 파라미터를 조정**하여 사용하였다. 

  

### 사용할 데이터셋 생성

본격적으로 분석을 하기 앞서, 에러와 불만 사이의 관계를 분석하기 위해, **에러로그와 불만데이터를 합친 데이터 셋**을 생성하고자 한다.

```python
# prob데이터 셋에 어떤 모델에 대한 compalin인지 used_model열 추가
# train_prob의 user_id가 불만을 제기한 time이전에 사용했던 model(들) 중 가장 최근에 사용했던 모델을 model_nm열에 추가
# model_nm리스트가 0인 경우가 있음-> 불만 제기 이전에 사용했던 모델이 train_err 데이터 셋엔 없을 수도 있다고 추정/ 이 경우엔 이후에 사용한 모델 중 가장 최근에 사용한 모델을 가져옴
# 마찬가지로 errtype에 대해서도 적용(불만제기 이전에 가장 최근에 발생한 errtype) 
# 마찬가지로 fwver에 대해서도 적용(model_nm은 삭제)

train_prob['fwver']=''
train_prob['errtype']=''

for i in range(len(train_prob)):
    try:
        train_prob['errtype'][i]= train_err['errtype'][(train_err['user_id']== train_prob['user_id'][i]) & (train_err['time']<=train_prob['time'][i])].unique()[-1]
        train_prob['fwver'][i]= train_err['fwver'][(train_err['user_id']== train_prob['user_id'][i]) & (train_err['time']<=train_prob['time'][i])].unique()[-1]
        
    except:
        train_prob['fwver'][i]= train_err['fwver'][(train_err['user_id']== train_prob['user_id'][i]) & (train_err['time']>=train_prob['time'][i])].unique()[0]
        
        
# train_err에 complain몇번 했는지 보여주는 complain열 추가
complain_cnt= train_prob.groupby(['user_id','fwver','errtype'],as_index=False).count()
complain_cnt.rename({'time':'complain'}, axis=1, inplace=True)

# train_err와 comlain_cnt를 통합
train_err1= pd.merge(train_err, complain_cnt, how= 'outer',on=['user_id', 'errtype','fwver'])

#train_err에서 user_id가 동일하면 해당 user_id의 complain열은 하나의 값을 제외하고 모두 0으로 변경 -> complain 중복 집계방지
wrong = train_err1[train_err1.duplicated(['user_id','model_nm','errtype','complain','fwver'],keep='first')].index
train_err1['complain'][wrong] = 0

train_err1.to_csv(".//train_prob1_1.csv", index=False)
```

### EDA

1. 모델과 fwver에 따른 complain수와 error 발생 수 집계

   모델과 fwver(펌웨어 버전)에 따라 발생하는 에러 수와 complain수가 상이할 것으로 생각했다. EDA 전, *''에러가 많이 발생하는 fwver일 수록 complain수도 많을 것이다.'*라는 가정을 세웠다. 

   이를 확인하기 위해 train_err에 대한 전처리 후 fwver에 따른 complain수와 err수를 집계한 막대그래프를 생성하였다. 

   ```python
   err_type=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]
   
   # 에러타입 에러 수 세는 함수 
   def errcnt(df,r,c):
       value= df.values
       mat= np.zeros((r,c))
       
       for row,col in value:
           mat[row,col-1] +=1
       return mat
   
   # 모델 별로 에러타입 개수 집계
   for i in range(9):
       model= train_err[train_err['model_nm']=='model_%d'%i]
   
       fwindex= pd.DataFrame(model['fwver'].unique())
   
       fwindex.reset_index(inplace=True)
       fwindex= fwindex.rename({0:'fwver'}, axis=1)
       fwindex
       leng= len(fwindex)
       # fw_index의 index를 train_err의 fwver에 맞게 매칭
   
       model= pd.merge(model, fwindex, on='fwver')
       
       df= model[['index','errtype']]
       r= leng
       c= 42
       
       model_err= pd.DataFrame(errcnt(df,r,c))
       model_err= pd.concat([fwindex,model_err], axis=1)
       model_err.drop('index',inplace=True, axis=1)
       model_err['err_sum']=model_err[err_type].sum(axis=1)
       globals()['model{}_err'.format(i)]= model_err
       
   model0_err['model_nm']='model_0'
   model1_err['model_nm']='model_1'
   model2_err['model_nm']='model_2'
   model3_err['model_nm']='model_3'
   model4_err['model_nm']='model_4'
   model5_err['model_nm']='model_5'
   model6_err['model_nm']='model_6'
   model7_err['model_nm']='model_7'
   model8_err['model_nm']='model_8'
   
   # model모두 합쳐주기
   model_fw_error= pd.concat([model0_err,model1_err,model2_err,model3_err,model4_err,model5_err,model6_err,model7_err,model8_err], axis=0)
   
   # 모델과 fwver에 따른 complain 수 집계
   # complain이 존재하는 행만 있는 데이터프레임 만들기
   train_err2= train_err1[train_err1['complain']!=0]
   
   # 에러타입에 따른 complain 수 세는 함수 
   def compcnt(df,r,c):
       value= df.values
       mat= np.zeros((r,c))
       
       for row,col,val in value:
           mat[row,col-1] += val
       return mat
   for i in range(9):
       model= train_err2[train_err2['model_nm']=='model_%d'%i]
   
       fwindex= pd.DataFrame(model['fwver'].unique())
   
       fwindex.reset_index(inplace=True)
       fwindex= fwindex.rename({0:'fwver'}, axis=1)
       fwindex
       leng= len(fwindex)
       # fw_index의 index를 train_err의 fwver에 맞게 매칭
   
       model= pd.merge(model, fwindex, on='fwver')
       
       df= model[['index','errtype','complain']]
       r= leng
       c= 42
       
       model_comp= pd.DataFrame(compcnt(df,r,c))
       model_comp= pd.concat([fwindex,model_comp], axis=1)
       model_comp.drop('index',inplace=True, axis=1)
       model_comp['comp_sum']=model_comp[err_type].sum(axis=1)
       globals()['model{}_comp'.format(i)]= model_comp
       
   model0_comp['model_nm']='model_0'
   model1_comp['model_nm']='model_1'
   model2_comp['model_nm']='model_2'
   model3_comp['model_nm']='model_3'
   model4_comp['model_nm']='model_4'
   model5_comp['model_nm']='model_5'
   model6_comp['model_nm']='model_6'
   model7_comp['model_nm']='model_7'
   model8_comp['model_nm']='model_8'
   
   # model모두 합쳐주기
   model_fw_comp= pd.concat([model0_comp,model1_comp,model2_comp,model3_comp,model4_comp,model5_comp,model6_comp,model7_comp,model8_comp], axis=0)
   ```

   ```python
   # model에 따른 fwver별 complain 수와 error발생 합계 시각화
   plt.figure(figsize=(10,30))
   plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=1)
   
   n=1
   for i in range(9):
       plt.subplot(9,2,n)
       plt.xticks(rotation = - 45 )
       plot= sns.barplot(model_fw_comp[model_fw_comp['model_nm']=='model_%d'%i]['fwver'], model_fw_comp[model_fw_comp['model_nm']=='model_%d'%i]['comp_sum'])
       plot.set_title('the number of complain in model%d'%i)
       n=n+1
       plt.subplot(9,2,n)
       plt.xticks(rotation = - 45 )
       plot2= sns.barplot(model_fw_err[model_fw_err['model_nm']=='model_%d'%i]['fwver'],model_fw_err[model_fw_err['model_nm']=='model_%d'%i]['err_sum'])
       plot2.set_title('the number of error in model%d'%i)
       n=n+1
   ```

   ![image-20210207220840605](C:\Users\seo\AppData\Roaming\Typora\typora-user-images\image-20210207220840605.png)

   간단한 EDA를 통해 앞서 세운 가정이 성립함을 확인하였다. 그래프에서 볼 수 있다시피, 특정 model에서 fwver의 에러 개수가 높을수록 불만 수가 높음을 확인할 수 있었다. 

   

2. 모델, fwver에 따른 errtype별 complain 수와 err수 집계

   앞서 생성한 model_fw_comp와 model_fw_err데이터 셋을 활용하여 **모델과 fwver에 따른 에러타입 별 불만 수와 에러 수를 집계하여 막대그래프**로 보고자 한다. 

   ![image-20210207221335106](C:\Users\seo\AppData\Roaming\Typora\typora-user-images\image-20210207221335106.png)

   ​	1번 EDA와는 달리 에러 수와 불만 수 사이에 특정 관계는 보이지 않는다. 



### 파생변수 생성



1. model_nm과 fwver에 따른 에러 개수와 불만 수의 비율

   모델에 따라 fwver 별로 err와 complain이 차지하는 비율을 계산한 후 comp_ratio에서 err_ratio을 나누어 비율을 산출하였다.

   ```python
   #model_fw_comp의 comp_sum과 model_fw_err의 err_sum합치기
   agg_df= pd.merge(model_fw_comp[['fwver','comp_sum']], model_fw_err[['fwver','err_sum','model_nm']], on='fwver',how='right')
   
   agg_df= agg_df.fillna(0)
   
   # 각 모델별로 comp_sum과 err_sum의 합 산출해서 붙여주기
   temp= agg_df.groupby('model_nm', as_index=False).sum()
   temp.rename({'comp_sum':'model_comp', 'err_sum':'model_err'}, axis=1, inplace=True)
   agg_df= pd.merge(agg_df, temp, on='model_nm', how= 'left')
   
   # 각 모델별 comp_sum과 err_sum의 비율 산출해서 붙여주기
   agg_df['comp_ratio']= agg_df['comp_sum']/agg_df['model_comp']
   agg_df['err_ratio']= agg_df['err_sum']/agg_df['model_err']
   
   # 최종 comp/err 비율 'ratio' 산출
   agg_df['ratio']= agg_df['comp_ratio']/ agg_df['err_ratio']
   ```

    ![image-20210207222923845](C:\Users\seo\AppData\Roaming\Typora\typora-user-images\image-20210207222923845.png)

   해당  ratio를 x의 user_id가 사용한 fwver에 맞게 붙여주었다. 

   ```python
   # df_error에 fwver 붙이기- 두개 이상 사용했을 경우 에러가 더 많이 집계된 fwver으로 사용
   # train_err를 user_id와 fwver로 그루핑, 개수집계-> 한 유저 아이디에 하위그룹인 fwver이 2개 이상 있을 때 개수가 더 많은 fwver선택
   err_group= train_err.groupby(['user_id','fwver'], as_index=False)['time'].count()
   user_dup= err_group[err_group['user_id'].duplicated()]['user_id'].values
   drop_idx=[]
   for i in user_dup:
       temp= err_group[err_group['user_id']==i]
       drop_idx=drop_idx+(temp[temp['time']!=temp['time'].max()].index.to_list())
   
   # drop_idx적용하여 err_group의 해당 인덱스 행을 삭제
   # reset_index
   err_group= err_group.drop(index= drop_idx)
   err_group.reset_index(inplace=True, drop=True)
   
   # time이 동일한 user_id인 경우 하나만 남기고 삭제
   err_group= err_group.drop_duplicates('user_id', keep='first')
   err_group.reset_index(drop=True, inplace=True)
   
   # df_error의 user_id에 맞게 fwver 붙여주기(나중에 error num도 붙이고 싶으면 이코드 수정)
   df_error= pd.merge(df_error,err_group[['user_id','fwver']], on='user_id', how='inner')
   
   # df_error의 fwver에 맞는 ratio 붙여주기
   # ratio table에 없는 fwver은 한번도 불만이 제기되지 않은 fwver이기 때문에 일단 0으로 넣어준다
   df_error= pd.merge(df_error, agg_df[['fwver','ratio']], on='fwver', how='left')
   df_error= df_error.fillna(0)
   ```

   ![image-20210207230834309](C:\Users\seo\AppData\Roaming\Typora\typora-user-images\image-20210207230834309.png)

   

2. 모델과 fwver, errtype에 따른 불만제기율

   1번 방법과 같은 방식으로 하되, errtype으로 데이터셋을 한번 더 분류하여 생성한 ratio를 파생변수로 사용하였다. 



3. 에러코드 별 불만제기율 

```python
# train_prob은 train_err_data와 train_problem_data를 합친 데이터 프레임임
errcode_lst= list(train_prob[train_prob['complain']>0]['errcode'].unique())
# 불만을 1번이상 제기한 행들을 추출
prob_o= train_prob[train_prob['complain']>0]

# complain을 제기한 errcode비율 산출
prob_o_group= prob_o.groupby(['errcode'], as_index=False)['complain'].sum()
prob_o_group['ratio']= prob_o_group['complain']/ prob_o_group['complain'].sum()

```

![image-20210207210016120](C:\Users\seo\AppData\Roaming\Typora\typora-user-images\image-20210207210016120.png)



위 코드를 통해 산출된 ratio는 고객이 불만을 제기할 경우, **에러코드에 따라 불만이 제기되는 비율**을 나타낸다. 이 비율을 x데이터의 파생변수로 추가하였다. 

```python
# train_err를 user_id와 errcode별로 그루핑한 후 time열의 개수를 집계하여 해당 유저가 어떤 에러코드에서 몇 번 에러를 겪었는지 계산
code_group= train_err.groupby(['user_id','errcode'], as_index=False)['time'].count()

# code_group의 errcode가 이전에 산출한 ratio의 errcode에 있는 것만 추출하여 errcode에 맞는 ratio 붙여줌
code_group= code_group[code_group['errcode'].isin(errcode_lst)]
code_group= pd.merge(code_group, prob_o_group[['errcode','ratio']], on='errcode', how='left')
code_group['multi']= code_group['time'] * code_group['ratio']
code_group= code_group.groupby('user_id', as_index=False)[['multi', 'time']].sum()
code_group['errcode_ratio']= code_group['multi']/ code_group['time']
code_group=code_group[['user_id','errcode_ratio']]
```

​	해당 코드를 실행함으로써 얻는 데이터프레임은 다음과 같다. 

![image-20210207211822691](C:\Users\seo\AppData\Roaming\Typora\typora-user-images\image-20210207211822691.png)

​	'errcode_ratio' 변수를 기존 x 데이터 프레임에 파생변수로 넣어준다. 



4. 시간 파생변수 생성

   train_err_data의 time에 대해 EDA한 결과, **시간에 따라 불만제기율이 다르게 나타나**는 것을 확인하였다. 따라서 시간과 관련한 파생변수 3가지를 생성하여 추가하였다. 

   비율 생성과 파생변수를 만들어 x에 추가하는 방식은 3번 코드와 유사한 방식으로 대상 변수만 시간변수로 바꾸어 실행하였다. 

* 요일 별 불만제기율 변수 'dow' 생성
* 일별 불만 제기율 변수 'day'생성
* 시간 별 불만 제기율 변수 'hour'생성



5. quality 파생변수 생성

quality EDA결과, **각 quality에서 나타나는 수치의 최대값과 중앙값이 불만을 제기한 유저와 불만을 제기하지 않은 유저에 차이**가 있는 것으로 나타났다. 따라서 각 quality 별 기록된 수치의 max값과 median값을 변수로 추가해주었다. 



6. errcode에서 connection error의 개수 집계

   **errcode에서 connection error와 관련된 에러가 많이 발생하고, 그에 따른 complain도 많이 발생하는** 것을 확인할 수 있었다. 따라서 각 유저 별 errcode에서 'connection'관련 에러가 발생한 개수를 집계하여 파생변수로 추가해주었다. 



작성자가 생성한 파생변수만을 합친 데이터셋은 다음과 같다. 

![image-20210207233142695](C:\Users\seo\AppData\Roaming\Typora\typora-user-images\image-20210207233142695.png)





### 모델학습

모델은 light GBM을 사용하였으며, 그에 대한 파라미터는 과적합은 최소로, 정확도는 최대가 되도록 조정하였다.  앞서 추가한 파생변수와 팀원들이 생성한 파생변수를 모두 추가한 데이터 셋을 x로,  y는 유저 별 불만수로 설정하여 모델을 학습하였다. 모델 학습 전, 데이터프레임 형식인 x와 y의 형식을 array로 바꾼 후 모델에 입력으로 넣어준다. 

```python
# Train
#-------------------------------------------------------------------------------------
# validation auc score를 확인하기 위해 정의
def f_pr_auc(probas_pred, y_true):
    labels=y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score=auc(r,p) 
    return "pr_auc", score, True
#-------------------------------------------------------------------------------------
models     = []
recalls    = []
precisions = []
auc_scores   = []
threshold = 0.5
# 파라미터 설정
params =      {
                'boosting_type' : 'dart',
                'objective'     : 'binary',
                'metric'        : 'auc',
                'seed': 101,
    'max_bin':400,
    'learning_rate':0.05,
    'num_iteration':300,
    'max_depth':-1,
    'num_leaves': 30,
    'min_data_in_leaf':10
    
                }
#-------------------------------------------------------------------------------------
# 5 Kfold cross validation
sk_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
for train_idx, val_idx in sk_fold.split(train_x, train_y):

    # split train, validation set
    X = train_x[train_idx]
    y = train_y[train_idx]
    valid_x = train_x[val_idx]
    valid_y = train_y[val_idx]

    d_train= lgb.Dataset(X, y)
    d_val  = lgb.Dataset(valid_x, valid_y)
    
    #run traning
    model = lgb.train(
                        params,
                        train_set       = d_train,
                        num_boost_round = 1000,
                        valid_sets      = [d_train,d_val],
                        feval           = f_pr_auc,
                        verbose_eval    = 20, 
                        early_stopping_rounds = 5
                       )
    
    # cal valid prediction
    valid_prob = model.predict(valid_x)
    valid_pred = np.where(valid_prob > threshold, 1, 0)
    
    # cal scores
    recall    = recall_score(    valid_y, valid_pred)
    precision = precision_score( valid_y, valid_pred)
    auc_score = roc_auc_score(   valid_y, valid_prob)

    # append scores
    models.append(model)
    recalls.append(recall)
    precisions.append(precision)
    auc_scores.append(auc_score)

    print('==========================================================')
```

 

모델 학습 결과 auc_scores의 평균은 **최대 84%**가 나왔다. 해당 모델로 **test set에 대한 불만제기율을 예측**한 결과, **최대 83.2%의 정확도**를 기록하여 전체등수 418등 중 48등으로 대회를 마무리하였다.  






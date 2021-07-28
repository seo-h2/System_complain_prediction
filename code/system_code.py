import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_user_id_max = 24999
train_user_id_min = 10000
train_user_number = 15000

# 모델 에러로그
train_err= pd.read_csv('C://users//seo//desktop//시스템분석//data//train_err_data.csv')
# 모델 문제 타입
train_prob= pd.read_csv('C://users//seo//desktop//시스템분석//data//train_problem_data.csv')
# 모델 품질 변화 기록
train_qual= pd.read_csv(".//data//train_quality_data.csv", thousands=',')

#================================================================사용할 데이터셋 생성:  에러로그와 불만데이터를 합친 데이터 셋

## prob데이터 셋에 어떤 모델에 대한 complain인지 used_model열 추가
## train_prob의 user_id가 불만을 제기한 time이전에 사용했던 model(들) 중 가장 최근에 사용했던 모델을 model_nm열에 추가
## model_nm리스트가 0인 경우가 있음-> 불만 제기 이전에 사용했던 모델이 train_err 데이터 셋엔 없을 수도 있다고 추정/ 이 경우엔 이후에 사용한 모델 중 가장 최근에 사용한 모델을 가져옴
## 마찬가지로 errtype에 대해서도 적용(불만제기 이전에 가장 최근에 발생한 errtype)
## 마찬가지로 fwver에 대해서도 적용(model_nm은 삭제)

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

#======================================================EDA
## 1. 모델과 fwver에 따른 complain수와 error 발생 수 집계

err_type=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]

# 에러타입 에러 수 세는 함수
def errcnt(df,r,c):
    value= df.values
    mat= np.zeros((r,c))

    for row,col in value:
        mat[row,col-1] +=1
    return mat
# 에러타입에 따른 complain 수 세는 함수
def compcnt(df,r,c):
    value= df.values
    mat= np.zeros((r,c))

    for row,col,val in value:
        mat[row,col-1] += val
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
model_fw_error.to_csv('.//model_fw_error.csv')

# 모델과 fwver에 따른 complain 수 집계
# complain이 존재하는 행만 있는 데이터프레임 만들기
train_err2= train_err1[train_err1['complain']!=0]


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
model_fw_comp.to_csv(".//model_fw_comp.csv",index=False )

# model에 따른 fwver별 complain수와 error발생 합계 시각화
plt.figure(figsize=(10,30))
plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=1)

n=1
for i in range(9):
    plt.subplot(9,2,n)
    plt.xticks(rotation = - 45 )
    plot= sns.barplot(model_fw_comp[model_fw_comp['model_nm']=='model_%d'%i]['fwver'], model_fw_comp[model_fw_comp['model_nm']=='model_%d'%i]['comp_sum'])
    plot.set_title('the number of complain in model%d'%i)
    n=n+1

# 특정 fwver에서 어떤 에러타입에서 에러와 불만이 많이 집계 되는지 볼 수 있는 그래프 생성 함수
def fw_error_graph(fwver):
    plt.figure(figsize=(20,10))
    plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=1)

    df= pd.melt(model_fw_error[model_fw_error['fwver']==fwver], id_vars='fwver', value_vars=err_type)
    df2= pd.melt(model_fw_comp[model_fw_comp['fwver']==fwver], id_vars='fwver', value_vars=err_type)

    plt.subplot(121)
    plot1= sns.barplot(df['variable'], df['value'])
    plt.xticks(rotation = - 45,fontsize=8 )
    plot1.set_title('the number of error in %s'%fwver)

    plt.subplot(122)
    plot2= sns.barplot(df2['variable'], df2['value'])
    plt.xticks(rotation = - 45,fontsize=8 )
    plot2.set_title('the number of complain in %s'%fwver)

    return plot1, plot2

# fwver에 따른 complain과 error발생 수를 모델별로 비교해본 결과, 대체로 error가 많이 발생하는 fwver일수록 complain수도 많았다.
# model과 fwver에 따른 error type별 complain수와 error 발생 수의 경우, 특정 패턴이 발견되지는 않았다.
#-----------------------------------------------------------------------------------
## 2. quality EDA
# quality data에서 불만을 제기했다고 추정되는 사람과 제기하지 않은 사람의 품질 데이터의 분포도를 보고, quality값의 max와 median값 비교해보고자 함.

# 불만제기자의 quality와 불만제기하지 않은 자의 quality데이터 셋 생성 분포 비교 그래프 생성 함수
def qual_EDA(qual_nm):
    plt.figure(figsize=(200,100))
    plt.subplot(2,1,1)

    # time과 user id 기준으로 quality_9의 합계 집계
    yes_qual= yes.groupby(['user_id','time'], as_index=False)[qual_nm].sum()

    # 12배 되었을 quality_9값들을 12로 나눠줌
    yes_qual[qual_nm]= yes_qual[qual_nm]/12

    yes_qual['var']= yes_qual[qual_nm].astype('str')

    # quality_9 값으로 그루핑해서 count집계해보기
    yes_qual_cnt= pd.DataFrame(yes_qual.groupby(qual_nm, as_index=False)['time'].count())

    sns.barplot(yes_qual_cnt[qual_nm], yes_qual_cnt['time'])
    plt.ylim(0,1000)

    plt.subplot(2,1,2)

    # time과 user id 기준으로 quality_9의 합계 집계
    no_qual= no.groupby(['user_id','time'], as_index=False)[qual_nm].sum()

    # 12배 되었을 quality_9값들을 12로 나눠줌
    no_qual[qual_nm]= no_qual[qual_nm]/12

    no_qual['var']= no_qual[qual_nm].astype('str')

    # quality_9 값으로 그루핑해서 count집계해보기
    no_qual_cnt= pd.DataFrame(no_qual.groupby(qual_nm, as_index=False)['time'].count())

    sns.barplot(no_qual_cnt[qual_nm], no_qual_cnt['time'])
    plt.ylim(0,1000)


    return yes_qual_cnt, no_qual_cnt

# 예시로 quality_9값 사용
yes_qual, no_qual= qual_EDA('quality_9')
print('max: \n',yes_qual.max())
print('med: \n',yes_qual.median())
print(no_qual.max())
print('\n',no_qual.median())

# 모든 quality변수에 대해 eda한 결과, 불만제기한 사람과 제기하지 않은 사람의 quality 값의 분포는 비슷한양상을 보임.
# 양 집단의 max와 median값은 차이를 보이기에 해당 값에 대한 파생변수를 생성해 넣어줌.

#==========================================================
# 파생변수 생성

## 1. model_nm과 fwver에 따른 에러 개수와 불만 수의 비율
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

# df_error에 fwver 붙이기- fwver을 두개 이상 사용했을 경우 에러가 더 많이 집계된 fwver으로 사용
# train_err를 user_id와 fwver로 그루핑, 개수집계-> 한 유저 아이디에 하위그룹인 fwver이 2개 이상 있을 때 개수가 가장 큰 fwver선택
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
err_group= err_group[['user_id','fwver']]

# df_error의 user_id에 맞게 fwver 붙여주기
df_error= pd.merge(df_error,err_group[['user_id','fwver']], on='user_id', how='inner')
df_error= pd.merge(df_error, agg_df[['fwver','ratio']], on='fwver', how='left')
df_error= df_error.fillna(0)
df_error.to_csv(".//df_error.csv")

#-------------------------------------
## 2. 모델과 fwver, errtype에 따른 불만제기율 파생변수 생성

# ratio테이블 만들기
mfe= pd.read_csv('.//model_fw_error.csv')
mfc= pd.read_csv('.//model_fw_comp.csv')

errtype=[]
for i in range(42):
    errtype.append(str(i))
fwver= list(mfc['fwver'].values)

for i in errtype:
    mfe[i]= mfe[i]/mfe['err_sum']
    mfc[i]= mfc[i]/mfc['comp_sum']
mfc= pd.merge(mfc,mfe[['fwver','model_nm']],on='fwver', how='outer')
mfc= mfc.fillna(0)

# mfc와 mfe의 인덱스를 fwver을 인덱스로 함
mfc=mfc.set_index('fwver')
mfe=mfe.set_index('fwver')

# mfc의 불만제기 비율 / mfe의 에러타입비율을 담을 데이터 프레임 생성
# 형식은 mfc와 같게
err_ratio= mfc.copy()

# err_ratio에서 알맞은 fwver행의 errtype열에 값 넣어주기
for fv in fwver:
    for et in errtype:
        err_ratio.at[fv,et]= mfc.loc[fv][et] / mfe.loc[fv][et]
err_ratio.drop('model_nm_x', axis=1, inplace=True)
err_ratio.drop('comp_sum', axis=1, inplace=True)
err_ratio.rename({'model_nm_y':'model_nm'}, axis=1, inplace=True)
err_ratio= err_ratio.fillna(0)

err_ratio.to_csv('.//err_ratio.csv')

#각 user_id가 fwver에서 겪은 에러타입에 ratio적용
err_group= pd.DataFrame(train_err.groupby(['user_id','fwver','errtype'], as_index=False)['time'].count())
# 각행에 fwver과 errtype에 맞는 ratio 붙여주기
err_group['ratio']=0.0
for i in range(len(err_group)):
    err_group.at[i,'ratio']= err_ratio[err_ratio['fwver']==err_group['fwver'][i]][str(err_group['errtype'][i]-1)]
# time과 ratio를 곱한 열 'multi'생성
err_group['multi']=0.0
err_group['multi']= err_group['time'] * err_group['ratio']

# user_id, fwver 기준으로 그루핑하면서 time과 multi는 합으로 집계
paseng2= pd.DataFrame(err_group.groupby(['user_id'], as_index=False)[['time','multi']].sum())
# multi를 time으로 나눠준 값을 각 user_id의 최종 파생변수2 값으로 사용

paseng2['mfe']= paseng2['multi'] / paseng2['time']
paseng2= paseng2[['user_id','mfe']]
paseng2.to_csv('.//mfe_df.csv', index=False)

#----------------------------------------------
## 3. 시간변수- 요일변수'dow' 생성

import datetime

# train_prob데이터로 요일별 불만제기율
# train_prob에 요일 열 붙이기

# time을 str형식으로 바꿈
train_prob['time']=train_prob['time'].astype('str')

days=['mon','tue','wed','thu','fri','sat','sun']
train_prob['dayofweek']= list(map(lambda x: days[datetime.date(int(train_prob['time'][x][0:4]),int(train_prob['time'][x][4:6]), int(train_prob['time'][x][6:8])).weekday()], range(len(train_prob))))

dayofweek_ratio= pd.DataFrame(train_prob.groupby('dayofweek', as_index=False)['time'].count())

# 불만제기 요일 비율 산출
dayofweek_ratio['ratio']= dayofweek_ratio['time']/dayofweek_ratio['time'].sum()
dayofweek_ratio.to_csv(".//dayofweek_ratio.csv", index=False)


# train_err에 요일변수 생성
# time을 str형식으로 바꿈
days=['mon','tue','wed','thu','fri','sat','sun']
train_err['time']=train_err['time'].astype('str')
train_err['dayofweek']= list(map(lambda x: days[datetime.date(int(train_err['time'][x][0:4]),int(train_err['time'][x][4:6]), int(train_err['time'][x][6:8])).weekday()], range(len(train_err))))

# user_id와 요일별 그루핑, time 개수 집계
day_group= train_err.groupby(['user_id','dayofweek'], as_index=False)['time'].count()

# daygroup의 요일에 맞게 ratio붙여주기
day_group= pd.merge(day_group, dayofweek_ratio[['dayofweek','ratio']], on='dayofweek', how='left')
day_group['multi']= day_group['time']*day_group['ratio']
day_group= day_group.groupby('user_id', as_index=False)[['time','multi']].sum()
day_group['dow']= day_group['multi']/day_group['time']

day_group= day_group[['user_id','dow']]
day_group.to_csv(".//dow.csv", index=False)

#-----------------------------------------------
## 4. 시간변수- 일변수'day'생성

# train_prob에서 일에 따른 불만제기율 데이터셋 생성
train_prob['day']= list(map(lambda x: int(train_prob['time'][x][6:8]), range(len(train_prob))))

# day로 그루핑하고 time의 개수 집계
day_ratio= pd.DataFrame(train_prob.groupby('day', as_index=False)['time'].count())
day_ratio['ratio']= day_ratio['time']/ day_ratio['time'].sum()
day_ratio.to_csv('.//day_ratio.csv', index= False)

# train_err에 'day'파생변수 붙이기
train_err['day']= train_err['time'].str.slice(start=6, stop=8)

# day의 타입을 str에서 int로 바꾼다.
train_err['day']= train_err['day'].astype('int')

# user_id와 day로 그루핑 후 time 개수 집계
day_group= train_err.groupby(['user_id','day'], as_index=False)['time'].count()

# daygroup의 day에 맞게 ratio 붙여주기
day_group= pd.merge(day_group, day_ratio[['day','ratio']], on='day', how='left')

day_group['multi']= day_group['time'] * day_group['ratio']
day_group= day_group.groupby(['user_id'], as_index=False)[['time','multi']].sum()
day_group['day']= day_group['multi']/day_group['time']

day_group.to_csv('.//day_group.csv', index=False)

#------------------------------------------------
# 5. 시간변수- 시간변수'hour'생성

## train_prob에서 hour ratio구하기
train_prob['hour']= train_prob['time'].str.slice(start=8, stop=10)
hour_ratio= train_prob.groupby('hour', as_index=False)['time'].count()
hour_ratio['ratio']= hour_ratio['time']/ hour_ratio['time'].sum()
hour_ratio.to_csv('.//hour_ratio.csv', index=False)

## train_err에서 hour변수 추가
train_err['hour']= train_err['time'].str.slice(start=8, stop=10)

# user_id와 hour로 그루핑 후 time 개수 집계
hour_group= train_err.groupby(['user_id','hour'], as_index=False)['time'].count()
hour_group['hour']= hour_group['hour'].astype('int')

#EDA결과 고객은 에러를 겪은 뒤 평균적으로 약 2시간 뒤 불만을 제기한다. 따라서 불만제기율은 에러발생 후 2시간 이후 시점의 비율을 사용.
hour_group['hour']= hour_group['hour']+2
hour_group.at[hour_group['hour']>24,'hour']=hour_group[hour_group['hour']>24]-24

# daygroup의 day에 맞게 ratio 붙여주기
hour_group= pd.merge(hour_group, hour_ratio[['hour','ratio']], on='hour', how='left')
hour_group['multi']= hour_group['time'] * hour_group['ratio']
hour_group= hour_group.groupby(['user_id'], as_index=False)[['time','multi']].sum()
hour_group['hour']= hour_group['multi']/hour_group['time']

hour_group.to_csv('.//hour_group.csv', index=False)

#--------------------------------------------------

# 6. err_code에서 connection관련 에러 수 파생변수 생성
df= pd.read_csv('.//data//train_err_data.csv')
connect_df=df[df['errcode'].str.contains('connection', na=False, case=False)]
connect_df= connect_df[['user_id','errcode']]
connect_df.reset_index(drop=True, inplace=True)
dummy= pd.get_dummies(connect_df['errcode'])
connect_df= pd.concat([connect_df, dummy], axis=1)
connect_df.drop('errcode',axis=1, inplace=True)
connect= connect_df.groupby('user_id', as_index=False).sum()
connect.to_csv('.//connect.csv', index=False)

#--------------------------------------------------
# 7. 2에서 5까지의 파생변수를 1의 df_error데이터 셋에 붙이기

# 생성한 데이터프레임 불러오기
mfe= pd.read_csv(".//mfe_df.csv")
dow= pd.read_csv(".//dow.csv")
day_group= pd.read_csv(".//day_group.csv")
hour_group= pd.read_csv(".//hour_group.csv")
connect= pd.read_csv('.//connect.csv')

### mfe 파생변수 추가
df_error=pd.merge(df_error, mfe, on='user_id', how='inner')

### dow파생변수 추가
df_error= pd.merge(df_error, dow, on='user_id', how='inner')

### day파생변수
df_error= pd.merge(df_error, day_group[['user_id','day']], on='user_id', how='left')

### hour변수 넣기
df_error= pd.merge(df_error, hour_group[['user_id','hour']], on='user_id', how='left')

### connect변수 넣기
df_error= pd.merge(df_error, connect, on='user_id', how='left')

df_error.to_csv('.//sh_train2.csv', index=False)
#=================================================
# 팀원들이 만든 파생변수 모두 합친 데이터셋으로 모델학습
df_error= pd.read_csv('.//train.csv')
error= np.array(df_error)

error.shape

problem = np.zeros(15000)
# error와 동일한 방법으로 person_idx - 10000 위치에
# person_idx의 problem이 한 번이라도 발생했다면 1
# 없다면 0
problem[train_prob.user_id.unique()-10000] = 1
problem.shape

problem

# 변수 이름 변경
# error  -> train_x
# problem-> train_y

train_x = error
train_y = problem
del error, problem
print(train_x.shape)
print(train_y.shape)

from sklearn.model_selection import StratifiedKFold
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')

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

print(np.mean(auc_scores))
#==========================
# test에 동일하게 파생변수 생성한 후 해당 모델로 predict

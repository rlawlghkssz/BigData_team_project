import pandas as pd
import pingouin
import seaborn as sns
import matplotlib.pyplot as plt
from  scipy import stats
import scikit_posthocs as sp
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
from IPython.display import set_matplotlib_formats
matplotlib.use('TkAgg')

# 전처리
peo = pd.read_csv("merge_sort.csv", encoding="utf-8")
droppeo = peo.dropna()
droppeo.loc[:,"날짜"] = pd.to_datetime(droppeo["날짜"], format="%Y%m%d")
droppeo.loc[:,"year"] = droppeo["year"].astype(int)
droppeo.loc[:,"month"] = droppeo["month"].astype(int)
droppeo2020 = droppeo[droppeo["year"] ==2020]
droppeo2019 = droppeo[droppeo["year"] ==2019]
droppeo2018 = droppeo[droppeo["year"] ==2018]
droppeo2020 = droppeo2020[droppeo2020["날짜"] >= "2020-02-15"]
droppeo2020 = droppeo2020[droppeo2020["날짜"] <= "2020-11-25"]
droppeo2020a = droppeo2020[droppeo2020['날짜'] <= "2020-10-14"]
droppeo2020b = droppeo2020[droppeo2020['날짜'] >= "2020-10-28"]
droppeo2019 = droppeo2019[droppeo2019["날짜"] >= "2019-02-15"]
droppeo2019 = droppeo2019[droppeo2019["날짜"] <= "2019-11-25"]
droppeo2018 = droppeo2018[droppeo2018["날짜"] >= "2018-02-15"]
droppeo2018 = droppeo2018[droppeo2018["날짜"] <= "2018-11-25"]
droppeo2018a = droppeo2018[droppeo2018['날짜'] <= "2018-10-14"]
droppeo2018b = droppeo2018[droppeo2018['날짜'] >= "2018-10-28"]
droppeo = pd.concat([droppeo2018a, droppeo2018b, droppeo2019, droppeo2020a, droppeo2020b])
peosum = droppeo.groupby(['year', "month"], as_index=False).sum()
plt.rc('font', family="Malgun Gothic")
%matplotlib inline

"""상관분석"""
# 서울시 전체
droppeo2020_sum = droppeo2020.groupby(['year', "month", 'day'], as_index=False).sum()
droppeo2020_sum = droppeo2020_sum[droppeo2020_sum['생활인구'] > 0]
scale = MinMaxScaler()
droppeo2020_sum["생활인구st"] = scale.fit_transform(np.array(droppeo2020_sum["생활인구"]).reshape(-1, 1))
droppeo2020_sum["확진자st"] = scale.fit_transform(np.array(droppeo2020_sum["확진자 수"]).reshape(-1, 1))
plt.rc('font', family="Malgun Gothic")
sns.regplot(data=droppeo2020_sum, x="생활인구st", y="확진자st")
plt.xlabel("생활인구")
plt.ylabel("확진자 수")
plt.title("생활인구-확진자수(서울시)")
print("서울시 전체 상관계수 : ", spearmanr(droppeo2020_sum["생활인구st"], droppeo2020_sum["확진자st"]))
plt.show()


# 구별 상관관계(Top5)
for i in res:
    droppeo2020_area = droppeo2020[droppeo2020['지역']==i]
    droppeo2020_area = droppeo2020_area[droppeo2020_area['생활인구'] > 0]
    droppeo2020_area["생활인구st"] = scale.fit_transform(np.array(droppeo2020_area["생활인구"]).reshape(-1, 1))
    droppeo2020_area["확진자st"] = scale.fit_transform(np.array(droppeo2020_area["확진자 수"]).reshape(-1, 1))
    sns.regplot(data=droppeo2020_area, x="생활인구st", y="확진자st")
    plt.xlabel("생활인구")
    plt.ylabel("확진자 수")
    plt.title("생활인구-확진자수 "+ i)
    print("상관계수 : "+i, spearmanr(droppeo2020_area["생활인구st"], droppeo2020_area["확진자st"]))
    # plt.savefig(filedir.format(i + " 산점도"), format='png')
    plt.show()
    accumulates_day = droppeo2020_area.set_index("날짜")
    accumulates_day = accumulates_day[["생활인구st", "확진자st"]]
    accumulates_day["생활인구st"].plot(label="생활인구st", figsize=(15,4))
    accumulates_day["확진자st"].plot(label="확진자st", figsize=(15,4))
    plt.title("생활인구-확진자수 일별 추이 "+ i)
    plt.show()
    print("______________________________________________________________________")

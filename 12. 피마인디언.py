import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#1. 데이터 읽기
df = pd.read_csv("./data/pima-indians-diabetes.csv",
                names = ['pregnant','plasma', 'pressure','thickness',
                        'insulin','BMI','pedigree','age','class'])

# [문제]: groupby를 사용하여 다음의 결과를 보여주는 데이터 프레임을 생성하시오

# pregnant  count   sum   mean
# ============================
#     0     111     38    0.34232

df_pregnant_group = df[['pregnant','class']].groupby('pregnant', as_index = False)
df_info = pd.DataFrame()
df_info['pregnant'] = df_pregnant_group.count()['pregnant']
df_info['count'] = df_pregnant_group.count()['class']
df_info['sum'] = df_pregnant_group.count()['class']
df_info['mean'] = df_pregnant_group.count()['class']

sns.heatmap(df.corr(), linewidths = 0.1, vmax = 0.5, cmap = plt.cm.gist_heat,
            linecolor = 'white', annot = True) # vmax : 색상의 밝기를 조절하는 인자. cmap : 미리 정해진 맷플롯립 색상의 설정 값을 불러옴
plt.show()
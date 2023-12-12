import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data("datasets/clean_data.csv")

st.title('Streamlit homework')

st.header('Графики распределения признаков')
fig = px.histogram(df, x='AGE', title=f'Возраст клиентов')
st.plotly_chart(fig)
st.text("Основной возраст от 23 до 58 лет")

income_bins = [0, 10000, 20000, 30000, 50000, 70000, df['PERSONAL_INCOME'].max()]
income_labels = ['До 10.000', '10.000-20.000','20.000-30.0000', '30.000 - 50.000', '50.000 - 70.000', 'Выше 70.000']
df['Income_Groups'] = pd.cut(df['PERSONAL_INCOME'], bins=income_bins, labels=income_labels)
fig = px.histogram(df, x='Income_Groups', title='Распределение заработка клиентов')
st.plotly_chart(fig)
st.text("Основной заработок до 20.000")
df.drop('Income_Groups', axis=1, inplace=True)

cat_columns = ['CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']

fig = px.histogram(df, x='GENDER', title='Распределение по гендеру')
st.plotly_chart(fig)
st.text("Число мужчин превышает почти в два раза")

for column in cat_columns:
    fig = px.pie(df, names=df[column].value_counts().index, values=df[column].value_counts().values,
                 title=f'Pie Chart for {column}')
    st.plotly_chart(fig)

st.markdown("Из графиков по категориальным переменным можно сделать следующие выводы:  \n \
    **Количество детей** - от 0 до 2  \n \
    **Количество иждивенцев** - чаще всего 0  \n \
    **Количество работающих клиентов** - большая часть, 91%  \n\
    **Количество кредитов** - больше 1 только у 25% клиентов  \n \
    **Количество закрытых кредитов** - больше половины закрыли 1 кредит  \n \
")

st.header('Распределение целевой переменной')

fig = px.histogram(df, x='TARGET', title='Распределение целевой переменной')
st.plotly_chart(fig)
st.text("Присутствует дисбаланс классов")

st.header('Матрица коррелляции')
correlation_matrix = df.corr()
fig_corr = px.imshow(correlation_matrix, labels=dict(color="Correlation"), title='Correlation Matrix', text_auto=True, color_continuous_scale = 'RdYlBu')
st.plotly_chart(fig_corr, width=1000, height=800)

st.header('Графики зависимости целевой переменной и признаков')

fig = px.box(df, x='TARGET', y='AGE', title='Box plot: Распределение возраста по целевой переменной')
st.plotly_chart(fig)

fig = px.box(df, x='TARGET', y='PERSONAL_INCOME', title='Box plot: Распределение дохода по целевой переменной')
st.plotly_chart(fig)

df_stat = df.drop(columns = ['AGREEMENT_RK', 'TARGET', 'GENDER', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL'])

numeric_stats = df_stat.describe()
mean_values = df_stat.mean()
min_values = df_stat.min()
max_values = df_stat.max()
median_values = df_stat.median()

st.header('Числовые характеристики числовых столбцов')
st.write('Numeric Statistics:')
st.write(numeric_stats)

st.write('Mean Values:')
st.write(mean_values)

st.write('Min Values:')
st.write(min_values)

st.write('Max Values:')
st.write(max_values)

st.write('Median Values:')
st.write(median_values)
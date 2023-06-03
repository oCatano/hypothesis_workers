import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import streamlit as st


def test_criterion(sample_1: pd.DataFrame, sample_2: pd.DataFrame) -> tuple:
    """

    :param sample_1: First sample
    :param sample_2: Second sample
    :return: Type of test, stats and p_value from test
    """

    # Проверка нормальности распределений
    normality_check_1 = stats.shapiro(sample_1['work_days'])
    normality_check_2 = stats.shapiro(sample_2['work_days'])

    # Уровень значимости для проверки нормальности
    alpha = 0.05

    # Если распределения нормальные, тогда используем t-тест Стьюдента
    # иначе тест Манна-Уитни
    if normality_check_1.pvalue > alpha and normality_check_2.pvalue > alpha:
        # Выполнение t-теста Стьюдента
        statistic, p_value = stats.ttest_ind(sample_1['work_days'], sample_2['work_days'], alternative='greater')
        test_type = "t-test"
    else:
        # Выполнение теста Манна-Уитни
        statistic, p_value = stats.mannwhitneyu(sample_1['work_days'], sample_2['work_days'], alternative='greater')
        test_type = "Mann-Whitney test"

    return test_type, statistic, p_value


def first_hypothesis(data: pd.DataFrame, work_days, alpha: float = 0.05):
    """

    :param alpha: min p_value
    :param data: data with work_days, age, sex
    :return: p_value
    """
    st.title("Задача №1")
    st.subheader("Мужчины пропускают в течение года более 2 рабочих дней по болезни значимо чаще женщин")

    # Отбираем информацию только о работниках, которые пропускают больше 2‑х дней
    data_first = data[data.work_days > work_days]

    # Разделяем информацию для мужчин и женщин
    male_data = data_first[data_first.sex == "Men"]
    female_data = data_first[data_first.sex == "Women"]

    test_type, statistic, p_value = test_criterion(male_data, female_data)

    trace1 = go.Box(y=male_data['work_days'], name='Male')
    trace2 = go.Box(y=female_data['work_days'], name='Female')
    drawbl = [trace1, trace2]
    layout = go.Layout(title='Распределение количества пропущенных рабочих дней',
                       yaxis=dict(title='Value', tickformat=".2f"),
                       xaxis=dict(title='SEX', tickformat=".3f")
                       )

    fig = go.Figure(data=drawbl, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    # Визуализация полигона распределения
    fig_poly = go.Figure()
    fig_poly.add_trace(
        go.Scatter(x=sorted(male_data['work_days']), y=[i / len(male_data['work_days']) for i in range(1, len(male_data['work_days']) + 1)], mode='lines',
                   name='Male'))
    fig_poly.add_trace(
        go.Scatter(x=sorted(female_data['work_days']), y=[i / len(female_data['work_days']) for i in range(1, len(female_data['work_days']) + 1)], mode='lines',
                   name='Female'))

    fig_poly.update_layout(title='Полигон распределения', xaxis_title='Values', yaxis_title='Cumulative Probability')
    st.plotly_chart(fig_poly)

    # Определение отвергаемости нулевой гипотезы
    if p_value <= alpha:
        hypothesis_status = "Есть статистическая разница между средними значениями двух выборок. \
                             То есть мужчины пропускают значимо чаще рабочих дней по болезни, чем женщины"
    else:
        hypothesis_status = "Нельзя отклонить нулевую гипотезу в пользу альтернативной. \
                             То есть мужчины НЕ пропускают значимо чаще рабочих дней по болезни, чем женщины"

    col1, col2 = st.columns(2)
    col1.metric(test_type, round(statistic, 3))
    col2.metric("P_value", round(p_value, 5))
    st.info(f'Вывод: {hypothesis_status}')


def second_hypothesis(data: pd.DataFrame, age, work_days, alpha: float = 0.05):
    """

    :param data: data with work_days, age, sex
    :param alpha: min p_value
    :return:
    """

    st.title("Задача №2")
    st.subheader("Работники старше 35 лет пропускают в течение года более 2 рабочих дней по болезни значимо чаще \
                 своих более молодых коллег")

    # Берем только данные, где количество пропусков более двух
    data_second = data.drop(['sex'], axis=1)
    data_second = data_second[data_second.work_days > work_days]
    data_second.age = data_second.age.apply(lambda x: "Older_35" if x > age-1 else "Younger_35")

    # Делим данные на две выборки - старше и младше 35 лет
    older_data = data_second[data_second.age == "Older_35"]
    younger_data = data_second[data_second.age == "Younger_35"]

    test_type, statistic, p_value = test_criterion(older_data, younger_data)

    trace1 = go.Box(y=younger_data['work_days'], name='Younger')
    trace2 = go.Box(y=older_data['work_days'], name='Older')
    drawbl = [trace1, trace2]
    layout = go.Layout(title='Распределение количества пропущенных рабочих дней',
                       yaxis=dict(title='Value', tickformat=".3f"),
                       xaxis=dict(title='X', tickformat=".3f")
                       )
    fig = go.Figure(data=drawbl, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    # Визуализация полигона распределения
    fig_poly = go.Figure()
    fig_poly.add_trace(
        go.Scatter(x=sorted(younger_data['work_days']),
                   y=[i / len(younger_data['work_days']) for i in range(1, len(younger_data['work_days']) + 1)],
                   mode='lines',
                   name='Younger'))
    fig_poly.add_trace(
        go.Scatter(x=sorted(older_data['work_days']),
                   y=[i / len(older_data['work_days']) for i in range(1, len(older_data['work_days']) + 1)],
                   mode='lines',
                   name='Older'))

    fig_poly.update_layout(title='Полигон распределения', xaxis_title='Values', yaxis_title='Cumulative Probability')
    st.plotly_chart(fig_poly)



    # Определение отвергаемости нулевой гипотезы
    if p_value <= alpha:
        hypothesis_status = f"Есть статистическая разница между средними значениями двух выборок.  \
               Работники старше {age} лет пропускают значимо чаще рабочих дней по болезни, чем более молодые коллеги"
    else:
        hypothesis_status = f"Нельзя отклонить нулевую гипотезу в пользу альтернативной.  \
               Работники старше {age} лет НЕ пропускают значимо чаще рабочих дней по болезни, чем более молодые коллеги"

    col1, col2 = st.columns(2)
    col1.metric(test_type, round(statistic, 3))
    col2.metric("P_value", round(p_value, 5))
    st.info(f'Вывод: {hypothesis_status}')


st.title("Criterions test")
st.write("Этот dashboard будет доказывть/опровергать значимость различия средних значений выборок")

spectra = st.file_uploader("Загрузите csv файл", type={"csv"})

# # Отображение введенных значений
# st.write("Введенное значение 1:", age)
# st.write("Введенное значение 2:", work_days)

if spectra is not None:
    data = pd.read_csv(spectra)

    col1, col2 = st.columns(2)
    age = col1.number_input("Age", value=35)
    work_days = col2.number_input("work_days", value=2)

    data.columns = ('work_days', 'age', 'sex')
    data.sex = data.sex.apply(lambda x: "Men" if "М" in x or "M" in x else "Women")

    try:
        first_hypothesis(data, work_days)
    except:
        st.error('К сожалению, вы ввели неверные значения age или work_days')

    try:
        second_hypothesis(data, age, work_days)
    except:
        st.error('К сожалению, вы ввели неверные значения age или work_days')



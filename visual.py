import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_close_column_counts(dataframe, stock_name, figsize):
    # Инициализируем счетчики для NAN и заполненных значений
    nan_counts = pd.Series(dtype=int)
    filled_counts = pd.Series(dtype=int)
    first_filled_date = pd.Series(dtype=str)  # Добавляем серию для дат первого заполненного значения
    # Перебираем столбцы и считаем количество NAN и заполненных значений в столбцах "CLOSE"
    for column in dataframe.columns:
        if 'CLOSE' in column:
            nan_counts[column] = dataframe[column].isna().sum()
            filled_counts[column] = dataframe[column].count()

            # Находим индекс первого заполненного значения
            first_filled_index = dataframe[column].first_valid_index()
            if first_filled_index:
                first_filled_date[column] = first_filled_index.strftime('%Y-%m-%d')
            else:
                first_filled_date[column] = 'N/A'

    # Удаляем "_CLOSE" из меток столбцов
    labels = [col.replace('_CLOSE', '') for col in nan_counts.index]

    # Переименование меток в соответствии со словарем
    labels = [stock_name.get(label, label) for label in labels]

    # Создаем столбчатую горизонтальную диаграмму с опцией stacked=True
    plt.figure(figsize=figsize)
    y = range(len(nan_counts))
    bar_height = 0.4

    plt.barh(y, nan_counts, height=bar_height, align='center', label='NAN')
    plt.barh(y, filled_counts, height=bar_height, align='center', label='Заполненные', left=nan_counts)

    # Настроим подписи на оси y
    plt.yticks(y, nan_counts.index)

    # Настроим метки на оси x с использованием дат
    plt.yticks(y, labels)

   # Добавляем текстовую метку с датой первого заполненного значения
    for i, date in enumerate(first_filled_date):
        plt.text(filled_counts[i] + nan_counts[i], i, f'Данные акции начинаются: {date}', va='center', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Акции', fontsize=14)
    plt.xlabel('Количество', fontsize=14)
    plt.title('Количество NAN и заполненных значений в столбцах с акциями', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_corr_heatmap(df_stock, stock_name):
    # Выбираем все столбцы "CLOSE" через цикл
    close_columns = [col for col in df_stock.columns if 'CLOSE' in col]

    # Выбираем только нужные столбцы из DataFrame
    subset = df_stock[close_columns]
    subset.columns = subset.columns.str.replace('_CLOSE', '')
    subset = subset.rename(columns=stock_name)

    # Строим карту корреляции
    corr_matrix = subset.corr()

    # Создаем фигуру
    plt.figure(figsize=(15, 15))

    # Строим тепловую карту с отображением значений на главной диагонали
    heatmap = sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

    # Добавляем зеленые линии к ячейкам, где коэффициент корреляции по модулю больше 0.9 на одной стороне
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='lime', lw=3))
                x = j + 0.5
                y = i + 0.5
                plt.plot([x, x], [len(corr_matrix), i], color='lime', linewidth=1)
                plt.plot([0, j+1], [y, y], color='lime', linewidth=1)

    # Добавляем фиолетовые линии к ячейкам, где коэффициент корреляции по модулю меньше 0.05 на противоположной стороне
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) < 0.05:
                heatmap.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor='purple', lw=3))
                x = j + 0.5
                y = i + 0.5
                plt.plot([y, y], [j, len(corr_matrix)], color='purple', linewidth=1)
                plt.plot([0, i+1], [x, x], color='purple', linewidth=1)

    plt.title('Карта корреляции между столбцами c ценами. Зеленый > 0.9, фиолетовый < 0.05 по модулю')
    plt.show()
    return corr_matrix

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_stock_predictions(scaler_pred, y_train, y_test, predictions, name_stock, start_date_index=None, predictions_future=None):
    if start_date_index is None:
        start_date_index = y_train.index[0]
    else:
        start_date_index = pd.Timestamp(start_date_index)

    # Обратное преобразование прогнозов
    predictions_df = pd.DataFrame(scaler_pred.inverse_transform(predictions.reshape(-1, 1)), index=y_test.index,  columns=['Predicted'])

    plt.figure(figsize=(14, 6))
    plt.plot(y_train[start_date_index:], label='Train', color='green')
    plt.plot(y_test[start_date_index:], label='Test', color='blue')
    plt.plot(predictions_df[start_date_index:], label='Predictions', color='red')

    # Рисуем 'Future' только если переданы прогнозы будущих значений
    if predictions_future is not None:

        # Инверсия масштабирования прогнозируемых значений
        predictions_future = scaler_pred.inverse_transform(np.array(predictions_future).reshape(-1, 1))

        # Создание новых индексов для предсказаний на 21 день вперед
        new_index = pd.date_range(start=y_test.index[-1] + pd.Timedelta(days=1), periods=len(predictions_future), freq='D')

        # Использование полученных индексов для прогнозов
        predictions_future = pd.DataFrame(predictions_future, index=new_index, columns=['Future'])

        plt.plot(predictions_future, label='Future', color='magenta')

    plt.title(f'{name_stock}: Реальная цена и прогноз')
    plt.xlabel('Год')
    plt.ylabel('Цена')
    plt.legend()
    plt.show()
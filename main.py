import pandas as pd

# Указываем путь к вашему CSV файлу
csv_file_path = 'LKOH.csv'  # Замените на реальный путь к файлу

# Импорт данных из CSV файла в DataFrame с указанием разделителя и названий столбцов
columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']
df = pd.read_csv(csv_file_path, sep=';', header=0, names=columns)

# Объединяем 'DATE' и 'TIME' в один столбец с форматом datetime
df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')

# Устанавливаем 'DATETIME' в качестве индекса DataFrame
df.set_index('DATE', inplace=True)

# Удаляем 'TIME', они не нужны
df.drop(['TIME'], axis=1, inplace=True)

# Печать первых нескольких строк DataFrame для проверки
print(df.head())

import matplotlib.pyplot as plt

# Построение графика для столбца 'CLOSE'
plt.figure(figsize=(12, 6))
plt.plot(df['CLOSE'], label='CLOSE', color='b')

# Добавление заголовка и меток осей
plt.title('График CLOSE')
plt.xlabel('Дата')
plt.ylabel('CLOSE')
plt.legend()
plt.grid(True)
plt.show()

import plotly.graph_objs as go

# Создаем график
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['CLOSE'], mode='lines', name='CLOSE'))

# Настройка макета графика
fig.update_layout(
    title='График CLOSE с использованием Plotly',
    xaxis=dict(title='Дата'),
    yaxis=dict(title='CLOSE'),
    template='plotly_dark'
)

# Отображение графика
fig.show()
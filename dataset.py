
import os
import pandas as pd

def rename_csv_files(directory_path):
    # Проверяем, что указанный путь существует и является директорией
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print("Указанный путь не существует или не является директорией.")
        return

    # Перебираем все файлы в указанной директории
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            # Получаем новое имя файла без символа "_" и всего, что после него
            new_filename = filename.split("_")[0] + ".csv"

            # Формируем полные пути к исходному и новому файлам
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)

            # Переименовываем файл
            os.rename(old_filepath, new_filepath)
            print(f"Переименован файл: {filename} -> {new_filename}")



def create_dataset_from_csv_files(directory_path):
    # Проверяем, что указанный путь существует и является директорией
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print("Указанный путь не существует или не является директорией.")
        return None

    # Создадим пустой DataFrame, в который будем добавлять данные
    dataset = pd.DataFrame()

    # Перебираем все файлы в указанной директории
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            # Префикс для столбцов берем из названия файла (без расширения .csv)
            prefix = os.path.splitext(filename)[0]

            # Полный путь к CSV файлу
            file_path = os.path.join(directory_path, filename)

            # Считываем данные из CSV файла и добавляем их в DataFrame с префиксом
            columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']

            data = pd.read_csv(file_path, sep=',', header=0, names=columns)

            # Объединяем 'DATE' и 'TIME' в один столбец с форматом datetime
            data['DATE'] = pd.to_datetime(data['DATE'].astype(str), format='%Y%m%d')

            # Устанавливаем 'DATETIME' в качестве индекса DataFrame
            data.set_index('DATE', inplace=True)

            # Удаляем 'TIME', они не нужны
            data.drop(['TIME'], axis=1, inplace=True)

            # Добавляем столбец с волатильностью (HIGH - LOW)
            data["VOLATILITY"] = data['HIGH'] - data['LOW']

            # Удаляем столбцы 'OPEN', 'HIGH', 'LOW'
            data.drop(['OPEN', 'HIGH', 'LOW'], axis=1, inplace=True)

            data.columns = [f"{prefix}_{col}" for col in data.columns]

            # Добавляем данные в общий dataset
            dataset = pd.concat([dataset, data], axis=1)

    return dataset

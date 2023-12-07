from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, target_variable, test_size):
    headers = df.columns.tolist()

    df = df.dropna(subset=[target_variable])
    # Разделение данных на тренировочную и тестовую выборки
    train, test = train_test_split(df, test_size=test_size, shuffle=False)

    # Удаление строк с NaN в целевом признаке из тренировочной и тестовой выборок
    train = train.dropna(subset=[target_variable])
    test = test.dropna(subset=[target_variable])

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    # Создание копий тренировочной и тестовой выборок
    train_norm = train.copy()
    test_norm = test.copy()

    # Инициализация и обучение MinMaxScaler на тренировочных данных
    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train_norm)
    test_norm = scaler.transform(test_norm)  # Используем параметры обученного scaler для нормализации тестовых данных

    # Преобразование обратно в датафрейм с использованием сохраненных заголовков
    train_norm = pd.DataFrame(train_norm, columns=headers)
    test_norm = pd.DataFrame(test_norm, columns=headers)

    # Определение признаков (features) и целевой переменной
    features = df.columns.tolist()  # все столбцы, кроме целевой переменной, являются признаками
    features.remove(target_variable)

    X_train, y_train = train[features], train[target_variable]
    X_test, y_test = test[features], test[target_variable]

    # Используйте нормализованные данные
    X_train_norm, y_train_norm = train_norm[features], train_norm[target_variable]
    X_test_norm, y_test_norm = test_norm[features], test_norm[target_variable]

    # Инициализация и обучение MinMaxScaler для предсказания
    scaler_pred = MinMaxScaler()
    scaler_pred.fit(y_train.values.reshape(-1, 1))

    return X_train, y_train, X_test, y_test, X_train_norm, y_train_norm, X_test_norm, y_test_norm, scaler, scaler_pred


from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

def get_predictions_and_scores(model, X_test, y_test):
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)


    print(f"R2: {r2}")
    print(f"MSE: {mse}")

    return predictions
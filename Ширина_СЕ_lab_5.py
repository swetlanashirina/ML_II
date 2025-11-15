import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def classification_training(data):
    # Создаем бинарную целевую переменную: 0 если < 15, 1 если >= 15
    y = (data['mental_wellness_index_0_100'] >= 15).astype(int)

    # Выбираем все столбцы, кроме целевого ('mental_wellness_index_0_100') и идентификатора ('user_id')
    feature_columns = [col for col in data.columns if col not in ['mental_wellness_index_0_100', 'user_id']]
    X = data[feature_columns].copy()

    # Обработка категориальных признаков
    categorical_features = ['gender', 'occupation', 'work_mode']
    le_dict = {}
    for feature in categorical_features:
        if feature in X.columns:
            le = LabelEncoder()
            X.loc[:, feature] = le.fit_transform(X[feature].astype(str))
            le_dict[feature] = le 

    # Разделение датасета на тренировочную и тестовую выборку
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        shuffle=True,
        random_state=37 
    )

    # Обучение модели K-ближайших соседей (KNN)
    # KNN чувствителен к масштабу признаков, поэтому необходимо их масштабировать.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    knn_model = KNeighborsClassifier(
        n_neighbors=5  #Среднее значение, обеспечивающее баланс.
    )
    knn_model.fit(X_train_scaled, y_train) 

    # Обучение модели решающего дерева
    dt_model = DecisionTreeClassifier(
        max_depth=7, # Среднее значение, которе не позволяет дереву разрастаться бесконечно. Упрощает модель, уменьшает чувствительность к шуму.
        min_samples_split=10, #Минимальное количество образцов во внетреннем узле. Позволяет искать более общие закономерности, что позволяет иизбежать переобучения
        min_samples_leaf=5, # Среднее значение для минимального количества образцов в листе, не сильно жесткое условие, компромис между переобучением и разнообразием
        random_state=37 
    )
    dt_model.fit(X_train_raw, y_train)

    # Предсказания и оценка для KNN
    y_pred_knn = knn_model.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    knn_f1 = f1_score(y_test, y_pred_knn)

    # Предсказания и оценка для решающего дерева
    y_pred_dt = dt_model.predict(X_test_raw) 
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    dt_f1 = f1_score(y_test, y_pred_dt)

    # Вывод
    print(f"KNN: {knn_accuracy:.4f}; {knn_f1:.4f}")
    print(f"DT: {dt_accuracy:.4f}; {dt_f1:.4f}")

    # Построение дерева
    plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
    plt.figure(figsize=(30, 15))
    plt.show() 

#data = pd.read_csv("DB_3_cleaned.csv") 
#classification_training(data)

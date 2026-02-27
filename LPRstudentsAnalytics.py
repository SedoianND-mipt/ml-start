"""
Проект: Влияние образа жизни на успеваемость студентов
Автор: Надежда Седоян ЛФИ МФТИ Б02-501
Описание: Ищем корреляцию между сном, кофе, стрессом и оценками
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Данные (замени на свои!)
# sleep, study, coffee, stress, grade
data = {
    'sleep': [6, 5, 7, 8, 4, 6, 7, 5, 6, 8, 4, 5, 7, 6, 5],
    'study': [5, 4, 3, 2, 6, 5, 4, 5, 3, 2, 7, 4, 3, 5, 6],
    'coffee': [2, 3, 1, 0, 4, 2, 1, 3, 2, 0, 4, 3, 1, 2, 3],
    'stress': [7, 8, 5, 3, 9, 6, 4, 7, 5, 2, 9, 8, 4, 5, 7],
    'grade': [85, 78, 92, 95, 70, 82, 88, 75, 84, 94, 65, 72, 90, 86, 74]
}

df = pd.DataFrame(data)

print("=" * 50)
print(" АНАЛИЗ ДАННЫХ СТУДЕНТОВ ЛФИ")
print("=" * 50)

# 1. Базовая статистика
print("\nБазовая статистика:")
print(df.describe())

# 2. Корреляция с оценками
print("\nКорреляция факторов с оценками:")
correlations = df.corr()['grade'].sort_values(ascending=False)
for factor, corr in correlations.items():
    if factor != 'grade':
        print(f"{factor}: {corr:.2f}")

# 3. Модель линейной регрессии
X = df[['sleep', 'study', 'coffee', 'stress']]
y = df['grade']

model = LinearRegression()
model.fit(X, y)

# Предсказания
predictions = model.predict(X)

print("\nМодель линейной регрессии:")
print(f"Коэффициент детерминации R²: {r2_score(y, predictions):.2f}")
print("\nКоэффициенты регрессии:")
for feature, coef in zip(['sleep', 'study', 'coffee', 'stress'], model.coef_):
    effect = "положительное" if coef > 0 else "отрицательное"
    print(f"  {feature}: {coef:.2f} ({effect} влияние)")

# 4. Визуализация
plt.figure(figsize=(12, 4))

# График 1: Сон vs Оценки
plt.subplot(1, 3, 1)
plt.scatter(df['sleep'], df['grade'], color='blue', alpha=0.6)
plt.xlabel('Часов сна')
plt.ylabel('Оценка')
plt.title('Зависимость оценок от сна')
plt.grid(True, alpha=0.3)

# График 2: Стресс vs Оценки
plt.subplot(1, 3, 2)
plt.scatter(df['stress'], df['grade'], color='red', alpha=0.6)
plt.xlabel('Уровень стресса (1-10)')
plt.ylabel('Оценка')
plt.title('Зависимость оценок от стресса')
plt.grid(True, alpha=0.3)

# График 3: Кофе vs Оценки
plt.subplot(1, 3, 3)
plt.scatter(df['coffee'], df['grade'], color='brown', alpha=0.6)
plt.xlabel('Чашек кофе в день')
plt.ylabel('Оценка')
plt.title('Зависимость оценок от кофе')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('student_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("ВЫВОДЫ:")
print("=" * 50)
best_sleep = df.loc[df['grade'].idxmax(), 'sleep']
print(f"• Оптимальное количество сна в выборке: {best_sleep} часов")
print("• Наблюдается корреляция между режимом сна и успеваемостью")
print("• Для более точных выводов требуется расширение выборки")

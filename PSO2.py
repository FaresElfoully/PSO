import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

data = pd.read_csv('Customer Churn.csv')

def fitness_function(data):
    missing_values = data.isnull().sum().sum()
    duplicates = data.duplicated().sum()
    return missing_values + duplicates

def particle_swarm_optimization(data, max_iterations=10, n_particles=5):
    particles = [data.copy() for _ in range(n_particles)]
    p_best = particles.copy()
    p_best_fitness = [fitness_function(p) for p in particles]
    g_best = p_best[np.argmin(p_best_fitness)]

    for _ in range(max_iterations):
        for i, particle in enumerate(particles):
            if np.random.rand() > 0.5:
                particle = particle.drop(particle.sample(1).index)

            for col in particle.select_dtypes(include='number').columns:
                particle[col] = particle[col].fillna(particle[col].mean())
            
            for col in particle.select_dtypes(exclude='number').columns:
                particle[col] = particle[col].fillna(particle[col].mode()[0])

            fitness = fitness_function(particle)

            if fitness < p_best_fitness[i]:
                p_best[i] = particle
                p_best_fitness[i] = fitness

            if fitness < fitness_function(g_best):
                g_best = particle

    return g_best

cleaned_data = particle_swarm_optimization(data)
cleaned_data = pd.get_dummies(cleaned_data, drop_first=True)

X = cleaned_data.drop('Exited', axis=1, errors='ignore')
if 'Exited' in cleaned_data.columns:
    y = cleaned_data['Exited']
else:
    print("Error: Target column 'Churn_Yes' not found.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

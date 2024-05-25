import pandas as pd
import matplotlib.pyplot as plt

# Primeiro grafico

df = pd.read_csv('emotion_times.csv')


plt.figure(figsize=(10, 6))
plt.bar(df['Emotion'], df['Time (seconds)'], color='skyblue')
plt.title('Presença de Emoções')
plt.xlabel('Emoção')
plt.ylabel('Tempo (segundos)')
plt.grid(axis='y')
plt.show()

data = pd.read_csv('people_times.csv')

plt.figure(figsize=(10, 6))
plt.bar( data['People Count'], data['Time (seconds)'], color='skyblue')
plt.title('Foco de Alunos')
plt.xlabel('Foco')
plt.ylabel('Tempo (segundos)')
plt.grid(True)
plt.xticks(data['People Count'])  # Ensure all people counts are shown on x-axis
plt.show()


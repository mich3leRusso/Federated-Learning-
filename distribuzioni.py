
import numpy as np
import matplotlib.pyplot as plt

# Parametri della distribuzione normale: media=0, deviazione standard=1
mean = 0
num_samples = 1000000
classes_per_task = 10
stds = [0.1, 0.5, 1, 2, 3, 10]


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, std in enumerate(stds):
    mat = np.random.normal(loc=mean, scale=std, size=(classes_per_task, num_samples))

    exp = np.exp(mat)
    sum_exp = np.sum(exp, axis=0, keepdims=True)
    new_mat = exp / sum_exp
    max = np.max(new_mat, axis=0)

    axes[i].hist(max, bins=100, color='skyblue', edgecolor='black')
    axes[i].set_title(f"deviazione standard = {std}")
    axes[i].set_xticks(np.arange(0, 1.01, 0.1))
    axes[i].set_xlim(0, 1)
    axes[i].grid(True)
plt.show()

num_samples = 100000
x = np.arange(0, 10.1, 0.1).tolist()
y = []
for i, std in enumerate(x):
    mat = np.random.normal(loc=mean, scale=std, size=(classes_per_task, num_samples))

    exp = np.exp(mat)
    sum_exp = np.sum(exp, axis=0, keepdims=True)
    new_mat = exp / sum_exp
    max = np.max(new_mat, axis=0)
    y.append(np.mean(max))
    plt.xlim(0, 10)
    plt.ylim(0, 1)

plt.plot(x,y)
plt.show()

'''mean_1 = 1
mean_2 = 0
std_1 = 1
std_2 = 1

logits_1 = np.random.normal(loc=mean_1, scale=std, size=(1, num_samples))
logits_2 = np.random.normal(loc=mean_2, scale=std, size=(classes_per_task-1, num_samples))
mat = np.vstack((logits_1, logits_2))

exp = np.exp(mat)
sum_exp = np.sum(exp, axis=0, keepdims=True)
new_mat = exp / sum_exp
max = np.max(new_mat, axis=0)

plt.hist(max, bins=100, color='skyblue', edgecolor='black')
plt.title(f"deviazione standard = {std}")
plt.xticks(np.arange(0, 1.01, 0.1))
plt.xlim(0, 1)
plt.grid(True)
plt.show()'''
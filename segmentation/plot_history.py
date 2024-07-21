import matplotlib.pyplot as plt

file_path2 = 'histories/CentralWeld-training_history-deeplabv3_resnet101-2024-03-22_12-19.txt'
file_path = 'histories/Electrode-training_history-deeplabv3_resnet101-2024-03-22_13-30.txt'

epochs1 = []
losses1 = []
epochs2 = []
losses2 = []

with open(file_path, 'r') as file:
    for line in file:
        parts = line.split(',')
        if len(parts) == 2:
            epoch = int(parts[0].split(' ')[1])
            loss = float(parts[1].split(' ')[2])
            epochs1.append(epoch)
            losses1.append(loss)

with open(file_path2, 'r') as file:
    for line in file:
        parts = line.split(',')
        if len(parts) == 2:
            epoch = int(parts[0].split(' ')[1])
            loss = float(parts[1].split(' ')[2])
            epochs2.append(epoch)
            losses2.append(loss)

plt.figure(figsize=(5, 5))
plt.plot(epochs1, losses1, marker='o', linestyle='-', color='g', label='Electrode')
plt.plot(epochs2, losses2, marker='o', linestyle='-', color='b', label='Central Weld')
plt.title('Training Loss Over Epochs')  # Electrode
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

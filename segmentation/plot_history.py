import matplotlib.pyplot as plt

# file_path = 'histories/CentralWeld-training_history-deeplabv3_resnet101-2024-03-22_12-19.txt'
file_path = 'histories/Electrode-training_history-deeplabv3_resnet101-2024-03-22_13-30.txt'

epochs = []
losses = []

with open(file_path, 'r') as file:
    for line in file:
        parts = line.split(',')
        if len(parts) == 2:
            epoch = int(parts[0].split(' ')[1])
            loss = float(parts[1].split(' ')[2])
            epochs.append(epoch)
            losses.append(loss)

plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='g', label='Training Loss')
plt.title('Training Loss Over Epochs for Central Weld')  # Electrode
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

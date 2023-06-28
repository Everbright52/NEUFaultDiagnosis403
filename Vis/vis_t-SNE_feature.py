import numpy as np
import torch
import time
from model import CNN, CA_CNN_1, CA_CNN_2, ResNet18, MobileNet_V2, MobileNet_V3, ViTModel, DeepViTModel
from torch.nn.parallel import DataParallel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_features(model, data):
    features = []
    def hook(module, input, output):
        features.append(output.flatten(1).detach().cpu().numpy())

    model.eval()
    handle = model.module.conv_layers.register_forward_hook(hook)  # 使用 model.module

    with torch.no_grad():
        _ = model(data)
    handle.remove()

    features = np.concatenate(features)
    return features

# Load Data
dir = '../data/image_cwt.npz'
data = np.load(dir)
datas = data['data']
labels = data['labels']

# Random select data
nums = 1500
random_indices = np.random.choice(len(datas), nums, replace=False)
batch_data = datas[random_indices]

# Convert data to PyTorch tensors
batch_data = torch.from_numpy(batch_data).permute(0, 3, 1, 2).float()

model_dict = {
    'CNN': {
        'model_class': CNN,
        'weight_path': '/home/evb52/DeepLearning/classification/model_and_metric/Normal_unfold/CNN-data(image_cwt)-ep(50)-lr(0.0001)-bs(32).npz',
        'init_args': {
            'num_classes': 10
        }
    },
    # 其他模型
}

model_name = 'CNN'  # 指定要使用的模型名称

if model_name not in model_dict:
    raise ValueError("Invalid model name.")

model_info = model_dict[model_name]
model_class = model_info['model_class']
weight_path = model_info['weight_path']

# Load model weights
model_result = np.load(weight_path, allow_pickle=True)
model_state_dict = model_result['model'].item()

model_class = model_dict[model_name]['model_class']
model = model_class(**model_info['init_args'])
model.load_state_dict(model_state_dict)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.to(device)

# Move data to GPU
batch_data = batch_data.to(device)

# Get features
# features = get_features(model.module.conv_layers.conv1, batch_data)
features = get_features(model, batch_data)


# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(features)


# Visualize t-SNE features
plt.figure(figsize=(8, 8))
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels[random_indices], cmap='tab10')
plt.colorbar()
plt.show()

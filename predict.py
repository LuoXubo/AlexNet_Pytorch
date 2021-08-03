import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    img_path = '../tulip.jpg'
    assert os.path.exists(img_path), 'file: "{}" does not exist.'.format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), 'file: "{}" does not exist.'.format(json_path)

    json_file = open(json_path, 'r')
    class_indict = json.load(json_file)

    model = AlexNet(num_classes=5).to(device)

    weight_path = './AlexNet.pth'
    assert os.path.exists(weight_path), 'file: "{}" does not exist.'.format(weight_path)
    model.load_state_dict(torch.load(weight_path))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = 'class: {} prob: {:.3}'.format(class_indict[str(predict_cla)],
                                               predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()

if __name__ == '__main__':
    main()
    

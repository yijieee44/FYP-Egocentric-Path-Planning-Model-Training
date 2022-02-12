import timm
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms as T
import cv2
import time

from utils.utils import draw_path_from_device_path, target_denormalize_mean_std_tensor

PRED_POINT_SIZE = (832, 640)

transform = T.Compose([
        T.Resize([320, 416]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

model = timm.create_model('efficientnetv2_rw_t', pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(in_features=1024, out_features=512),
    nn.Linear(in_features=512, out_features=300),
    nn.Unflatten(1, (100, 3))
)
model.load_state_dict(torch.load("./models/eff_modelv2-epoch-9.pth"))
model.eval()
for param in model.parameters():
    param.grad = None



def predict_and_plot_path(img):
    img_copy = img.copy()
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    img_trans = transform(img_pil)

    since = time.time()

    with torch.no_grad():
        pred_path_norm = model.forward(img_trans.unsqueeze(0)).squeeze(0)

    print("Time: {:.2f}s".format((time.time()-since)))
    pred_path = target_denormalize_mean_std_tensor(pred_path_norm)

    plot_img = draw_path_from_device_path(pred_path.detach().numpy(), img_copy, fill_color=(128,0,255), line_color=(255, 255, 255))

    return plot_img
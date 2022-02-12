from PIL import Image
import numpy as np
import onnxruntime
from torchvision import transforms as T
import cv2
import time

from utils.utils import draw_path_from_device_path, target_denormalize_mean_std_np

transform = T.Compose([
        T.Resize([320, 416]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

ort_session = onnxruntime.InferenceSession("./models/eff_modelv2-epoch-9.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def predict_and_plot_path(img):
    img_copy = img.copy()
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    img_trans = transform(img_pil)

    since = time.time()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_trans.unsqueeze(0))}
    ort_outputs = ort_session.run(None, ort_inputs)

    pred_path_norm = ort_outputs[0].reshape(100,3)

    print("Time: {:.2f}ms".format((time.time()-since) * 1000))
    pred_path = target_denormalize_mean_std_np(pred_path_norm)

    plot_img = draw_path_from_device_path(pred_path, img_copy, fill_color=(128,0,255), line_color=(255, 255, 255))

    return plot_img
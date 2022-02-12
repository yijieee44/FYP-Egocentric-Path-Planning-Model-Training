import cv2

## change here to test different model
## run_onnx_v2_model, run_onnx_lite_model, run_v2_model, run_lite_model
from inference.run_onnx_v2_model import predict_and_plot_path


def main():
    IMG_DIR = ".//resource//0170_img.jpg"
    img = cv2.imread(IMG_DIR)

    plot_img = predict_and_plot_path(img)

    cv2.imshow("Original Image", img)
    cv2.imshow("Plot Image", plot_img)
    cv2.waitKey(0)

    pass


if __name__ == '__main__':
    main()

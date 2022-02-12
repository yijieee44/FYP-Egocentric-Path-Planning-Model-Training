import cv2

## change here to test different model
## run_onnx_v2_model, run_onnx_lite_model, run_v2_model, run_lite_model
from inference.run_onnx_v2_model import predict_and_plot_path

def project_path(data_dir):
    vid_cap = cv2.VideoCapture(data_dir)
    frame_count = 0

    while vid_cap.isOpened():
        ret, frame = vid_cap.read()  # ret return True if video is captured

        if ret:
            if frame_count % 3 == 0:
                frame_with_path = predict_and_plot_path(frame)
                frame_with_path = cv2.addWeighted(frame_with_path, 0.3, frame, 0.7, 0, frame_with_path)

                cv2.imshow('Driven path projected onto video', frame_with_path)
                cv2.waitKey(10)

        else:
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

        frame_count += 1


def main():
    VID_DIR = "./resource/day_time_f.hevc"
    project_path(VID_DIR)


if __name__ == '__main__':
    main()

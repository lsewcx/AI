import cv2
import fastdeploy.vision as vision
import fastdeploy as fd

# 读取视频文件
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)


# 创建视频写入对象
output_path = "output_video.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 逐帧进行推理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    option = fd.RuntimeOption()
    option.use_gpu(device_id=0)
    model = fd.vision.detection.PPYOLOE(
        "model.pdmodel", "model.pdiparams", "infer_cfg.yml" ,runtime_option=option)
    # model = vision.detection.PPYOLOE("model.pdmodel",
    #                                  "model.pdiparams",
    #                                  "infer_cfg.yml")
    # 进行推理
    # frame1 = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    result = model.predict(frame)
    vis_im = vision.vis_detection(frame, result, score_threshold=0.5)

    # result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)

    # 过滤推理结果
    # vis_frame = vision.vis_detection(frame, result, score_threshold=0.5
    # 写入帧
    cv2.imshow("frame", vis_im)

    key = cv2.waitKey(1)
    if (key == 27):
        break;

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

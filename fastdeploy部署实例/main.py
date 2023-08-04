import cv2
import fastdeploy.vision as vision
import fastdeploy



# 读取视频文件
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)
option = fastdeploy.RuntimeOption()
option.use_gpu(device_id=0)
model = fastdeploy.vision.detection.PPYOLOE(
    "model.pdmodel", "model.pdiparams", "infer_cfg.yml", runtime_option=option)
print(model.model_name())

# 逐帧进行推理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = model.predict(frame)
    for i, value in enumerate(result.scores):
        if value > 0.9:
            if result.label_ids[i]==0:
                print("减速带")
            if result.label_ids[i]== 1:
                print("粮仓识别")
            elif result.label_ids[i]==2:
                print("斑马线识别")
            elif result.label_ids[i]==3:
                print("锥桶识别")
            elif result.label_ids[i] == 4:
                print("桥")
            elif result.label_ids[i] == 5:
                print("猪")
            elif result.label_ids[i]==6:
                print("拖拉机")
            elif result.label_ids[i] == 7:
                print("玉米")
    vis_im = vision.vis_detection(frame, result, score_threshold=0.5)
    cv2.imshow("frame", vis_im)

    key = cv2.waitKey(1)
    if key == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

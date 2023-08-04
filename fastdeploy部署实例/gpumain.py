import fastdeploy as fd
import cv2

# 配置runtime，加载模型
model = fd.vision.detection.PPYOLOE(
    "model.pdmodel", "model.pdiparams", "infer_cfg.yml")

# 预测图片检测结果
im = cv2.imread("64df7a1f376fe297.jpg")
result = model.predict(im)
for i, value in enumerate(result.scores):
    if value > 0.6:
        if result.label_ids[i] == 0:
            print("减速带")
        if result.label_ids[i] == 1:
            print("粮仓识别")
        elif result.label_ids[i] == 2:
            print("斑马线识别")
        elif result.label_ids[i] == 3:
            print("锥桶识别")
        elif result.label_ids[i] == 4:
            print("桥")
        elif result.label_ids[i] == 5:
            print("猪")
        elif result.label_ids[i] == 6:
            print("拖拉机")
        elif result.label_ids[i] == 7:
            print("玉米")

# # 预测结果可视化11
vis_im = fd.vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("visualized_result.jpg", vis_im)
up_width = 480
up_height = 640
up_points = (up_width, up_height)
resized_up = cv2.resize(vis_im, up_points, interpolation= cv2.INTER_LINEAR)
cv2.imshow("visua",resized_up)
cv2.waitKey(0)

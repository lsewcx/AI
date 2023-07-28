# 引入opencv和fastdeploy包
import cv2
import fastdeploy.vision as vision
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 读取推理图片
im = cv2.imread("000000014439.jpg")
# 加载推理模型
model = vision.detection.PPYOLOE("model.pdmodel",
                                 "model.pdiparams",
                                 "infer_cfg.yml")



# 推理
result = model.predict(im)

# 打印推理结果
print(result)
# 过滤推理结果
vis_im = vision.vis_detection(im, result, score_threshold=0.5)
# 保存推理结果
cv2.imwrite("vis_image.jpg", vis_im)
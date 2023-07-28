import fastdeploy as fd
import cv2

model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz"
image_url = "https://bj.bcebos.com/fastdeploy/tests/test_det.jpg"
fd.download_and_decompress(model_url, path=".")
fd.download(image_url, path=".")

model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel"
params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams"
infer_cfg_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml"
model = fd.vision.detection.PPYOLOE(model_file, params_file, infer_cfg_file)


im = cv2.imread("000000014439.jpg")

result = model.predict(im)
print(result)

vis_im = fd.vision.visualize.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image123.jpg", vis_im)
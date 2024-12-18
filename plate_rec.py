import numpy as np
import cv2
from ocr_rec import TextRecognizer, init_args
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")
class PlateRecognizer:
    def __init__(self, det_model_path="license_models/y11n-pose_plate_best.onnx"):
        self.model_det = YOLO(det_model_path)
        parser = init_args().parse_args()
        self.model_ocr = TextRecognizer(parser)
    def recognize(self, img):
        plate_objs=[]
        # detect plates
        plates = self.model_det(img, verbose=False)
        for plate, conf in zip(plates[0].boxes.xyxy, plates[0].boxes.conf):
            x1, y1, x2, y2 = plate.cpu()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            plate_img = img[y1:y2, x1:x2]
            # recognize text
            try:
                rec_res, _ = self.model_ocr([plate_img])
            except Exception as E:
                print(E)
                exit()
            if len(rec_res[0])>0:
                obj = {}
                obj['text'] = rec_res[0][0]
                obj['score_text'] = rec_res[0][1]
                obj['bbox'] = [x1, y1, x2, y2]
                obj['score_bbox'] = conf.cpu().numpy().item()
                plate_objs.append(obj)
        return plate_objs
      

def DrawPlateNum(img, palte_num, x1, y1):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("simsun.ttc", 40)
    draw.text((x1, y1), palte_num, font=font, fill=(255, 255, 0))
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_bgr

def demo():
    parser = init_args().parse_args()
    img = cv2.imread(parser.image_path)
    if img is None:
        print("Error: no image found")
        return
    plate_rec = PlateRecognizer()
    plate_objs = plate_rec.recognize(img)
    print(plate_objs)
    for bbox in plate_objs:
        x1, y1, x2, y2 = bbox['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        img = DrawPlateNum(img, bbox['text'], x1, y1)
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    cv2.imwrite("result.jpg", img)

# 计算平均识别时间
def test_ocr_speed():
    plate_rec = PlateRecognizer()
    imgs = [cv2.imread("test_plate/ec4a268fc_r.jpg"),
     cv2.imread("test_plate/hcqccip6335913.jpg"),
     cv2.imread("test_plate/u=1570203173,1434495175&fm=253&fmt=auto&app=138&f=JPEG.webp"),
     ]
    #warm up
    plate_objs = plate_rec.recognize(imgs[0])
    time_start = cv2.getTickCount()
    cal_count = 0
    while True:
        plate_objs = plate_rec.recognize(imgs[cal_count%len(imgs)])
        cal_count += 1
        time_pause = cv2.getTickCount()
        time_cost = (time_pause - time_start)*1000 / cv2.getTickFrequency()
        print(f"avgtime cost: {time_cost/(cal_count)}ms {plate_objs}")
        

if __name__ == "__main__":
    parser = init_args().parse_args()
    if parser.speed_test:
        test_ocr_speed()
    else:        
        demo()

        



import torch
import cv2
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class ObjectDetector:
    def __init__(self):

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.label_map = self.load_label_map()

    @staticmethod
    def load_label_map():

        return {
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            5: "airplane",
            6: "bus",
            7: "train",
            8: "truck",
            9: "boat",
            10: "traffic light",

        }

    def detect_objects(self, frame):

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(image_tensor)


        objects = []
        for idx, box in enumerate(predictions[0]['boxes']):
            score = predictions[0]['scores'][idx].item()
            label_id = predictions[0]['labels'][idx].item()

            if score > 0.5 and label_id in self.label_map:
                objects.append({
                    "name": self.label_map[label_id],
                    "bbox": box.int().tolist(),
                    "id": idx + 1,
                    "sub_objects": []
                })

        return objects

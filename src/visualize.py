import cv2
import json

def visualize_bbox(image_path, bbox_path):
    image = cv2.imread(image_path)
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)
    x1, y1, x2, y2 = bbox['bbox']
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = '../data/test3.png'
    image_path = '/home/user/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/val2017/000000002261.jpg'
    bbox_path = '../data/3_bbox.json'
    visualize_bbox(image_path, bbox_path)
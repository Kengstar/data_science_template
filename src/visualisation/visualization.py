import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision.datasets import VOCDetection, CocoDetection


def plot_bb_voc(img, gt_voc):
    """

    :param PIL img:
    :param list gt_voc: format (x1,y1,x2,y2) #TODO check format, list of objects, look at key
    :return:
    """
    im = np.asarray(img)
    for sample in gt_voc:
        label = sample['name']
        bbox = sample['bndbox']
        x1, y1 = int(bbox['xmin']), int(bbox['ymin'])
        x2, y2 = int(bbox['xmax']), int(bbox['ymax'])
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return im


def plot_bb(img, target):
    im = np.asarray(img)
    import cv2
    for sample in zip(target['labels'], target['boxes']):
        label = sample[0]
        bbox = sample[1]
        x1, y1 = bbox[0], bbox[1]
        x2, y2 = bbox[2], bbox[3]
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pascal_voc = VOCDetection(root="VOC", year="2012",
                              image_set="train", download=False)

    sample = pascal_voc[0]
    img, gt_dict = sample

    bbox_gt = gt_dict['annotation']['object']
    print(bbox_gt)
    plot_bb_voc(img, bbox_gt)
    #coco = CocoDetection("COCO")

    """
    root, annFile, transform=None, target_transform=None, transforms=None
    """
    #sample = coco[0]
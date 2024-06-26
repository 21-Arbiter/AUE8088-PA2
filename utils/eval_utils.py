import torch
import numpy as np
import matplotlib.pyplot as plt

# 평가 함수
def evaluate_model(model, dataloader, device):
    model.eval()
    all_detections = []
    all_annotations = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            for output, target in zip(outputs, targets):
                all_detections.append(output)
                all_annotations.append(target)
    return all_detections, all_annotations

# IoU 계산 함수
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Miss rate와 FPPI 계산 함수
def calculate_miss_rate_fppi(detections, annotations, iou_threshold=0.5):
    fppi = []
    miss_rate = []
    for conf_threshold in np.linspace(0, 1, 101):
        tp = 0
        fp = 0
        fn = 0
        num_images = len(detections)
        for dets, annos in zip(detections, annotations):
            dets = [det for det in dets if det['score'] > conf_threshold]
            if len(annos['boxes']) == 0:
                fp += len(dets)
                continue
            matched = np.zeros(len(annos['boxes']), dtype=bool)
            for det in dets:
                best_iou = 0
                best_idx = -1
                for i, box in enumerate(annos['boxes']):
                    iou = calculate_iou(det['box'], box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_iou > iou_threshold:
                    if not matched[best_idx]:
                        tp += 1
                        matched[best_idx] = True
                    else:
                        fp += 1
                else:
                    fp += 1
            fn += len(annos['boxes']) - sum(matched)
        fppi.append(fp / num_images)
        miss_rate.append(fn / (tp + fn))
    return fppi, miss_rate

# Miss rate - FPPI plot 그리기
def plot_miss_rate_fppi(fppi, miss_rate):
    plt.figure()
    plt.plot(fppi, miss_rate, marker='o', color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positives Per Image (FPPI)')
    plt.ylabel('Miss Rate')
    plt.title('Miss Rate vs. FPPI')
    plt.grid(True)
    plt.show()

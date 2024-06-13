import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 다이어그램 그리기
fig, ax = plt.subplots(figsize=(10, 20))

# 레이어 리스트
layers = [
    ("Input", 416, 416, 3),
    ("Conv (3, 6, 2, 2)", 208, 208, 32),
    ("Conv (3, 3, 1, 1)", 208, 208, 64),
    ("BottleneckCSP (128)", 208, 208, 128),
    ("Conv (3, 3, 1, 1)", 208, 208, 256),
    ("BottleneckCSP (256)", 208, 208, 512),
    ("Conv (3, 3, 1, 1)", 208, 208, 1024),
    ("FPN (256)", 208, 208, 256),
    ("PAN (128)", 208, 208, 128),
    ("Detect", "num_classes, anchors")
]

# 직사각형 추가
y_offset = 0
for layer in layers:
    if len(layer) == 4:
        name, h, w, c = layer
        label = f"{name}: {h}x{w}x{c}"
    else:
        name, details = layer
        label = f"{name}: {details}"
    rect = patches.Rectangle((0, y_offset), 1, 1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(1.1, y_offset + 0.5, label, verticalalignment='center', fontsize=12)
    y_offset += 2

# 축 설정
ax.set_xlim(0, 5)
ax.set_ylim(-1, len(layers) * 2)
ax.axis('off')


# 다이어그램 저장
plt.tight_layout()
plt.savefig('/home/jmsong/Dev_Folder/AUE8088-PA2/yolov5n_architecture.png')
plt.show()

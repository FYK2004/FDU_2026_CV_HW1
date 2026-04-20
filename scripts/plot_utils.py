# 测试脚本将输出混淆矩阵图片
# 1. 导入 matplotlib
# 2. 在 main() 中绘制并保存图片
# 3. 图片保存到 artifacts/confusion_matrix.png

import matplotlib.pyplot as plt
import numpy as np

# ...existing code...

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

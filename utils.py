import os
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict


def generate_vs_images(task, subtasks=["default", "warmup", "cosine", "linear"], output_path="./output"):
    os.makedirs(os.path.join(output_path, "vs_images"), exist_ok=True)

    def load_images(path):
        images = []
        for file in os.listdir(path):
            if not file.endswith(".png"):
                continue
            image_path = os.path.join(path, file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    data = OrderedDict()
    data["gt"] = load_images(os.path.join(output_path, f"{task}_{subtasks[0]}", "gt"))
    data["mask"] = load_images(os.path.join(output_path, f"{task}_{subtasks[0]}", "gt_keep_mask"))
    data["gt_masked"] = load_images(os.path.join(output_path, f"{task}_{subtasks[0]}", "gt_masked"))

    for subtask in subtasks:
        data[subtask] = load_images(os.path.join(output_path, f"{task}_{subtask}", "inpainted"))

    m, n = len(data["gt"]), 3 + len(subtasks) # (8, 7)
    fig, axes = plt.subplots(m, n, figsize=(4*n, 4*m + 1.5))  # Increased figure height

    col_labels = ["gt", "mask", "gt_masked"] + subtasks
    label_fontsize = 30  # Reduced font size
    
    for j, (key, images) in enumerate(data.items()):
        for i, image in enumerate(images):
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
        
        axes[-1, j].text(0.5, -0.1, col_labels[j], rotation=0, ha='center', va='top', 
                         transform=axes[-1, j].transAxes, fontsize=label_fontsize, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"comparison_{task}.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    model_name = "celeba"
    tasks = ["ev2li", "ex64", "genhalf", "nn2", "thick", "thin"]
    for task in tasks:
        generate_vs_images(task)


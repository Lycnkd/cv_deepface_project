from deepface import DeepFace
import os

# 定义符合条件的函数
def is_match(result, min_age, gender, emotion):
    return result['age'] > min_age and result['dominant_gender'] == gender and result['dominant_emotion'] == emotion

# 指定条件
min_age = 20
gender = "Man"
emotion = "neutral"

# 获取要分析的图像路径列表
image_folder = "/home/liuyichen/deepface/tests/dataset"
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('jpg', 'jpeg', 'png'))]

# 保存符合条件的图像路径
matched_images = []

# 分析每个图像并过滤结果
for img_path in image_paths:
    try:
        result = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion'])[0]
        print(result)
        if is_match(result, min_age, gender, emotion):
            matched_images.append(img_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# 输出符合条件的图像路径
print("Matched images:")
for img in matched_images:
    print(img)
import os
import sys
from deepface import DeepFace
from lyc import clustering, video_detect
import logging

# 设置环境变量，抑制 TensorFlow 的日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('deepface').setLevel(logging.ERROR)



# 定义数据库路径
DB_PATH = "/home/liuyichen/deepface/tests/dataset"

def add_user(image_path):
    if os.path.exists(image_path):
        os.makedirs(DB_PATH, exist_ok=True)
        dest_path = os.path.join(DB_PATH, os.path.basename(image_path))
        os.rename(image_path, dest_path)
        print(f"Image added to database at {dest_path}")
    else:
        print("Image path does not exist.")

def delete_user(image_path):
    db_image_path = os.path.join(DB_PATH, os.path.basename(image_path))
    if os.path.exists(db_image_path):
        os.remove(db_image_path)
        print(f"Image {db_image_path} deleted from database.")
    else:
        print("Image path not found in database.")

def face_match(image_path):
    try:
        dfs = DeepFace.find(img_path=image_path, db_path=DB_PATH, model_name="VGG-Face")
        if len(dfs) > 0:
            print("Match found:", dfs)
            return True
        else:
            print("No match found.")
            return False
    except Exception as e:
        print(f"Error during face matching: {e}")
        return False

def face_search(min_age, gender, emotion):
    def is_match(result, min_age, gender, emotion):
        return result['age'] > min_age and result['dominant_gender'] == gender and result['dominant_emotion'] == emotion

    image_paths = [os.path.join(DB_PATH, img) for img in os.listdir(DB_PATH) if img.endswith(('jpg', 'jpeg', 'png'))]
    matched_images = []

    for img_path in image_paths:
        try:
            result = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion'])[0]
            print(result)
            if is_match(result, min_age, gender, emotion):
                matched_images.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("Matched images:")
    for img in matched_images:
        print(img)

def face_search_in_video(video_path):
    # try:
        # video_results = DeepFace.analyze(video_path, db_path=DB_PATH, model_name="VGG-Face")
        # if len(video_results) > 0:
        #     print("Matches found in video:", video_results)
        #     return True
        # else:
        #     print("No matches found in video.")
        #     return False
    video_detect(video_path)
    # except Exception as e:
    #     print(f"Error during face search in video: {e}")
    #     return False
    

def menu():
    while True:
        print("\nMenu:")
        print("1. Add User")
        print("2. Delete User")
        print("3. Face Match")
        print("4. Face Search")
        print("5. Face Search in Video")
        print("6. Clustering faces in database")
        print("7. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            image_path = input("Enter the image path to add: ")
            add_user(image_path)
        elif choice == '2':
            image_path = input("Enter the image path to delete: ")
            delete_user(image_path)
        elif choice == '3':
            image_path = input("Enter the image path for face match: ")
            face_match(image_path)
        elif choice == '4':
            min_age = int(input("Enter minimum age: "))
            gender = input("Enter gender (Man/Woman): ")
            emotion = input("Enter emotion: ")
            face_search(min_age, gender, emotion)
        elif choice == '5':
            video_path = input("Enter the video path for face search: ")
            face_search_in_video(video_path)
        elif choice == '6':
            clustering(DB_PATH)
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
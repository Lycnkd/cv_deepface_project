from deepface import DeepFace
import matplotlib.pyplot as plt
import os
import cv2


# directory_path = '/home/liuyichen/deepface/tests/dataset'

def count_all_files(directory_path):
    file_count = 0

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory_path):
        # 累加文件数量
        file_count += len(files)

    return file_count


def is_image_file(file_path):
    # 定义支持的图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    return os.path.splitext(file_path)[1].lower() in image_extensions

def get_file_indices_in_directory(directory_path, file_paths):
    # 获取目录下的所有文件
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # 查找文件路径在所有文件列表中的索引
    file_indices = [all_files.index(file_path) for file_path in file_paths if file_path in all_files]
    
    return file_indices

def get_file_at_index(directory_path, index):
    # 获取目录下的所有文件
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # 检查索引是否在范围内
    if index < 0 or index >= len(all_files):
        raise IndexError("Index out of range.")
    
    # 返回第 index 个文件的路径
    return all_files[index]

def clustering(db_path):
    dict = {}
    pointer = 0
    size_of_database = count_all_files(db_path)

    sig_strand = [0] * size_of_database
    cat = 0
    
    while(pointer < size_of_database):
        print(f'pointer is {pointer}')
        if(sig_strand[pointer] == 0):
            if(is_image_file(get_file_at_index(db_path, pointer))):
                dfs = DeepFace.find(
                img_path = get_file_at_index(db_path, pointer),
                db_path = db_path,
                model_name="VGG-Face"
                )
                # 提取匹配结果中的文件路径
                matched_files = dfs[0]['identity']
                print(f'match file is {matched_files}')

                # 获取文件在目录中的索引
                file_indices = get_file_indices_in_directory(db_path, matched_files)
                sig_strand[pointer] = 1
                # print(f'file_indices is {file_indices}')
                for i in file_indices:
                    sig_strand[i] = 1
                    
                # file_indices.append(pointer)
                result_list = []
                for i in file_indices:
                    result_list.append(get_file_at_index(db_path, i))
                    
                dict[cat] = result_list
                cat += 1
            else:
                sig_strand[pointer] = 1
        else:
            pointer += 1
            
    for key, value in dict.items():
        print(f"{key}: {value}")
        

def video_detect(video_path):
    # output_folder = "/home/liuyichen/deepface/temp_frame"
    output_folder = "./temp_frame"

    cap = cv2.VideoCapture(video_path)
        
    # 检查视频是否打开成功
    if not cap.isOpened():
        print("Error: Could not open video.")

    else:
        # 获取视频的帧率
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 当前帧计数
        frame_count = 0

        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每秒截取一帧
            if frame_count % int(fps) == 0:
                frame_number = frame_count // int(fps)
                frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
            
            try:
                dfs = DeepFace.find(
                    img_path = frame_filename,
                    # db_path = "/home/liuyichen/deepface/tests/dataset",
                    db_path = "./tests/dataset",
                    model_name="VGG-Face"
                )
                
                if not str(dfs[0]).startswith('Empty'):
                    print('Detect True!')
                    print(dfs)
                    break
            
            except ValueError:
                print('no face')
                
            
            

        # 释放视频捕获对象
        cap.release()
        # print("Done.")
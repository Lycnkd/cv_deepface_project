# FRD x deepface


<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240"></p>

Facial Recognition with Database (FRD)  is a lightweight face recognition and facial attribute analysis ([age](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), gender, emotion and race) application for python. It is a hybrid face recognition framework using models `VGG-Face`

[`Experiments`]([https://github.com/Lycnkd/cv_deepface_project/tree/master/benchmarks]) show that human beings have 97.53% accuracy on facial recognition tasks whereas those models already reached and passed that accuracy level.

## Installation 

you can install FRD from its source code.

```shell
$ git clone https://github.com/Lycnkd/cv_deepface_project.git
$ cd cv_deepface_project
$ pip install -e .
```

Then you will be able to import the library and use its functionalities.

```python
$ pip install deepface

from deepface import DeepFace
```

**Facial Recognition**

A modern [**face recognition pipeline**](https://sefiks.com/2020/05/01/a-gentle-introduction-to-face-recognition-in-deep-learning/) consists of 5 common stages: [detect](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [normalize](https://sefiks.com/2020/11/20/facial-landmarks-for-face-recognition-with-dlib/), [represent](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and [verify](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). While Deepface handles all these common stages in the background, you don’t need to acquire in-depth knowledge about all the processes behind it. You can just call its verification, find or analysis function with a single line of code.



## Face Recognition System

This system provides multiple face recognition and management functionalities, including adding users, deleting users, face matching, face searching, video face detection, and face clustering.

### Overview

The system allows users to manage a local database of face images and perform various operations. Users can add or remove images, search for faces based on specific attributes, match a face against the database, and detect faces in videos.

### Adding a User

To add a user, input the user ID followed by the image path. The image will be added to the database.

```python
def add_user(user_id, image_path):
    # Logic to add image to database
    pass
```

### Deleting a User

To delete a user, input the image path from the database. The corresponding user image will be removed.

```python
def delete_user(image_path):
    # Logic to delete image from database
    pass
```

### Face Matching

For face matching, input the image path to be matched. The system uses the following method to perform the match. If a match is found, it returns `True`.

```python
from deepface import DeepFace

def face_match(path1):
    dfs = DeepFace.find(
        img_path=path1,
        db_path="/deepface/tests/dataset",
        model_name="VGG-Face"
    )
    return not dfs.empty
```

### Face Searching

Face searching allows you to input three keywords: age, gender, and emotion. The system will detect all faces that match these criteria and display them.

```python
from deepface import DeepFace
import os

# Define a function to check if a face matches the criteria
def is_match(result, min_age, gender, emotion):
    return result['age'] > min_age and result['dominant_gender'] == gender and result['dominant_emotion'] == emotion

def face_search(min_age, gender, emotion):
    # Get the list of image paths to analyze
    image_folder = "/deepface/tests/dataset"
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('jpg', 'jpeg', 'png'))]

    # Store paths of images that match the criteria
    matched_images = []

    # Analyze each image and filter results
    for img_path in image_paths:
        try:
            result = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion'])[0]
            if is_match(result, min_age, gender, emotion):
                matched_images.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Output paths of images that match the criteria
    return matched_images

```

### Video Face Detection

This feature supports uploading a video and detecting if any faces in the video match those in the database.

### Face Clustering

Face clustering identifies and groups all similar faces in the database, displaying the results.

### Exiting the Program

Exit the program.

## Usage Instructions

1. Start the program and select the desired function by entering the corresponding number.
2. Follow the prompts to input the required information or paths.
3. The system will execute the requested function and return the results.

## System Requirements

- Python 3.10
- DeepFace library





**Face recognition models selection**

FRD is a **hybrid** face recognition package. It currently wraps many **state-of-the-art** face recognition models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/), `SFace` and `GhostFaceNet`. The default configuration uses VGG-Face model.

```python
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

#face verification
result = DeepFace.verify(
  img1_path = "img1.jpg",
  img2_path = "img2.jpg",
  model_name = models[0],
)

#face recognition
dfs = DeepFace.find(
  img_path = "img1.jpg",
  db_path = "C:/workspace/my_db", 
  model_name = models[1],
)

#embeddings
embedding_objs = DeepFace.represent(
  img_path = "img.jpg",
  model_name = models[2],
)
```

FaceNet, VGG-Face, ArcFace and Dlib are overperforming ones based on experiments - see [`BENCHMARKS`](https://github.com/serengil/deepface/tree/master/benchmarks) for more details. You can find the measured scores of various models in DeepFace and the reported scores from their original studies in the following table.

| Model        | Measured Score | Declared Score |
| ------------ | -------------- | -------------- |
| Facenet512   | 98.4%          | 99.6%          |
| Human-beings | 97.5%          | 97.5%          |
| Facenet      | 97.4%          | 99.2%          |
| Dlib         | 96.8%          | 99.3 %         |
| VGG-Face     | 96.7%          | 98.9%          |
| ArcFace      | 96.7%          | 99.5%          |
| GhostFaceNet | 93.3%          | 99.7%          |
| SFace        | 93.0%          | 99.5%          |
| OpenFace     | 78.7%          | 92.9%          |
| DeepFace     | 69.0%          | 97.3%          |
| DeepID       | 66.5%          | 97.4%          |

Conducting experiments with those models within DeepFace may reveal disparities compared to the original studies, owing to the adoption of distinct detection or normalization techniques. Furthermore, some models have been released solely with their backbones, lacking pre-trained weights. Thus, we are utilizing their re-implementations instead of the original pre-trained weights.



### API

FRD serves an API as well - see `api folder` for more details. You can clone deepface source code and run the api with the following command. It will use gunicorn server to get a rest service up. In this way, you can call deepface from an external system such as mobile app or web.

```shell
cd scripts
./service.sh
```

Face recognition, facial attribute analysis and vector representation functions are covered in the API. You are expected to call these functions as http post methods. Default service endpoints will be `http://localhost:5000/verify` for face recognition, `http://localhost:5000/analyze` for facial attribute analysis, and `http://localhost:5000/represent` for vector representation. You can pass input images as exact image paths on your environment, base64 encoded strings or images on web. [Here](https://github.com/serengil/deepface/tree/master/deepface/api/postman), you can find a postman project to find out how these methods should be called.



## Contribution

Pull requests are more than welcome! If you are planning to contribute a large patch, please create an issue first to get any upfront questions or design decisions out of the way first.

Before creating a PR, you should run the unit tests and linting locally by running `make test && make lint` command. Once a PR sent, GitHub test workflow will be run automatically and unit test and linting jobs will be available in [GitHub actions](https://github.com/serengil/deepface/actions) before approval.

If you have any questions or suggestions, please contact 12112722@mail.sustech.edu.cn.

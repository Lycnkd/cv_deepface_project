### Experiments

#### Dataset

Nowadays, a new state-of-the-art face recognition model is raised everyday. Researchers adopted Labeled faces in the wild or shortly LFW data set as a de facto standard to evaluate face recognition models and compare with existing ones. Luckily, scikit-learn provides LFW data set as an out-of-the-box module. In our experiments, we will evaluate our model on LFW data set within scikit-learn API.

Data set stores low resolution images as seen. This will be challenging for a face recognition model.

![image-20240615200625298](/Users/sco/Library/Application Support/typora-user-images/image-20240615200625298.png)

fetch_lfw_pairs function loads LFW data set. Default calling loads images in gray scale. That’s why, I’ll set its color argument to true. Besides, default usage loads train set images but in this post, I’ll just evaluate an existing model on LFW data set. I just need test set. That’s why, I’ll set subset argument to test. Finally, fetch_lfw_pairs function decreases the image resolution. Setting resize argument to 1 saves the original size. 

There are 1000 instances in the test set. First half of them are same person whereas second half is different persons.

We use the following code to 



#### Implementation Details

We’ve retrieved pair images in the code block above. Face recognition task will be handled in the same for loop. DeepFace package for python can handle face recognition with a few lines of code. It wraps several state-of-the-art face recognition models: VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID and Dlib ResNet. I can switch the face recognition model by specifying the model name argument in the verify function. I will use VGG-Face 2 model in this experiment.

<img src="/Users/sco/Library/Application Support/typora-user-images/image-20240615194320006.png" alt="image-20240615194320006" style="zoom:50%;" />

We stored actual and prediction labels in the dedicated variables. Sklearn offers accuracy metric calculations as out-of-the-functions as well.

![image-20240615194515324](/Users/sco/Library/Application Support/typora-user-images/image-20240615194515324.png)



#### Metrics

To assess the performance of our face recognition model, we will use several key metrics commonly employed in the field. These include:

1. **Accuracy**: This measures the proportion of correctly identified pairs out of the total pairs. It is the most straightforward metric but can be misleading if the dataset is imbalanced.
2. **Precision**: This measures the proportion of true positive identifications out of all positive identifications made by the model. It indicates how many of the positive identifications were actually correct.
3. **Recall (Sensitivity)**: This measures the proportion of true positive identifications out of all actual positive instances. It indicates how well the model identifies all positive instances.
4. **F1 Score**: This is the harmonic mean of precision and recall, providing a single metric that balances both concerns, especially useful for imbalanced datasets.
5. **Confusion Matrix**: This provides a detailed breakdown of true positives, false positives, true negatives, and false negatives, offering deeper insights into where the model may be making errors.



#### Experimental design & results

The performance of VGG-Face 2 on LFW data set for test subset is shown below. Results seem satisfactory for low resolution pairs.

The results can be visualized through the metrics mentioned above.

- **Accuracy**: 0.95
- **Precision**: 0.94
- **Recall**: 0.96
- **F1 Score**: 0.95

<img src="/Users/sco/Library/Application Support/typora-user-images/image-20240615194627038.png" alt="image-20240615194627038" style="zoom:50%;" />

Confusion matrix might give some insights as well.

<img src="/Users/sco/Library/Application Support/typora-user-images/image-20240615194709927.png" alt="image-20240615194709927" style="zoom:50%;" />

Confusion matrix is demonstrated below.

<img src="/Users/sco/Library/Application Support/typora-user-images/image-20240615194728866.png" alt="image-20240615194728866" style="zoom:50%;" />

These results demonstrate the effectiveness of the VGG-Face 2 model in recognizing faces within the LFW dataset. Despite the challenges posed by low-resolution images, the model performs well, achieving high accuracy and balanced precision and recall scores.

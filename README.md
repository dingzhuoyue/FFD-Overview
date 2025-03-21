# FFD-Overview
Advancing Forest Fire Detection with Computer Vision: A Comprehensive Review of Classification, Detection, and Segmentation Techniques
# Highlights
1、A comprehensive review of advancements in computer vision technology for forest fire detection over the past decade: This study systematically examines the application of the three core tasks—image classification, object detection, and image segmentation—in forest fire detection, providing a thorough reference for research in related fields.\
2、An in-depth analysis of the application of image processing and deep learning algorithms in forest fire detection: The study not only summarizes the advantages of existing technologies but also highlights the limitations of deep learning methods (such as data scarcity, class imbalance, and high computational complexity), as well as the challenges image processing methods face in handling complex backgrounds.\
3、It provides valuable insights for the development of advanced computer vision-based forest fire detection technologies: This paper summarizes some of the challenges faced by current methods, while also exploring future directions in the field, offering new perspectives for the advancement of forest fire detection technologies.
# Abstract
Forest fires pose significant threats to human life and natural resources, causing substantial economic losses. Computer vision technology, with its remarkable accuracy in minimizing these risks, has emerged as an indispensable tool for forest fire detection. This study provides a comprehensive review of advancements in computer vision techniques for forest fire detection over the past decade. We analyze the three core tasks of image classification, object detection, and image segmentation in detail. By examining the application of image processing and deep learning algorithms, we address the current challenges and explore potential future developments in this rapidly evolving field. Our review reveals that deep learning methods, despite their superior detection accuracy, face challenges such as data scarcity, class imbalance, and computational complexity. In contrast, image processing methods offer high computational efficiency but struggle with complex backgrounds. Here, we show that by combining the strengths of both approaches, researchers can develop more robust and efficient forest fire detection systems. This study provides valuable insights for developing advanced computer vision-based forest fire detection technologies.\
![img](https://github.com/dingzhuoyue/FFD-Overview/blob/master/img/1.png)\
Fig. 1. Schematic diagram of computer vision technology for the detection of forest fire.
# Methods:A Survey
## Classification task
| Method | Objects | Performance | Application | Paper Title | 
| --- | --- | --- | --- | --- |
| Image Processing | Flame | 25 frames/s | Classification | An image processing technique for fire detection in video images |
| Image Processing | Flame | response time is 56s | Classification | An early fire-detection method based on image processing |
| Image Processing | Flame and Smoke | Accuracy of 99% | Classification | Fire and smoke detection without sensors: Image processing based approach |
| Image Processing | Flame and Smoke | N/A | Classification | Fire and smoke detection using wavelet analysis and disorder characteristics |
| Deep Learning | Flame | Accuracy of 98.42% | Classification | FFireNet: Deep Learning Based Forest Fire Classification and Detection in Smart Cities |
| Deep Learning | Flame | response time is 0.7s | Classification | Research on the identification method for the forest fire based on deep learning |
| Deep Learning | Flame and Smoke | F1 score is 95.77% | Classification | EdgeFireSmoke: A Novel Lightweight CNN Model for Real-Time Video Fire–Smoke Detection |
## Object detection task
| Method | Objects | Performance | Application | Paper Title | 
| --- | --- | --- | --- | --- |
| Image Processing | Flame | True Positive Rate is 89.97% | Object Detection | Real-Time Forest Fire Detection Framework Based on Artificial Intelligence Using Color Probability Model and Motion Feature Analysis |
| Image Processing | Smoke | mAP is 97.8% | Object Detection | A Smoke Detection Algorithm with Multi-Texture Feature Exploration Under a Spatio-Temporal Background Model |
| Image Processing | Smoke | F1 score is 97.2% | Object Detection | 改进局部三值模式的烟雾识别和纹理分类 |
| Deep Learning | Flame | F1 score is 99.7% | Object Detection | Fire Detection from Images Using Faster R-CNN and Multidimensional Texture Analysis |
| Deep Learning | Flame | Accuracy is 85.2% | Object Detection | MS-FRCNN: A Multi-Scale Faster RCNN Model for Small Target Forest Fire Detection |
| Deep Learning | Smoke | 74.6FPS | Object Detection | Ea-yolo: efficient extraction and aggregation mechanism of YOLO for fire detection |
| Deep Learning | Smoke | 25FPS | Object Detection | Light-Weight Student LSTM for Real-Time Wildfire Smoke Detection |
| Deep Learning | Flame and Smoke | 24.57FPS | Object Detection | YOLO-ULNet: Ultralightweight Network for Real-Time Detection of Forest Fire on Embedded Sensing Devices |
## Image segmentation task
| Method | Objects | Performance | Application | Paper Title | 
| --- | --- | --- | --- | --- |
| Image Processing | Flame | N/A | Image Segmentation | UAV-based forest fire detection and tracking using image processing techniques |
| Image Processing | Flame | N/A | Image Segmentation | Forest Fire Detection Algorithm Based on Digital Image |
| Image Processing | Flame | Accuracy is 96.67% | Image Segmentation | 样本熵融合聚类算法的森林火灾图像识别研究 |
| Image Processing | Flame | N/A | Image Segmentation | A new flame‐based colour space for efficient fire detection |
| Deep Learning | Flame | Accuracy is 96.67% | Image Segmentation | ATT Squeeze U-Net: A Lightweight Network for Forest Fire Detection and Recognition |
| Deep Learning | Flame | Accuracy is 99.9% | Image Segmentation | Deep Learning and Transformer Approaches for UAV-Based Wildfire Detection and Segmentation |
* Note:The code and datasets in the article can be found in the above folder.
# Citation
If you find this work useful in your research, please consider cite:
```
@article{
  title={Advancing Forest Fire Detection with Computer Vision: A Comprehensive Review of Classification, Detection, and Segmentation Techniques},
  author={Zhuoyue Ding and Lei Huang and Cheng Zhang},
  booktitle ={The Visual Computer},
  month = {March},
  year={2025}
}
```

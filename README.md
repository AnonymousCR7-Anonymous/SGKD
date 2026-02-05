# SGKD

**Implementation of Saliency-guided knowledge distillation for driver-aware salient object detection**

## Abstract:
Salient object detection (SOD) isolates the most visually prominent and attention-grabbing parts of the driving scene by mimicking human visual attention to focus on important and most relevant objects for downstream perception and decision-making tasks. To inject driver attention into an object detection pipeline, we propose a saliency-guided knowledge distillation (SGKD) framework consisting of a teacher network trained to estimate driver saliency maps and a lightweight object detection model as the student. For each student-predicted bounding box, the teacher computes the attention energy inside the box and combines it with box-level features to form a compact vector, which is fed to a trainable rescoring head to predict whether the object is salient. We introduced the saliency vision Transformer (S-ViT) as a teacher network for saliency map estimation, while YOLO-Tiny is used as the student object detector. Experiments on the TrafficGaze dataset demonstrate the superior saliency prediction performance of S-ViT. Furthermore, results on TrafficGaze-SOD dataset show that SGKD significantly improves SOD accuracy, achieving an mAP\% of 67.90\%. With only 5.08M trainable parameters and 3.26 GFLOPs, the proposed SGKD framework is highly optimized for deployment on edge devices as a critical requirement for the demanding automotive sector.

## TrafficGaze-based frame sampling and salient object annotation
Please refer to ObjAnnotation.py. This code, First the 

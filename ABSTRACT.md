The authors release **BUP19: Sweet Pepper Dataset** to explore the issue of generalisability of the large differences between cropping environments by considering a fruit (sweet pepper) that is grown using different cultivars (sub-species) and in different environments (field vs glasshouse).

Note, similar **BUP19: Sweet Pepper Dataset** datasets are also available on the [DatasetNinja.com](https://datasetninja.com/):

- [BUP20: Sweet Pepper Dataset](https://datasetninja.com/bup20)


## Motivation

The field of agricultural robotics is experiencing rapid growth, driven by advancements in computer vision, machine learning, and robotics, coupled with an increasing demand in agriculture. Despite these strides, a significant gap exists between the technology available and the diverse requirements of farming, primarily due to the substantial variations in cropping environments. This underscores the urgent need for models with enhanced generalizability.

Addressing this gap, the BUP19 dataset was meticulously curated, encompassing diverse domains, cultivars, cameras, and geographic locations. The authors leverage this dataset in both individual and combined approaches to assess the detection and classification of sweet peppers in the wild. They opt for Faster-RCNN for detection, benefiting from its seamless adaptability to multitask learning through the incorporation of the Mask-RCNN framework for instance-based segmentation.

In evaluating sub-class classification, with a focus on the accuracy of correct detections, the authors achieve an impressive accuracy score of 0.9 in cross-domain evaluations. Notably, their experiments reveal that intra-environmental inference tends to be suboptimal. However, by enriching the data through a combination of diverse datasets, the authors enhance performance by introducing greater diversity into the training data.

In summary, the presentation of unique and varied datasets exemplifies the capacity of multi-task learning to improve cross-dataset generalization. Concurrently, it emphasizes the crucial role of diverse data in the efficient training and evaluation of real-world systems.

## Dataset description

The BUP19 dataset was captured in a glass house replicating a commercial setting at Campus Klein-Altendorf. Two different cultivar of sweet pepper were grown simultaneously during experiments: Mazurka (Rijk Zwaan) and Mavras (Enza Zaden). The glass house for sweet pepper cultivation was arranged into six rows of approximately 40m in length each. Data was recorded into bagfiles using an Intel RealSense D435i camera at 30fps. For recording each row was separated into
four equally spaced sections. Post processing was completed to align the depth and RGB images using the [pyrealsense23](https://github.com/IntelRealSense/librealsense) libraries. The stored depth image is a uint16 TIFF format file where 1mm is represented by each change in value.

<img src="https://github.com/dataset-ninja/bup19/assets/120389559/a2f03fc9-b4fd-40e4-ac1e-104c73c9816e" alt="image" width="1000">

<span style="font-size: smaller; font-style: italic;">Example images from the BUP19 dataset: (a) is the raw image, (b) is a colourised version of the instance masks, \(c\)-(f) are representations of the instance masks for black, green, mixed, and red, and (g) is a quantized version of the depth image for visualisation.</span>

For annotation, the glasshouse data was separated into three distinct sections: 1/3 training, 1/3 validation, and 1/3 evaluation. The separation of sections during recordings allowed for the data to be evenly split between each sub-set. Extending beyond bounding box regression, instance based masks are annotated. Annotation was completed by three individuals who annotated different images. A separate mask is included for each sub-class where zero denotes “background”, and a numbered response indicates the presence of a sweet pepper.

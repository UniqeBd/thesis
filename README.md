# Chapter 2: Methodology

Initially, the RSUD20K dataset was utilized for this research. The dataset contains Bangladeshi road scene images with comprehensive vehicle annotations. Following data collection, preprocessing tasks such as normalization, image scaling, cropping, and augmentation were carried out to prepare the data for training multiple deep learning architectures.

![Block Diagram](figures/block_diagram.png)
*Figure 2.1: Block diagram that represents overall working flow.*

Multiple deep learning models including YOLO (v8, v10, v11), CNN (ResNet18), Vision Transformer (ViT), DINOv2, and DETR were then trained and evaluated. Both object detection and classification approaches were explored to determine the most effective methodology for Bangladeshi vehicle recognition. The outcome was expressed in two ways: quantitatively (using metrics like mAP, accuracy, precision, recall) and qualitatively (through visual analysis of detection results). Model comparison was determined using comprehensive performance metrics across all architectures. Finally, comparative model performance and inference-based representation were demonstrated. Figure 2.1 depicts a block diagram of this thesis flowchart.

---

## 2.1 Data Collection

Road Scene Understanding Dataset (RSUD20K) is a comprehensive collection of Bangladeshi road scene images specifically designed for vehicle detection and classification tasks in complex urban environments. The dataset was curated to address the unique challenges present in South Asian traffic scenarios, including high vehicle density, diverse vehicle types, and complex occlusion patterns.

The primary benefit of adopting the RSUD20K dataset is that it provides high-resolution color images with detailed bounding box annotations for 13 distinct vehicle classes. Unlike generic vehicle datasets, RSUD20K focuses specifically on Bangladeshi vehicles, including unique vehicle types such as rickshaws, auto-rickshaws, human haulers, and rickshaw vans that are not commonly found in Western datasets.

The dataset is designed to concentrate on two primary tasks:
1. **Object Detection:** Localizing and classifying multiple vehicles within complex road scenes
2. **Image Classification:** Categorizing individual cropped vehicle images into specific classes

RSUD20K is based on the collection of high-quality images from various road conditions and lighting scenarios. The primary task is to execute the detection of vehicles fundamentally diverging in shape, size, and appearance. The secondary task is to evaluate classification performance on cropped vehicle instances.

The dataset contains images captured from real-world Bangladeshi road scenes with natural variations in weather, lighting, traffic density, and camera angles. These images are divided into training, validation, and testing sets with careful consideration of class balance and scene diversity.

---

## 2.2 Dataset Preview

The RSUD20K dataset contains images stored in standard JPEG format (.jpg), with corresponding annotation files in YOLO format (.txt). Each image is annotated with bounding boxes for all visible vehicles in the scene. The dataset structure is organized as follows:

**Dataset Statistics:**
- **Total Images:** 20,334
- **Total Annotations:** ~130,000 bounding boxes
- **Training Set:** 18,681 images (91.9%)
- **Validation Set:** 1,004 images (4.9%)
- **Test Set:** 649 images (3.2%)
- **Image Resolution:** Variable (typical range: 640×480 to 1920×1080)
- **Format:** YOLO format (class_id, x_center, y_center, width, height - normalized)

**Vehicle Classes (13 classes):**
The annotations include the following vehicle and object categories with their corresponding class indices:

1. **person** (Class 0): Pedestrians and people in the scene
2. **rickshaw** (Class 1): Traditional cycle rickshaws
3. **rickshaw_van** (Class 2): Larger cargo-carrying rickshaws
4. **auto_rickshaw** (Class 3): Three-wheeled motorized vehicles (CNG)
5. **truck** (Class 4): Large commercial trucks
6. **pickup_truck** (Class 5): Small pickup vehicles
7. **private_car** (Class 6): Passenger cars
8. **motorcycle** (Class 7): Two-wheeled motorized vehicles
9. **bicycle** (Class 8): Non-motorized two-wheelers
10. **bus** (Class 9): Large passenger buses
11. **micro_bus** (Class 10): Small passenger vans
12. **covered_van** (Class 11): Enclosed cargo vans
13. **human_hauler** (Class 12): Three-wheeled cargo vehicles

The dataset was preprocessed and properly structured with images and labels separated into corresponding directories. Each label file contains one line per object with the format: `<class_id> <x_center> <y_center> <width> <height>`, where all coordinates are normalized to [0, 1] range.

In Figure 2.2, a sample training image from the RSUD20K dataset is shown with bounding box annotations overlaid. The image demonstrates the complexity of Bangladeshi road scenes with multiple vehicle types, varying scales, and partial occlusions. Figure 2.3 depicts the class distribution across the training dataset.

![RSUD20K Sample](figures/rsud20k_sample.png)
*Figure 2.2: RSUD20K dataset sample with bounding box annotations.*

![Class Distribution](figures/class_distribution.png)
*Figure 2.3: Distribution of vehicle classes in the training dataset.*

---

## 2.3 Detection Results Data

The object detection models were evaluated on 18,600 test images to analyze real-world performance. Table 2.1 includes comprehensive detection statistics including per-class performance metrics.

**Table 2.1: Object Detection Statistics (18,600 images)**

| Metric | Value |
|--------|-------|
| Total Images Processed | 18,600 |
| Total Objects Detected | 134,031 |
| Average Detections per Image | 7.21 |
| Processing Time | 9.36 minutes |

**Class Distribution in Detection Results:**

| Class | Count | Percentage |
|-------|-------|------------|
| person | 32,020 | 23.89% |
| rickshaw | 30,711 | 22.91% |
| private_car | 20,123 | 15.01% |
| auto_rickshaw | 18,567 | 13.85% |
| motorcycle | 16,485 | 12.30% |
| bus | 7,152 | 5.34% |
| rickshaw_van | 2,526 | 1.88% |
| micro_bus | 2,294 | 1.71% |
| bicycle | 1,579 | 1.18% |
| truck | 1,295 | 0.97% |
| pickup_truck | 596 | 0.44% |
| human_hauler | 454 | 0.34% |
| covered_van | 229 | 0.17% |

The distribution reveals that common classes like 'person', 'rickshaw', and 'private_car' dominate the dataset, while specialized vehicles like 'covered_van' and 'human_hauler' are significantly underrepresented. This class imbalance presents challenges for model training and is addressed through data augmentation and weighted loss functions.

![Detection Distribution](figures/detection_distribution.png)
*Figure 2.4: Object detection class distribution across 18,600 test images.*

---

## 2.4 Data Pre-Processing

To aid in model training and evaluation, each image was carefully preprocessed. The preprocessing pipeline differs based on the task (object detection vs. classification) and the model architecture being used.

### 2.4.1 Object Detection Preprocessing (YOLO Models)

For YOLO-based object detection models, the following preprocessing steps were applied:

**Step 1: Image Loading and Validation**
- Load images and verify corresponding label files exist
- Validate bounding box coordinates are within [0, 1] range
- Remove corrupted or invalid images

**Step 2: Image Resizing**
- Resize images to model-specific input sizes:
  - YOLOv8/v10/v11: 640×640 (default)
  - Maintains aspect ratio with letterboxing
- Apply padding to preserve original aspect ratio

**Step 3: Normalization**
- Pixel values normalized to [0, 1] range
- RGB channel ordering maintained
- No additional statistical normalization required (YOLO handles internally)

**Step 4: Bounding Box Transformation**
- Convert YOLO format (x_center, y_center, width, height) to absolute coordinates
- Apply mosaic augmentation during training (combines 4 images)
- Adjust bounding boxes for augmented images

### 2.4.2 Classification Preprocessing (CNN, ViT, DINOv2)

For classification models, a different preprocessing approach was used since these models require cropped vehicle images rather than full scenes:

**Step 1: Object Cropping from YOLO Labels**
```python
# Convert normalized YOLO coordinates to pixel coordinates
x_center_px = x_center * image_width
y_center_px = y_center * image_height
width_px = width * image_width
height_px = height * image_height

# Calculate bounding box corners
x1 = x_center_px - width_px / 2
y1 = y_center_px - height_px / 2
x2 = x_center_px + width_px / 2
y2 = y_center_px + height_px / 2

# Crop object from image
cropped_vehicle = image.crop((x1, y1, x2, y2))
```

**Step 2: Image Resizing**
- CNN (ResNet18): 224×224 pixels
- ViT (Vision Transformer): 224×224 pixels
- DINOv2: 224×224 pixels

**Step 3: Statistical Normalization (ImageNet Statistics)**
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalized_image = (image - mean) / std
```

This normalization uses ImageNet pre-training statistics, which improves transfer learning performance for all classification models.

### 2.4.3 Augmentation

Data augmentation was applied during training to increase model robustness and generalization. The augmentation strategy differs between detection and classification tasks.

**YOLO Detection Augmentation:**
- **Mosaic Augmentation:** Combines 4 training images into one
- **Random Scaling:** Scale images between 0.5× to 1.5×
- **Random Translation:** Shift images by ±10% in x and y directions
- **Horizontal Flip:** 50% probability
- **HSV Augmentation:** Adjust hue, saturation, and value
- **Mixup:** Blend two images with alpha blending

**Classification Augmentation (Training Set Only):**
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

The augmented data ensures that models learn robust features invariant to lighting conditions, vehicle orientations, and minor occlusions. Figure 2.5 shows examples of augmented training images with various transformations applied.

![Augmentation Examples](figures/augmentation_examples.png)
*Figure 2.5: Examples of augmented images showing horizontal flip, color jitter, and scaling transformations.*

### 2.4.4 Normalization

Normalization is the process of transforming each pixel value in an image so that the overall distribution has zero mean and unit standard deviation. This process accelerates training convergence and improves model performance.

For classification models, ImageNet normalization statistics were used:

$$
\mu = \frac{1}{|N|}\sum_{p \in N} I(p)
$$
(Equation 2.1)

$$
\sigma = \sqrt{\frac{\sum_{p \in N} (I(p) - \mu)^2}{|N| - 1}}
$$
(Equation 2.2)

$$
I_{norm}(p) = \frac{I(p) - \mu}{\sigma}
$$
(Equation 2.3)

Here,
- $p$: Pixel value
- $\mu$: Mean value (channel-wise)
- $\sigma$: Standard deviation (channel-wise)
- $I(p)$: Original image intensity
- $I_{norm}(p)$: Normalized image intensity

For YOLO models, a simpler min-max normalization is applied:
$$
I_{norm} = \frac{I}{255}
$$
(Equation 2.4)

This scales pixel values from [0, 255] to [0, 1] range.

### 2.4.5 Dataset Splitting and Validation

The final preprocessed dataset was organized as follows:
- **Training Set:** 18,681 images → Used for model parameter optimization
- **Validation Set:** 1,004 images → Used for hyperparameter tuning and early stopping
- **Test Set:** 649 images → Used for final performance evaluation (kept separate until final testing)

All preprocessing transformations were applied consistently across train/val/test splits to ensure fair comparison.

---

## 2.5 Model Architectural Terms

Before describing individual model architectures, it is essential to define common components and operations used across all models in this research.

### 2.5.1 Convolutional Layer

A convolutional layer applies a set of learnable filters to an input image or feature map. Each filter convolves across the spatial dimensions (width and height) to produce a feature map. The convolution operation is defined as:

$$
Y(i, j) = \sum_{m}\sum_{n} X(i+m, j+n) \cdot K(m, n) + b
$$
(Equation 2.5)

Where:
- $X$: Input feature map
- $K$: Convolutional kernel (filter)
- $b$: Bias term
- $Y$: Output feature map

For a 2D convolutional layer with input dimensions $H \times W \times C_{in}$ and $C_{out}$ filters of size $k \times k$, the output dimensions are:

$$
H_{out} = \frac{H_{in} + 2p - k}{s} + 1, \quad W_{out} = \frac{W_{in} + 2p - k}{s} + 1
$$
(Equation 2.6)

Where $p$ is padding and $s$ is stride.

### 2.5.2 Pooling Layer

A 2D filter is slid across each layer of the feature space during the pooling procedure, and the features contained within the space that the filter covers are summarized. The output dimensions of a pooling layer for a feature space with dimensions $m_h \times m_w \times m_c$ are as follows:

$$
\left(\frac{m_h - f_r + 1}{s_t}\right) \times \left(\frac{m_w - f_r + 1}{s_t}\right) \times m_c
$$
(Equation 2.7)

Here,
- $m_h$: Height
- $m_w$: Width
- $m_c$: Channel number
- $f_r$: Filter size
- $s_t$: Length of stride

**Max Pooling** selects the largest element from the filter's feature map region. This operation provides translation invariance and reduces computational complexity while retaining the most prominent features.

![Max Pooling Operation](figures/max_pooling.png)
*Figure 2.6: Max pooling operation with 2×2 filter and stride 2.*

### 2.5.3 ReLU Function

The Rectified Linear Unit (ReLU) is the most commonly used activation function in modern deep learning. It introduces non-linearity into the network while being computationally efficient. The function is defined as:

$$
f(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
$$
(Equation 2.8)

![ReLU Function](figures/relu_function.png)
*Figure 2.7: ReLU activation function graph.*

The main advantages of ReLU over other activation functions are:
1. **Computational Efficiency:** Simple thresholding operation
2. **Sparse Activation:** Only activates neurons with positive inputs
3. **Mitigates Vanishing Gradient:** Gradient is 1 for positive inputs
4. **Biological Plausibility:** Resembles neuronal activation patterns

### 2.5.4 Sigmoid Function

A mathematical function with a distinctive "S"-shaped curve, the sigmoid function is used primarily in the output layer for binary classification tasks. The sigmoid function is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
(Equation 2.9)

The sigmoid function maps any real-valued number to the range (0, 1), making it suitable for probability estimation. In YOLO models, sigmoid is used to predict objectness scores and class probabilities.

![Sigmoid Function](figures/sigmoid_function.png)
*Figure 2.8: Sigmoid activation function graph.*

### 2.5.5 Batch Normalization

Batch normalization normalizes the inputs of each layer to have zero mean and unit variance across a mini-batch. This technique accelerates training and reduces sensitivity to initialization:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
(Equation 2.10)

$$
y_i = \gamma \hat{x}_i + \beta
$$
(Equation 2.11)

Where:
- $\mu_B$: Mini-batch mean
- $\sigma_B^2$: Mini-batch variance
- $\epsilon$: Small constant for numerical stability
- $\gamma$, $\beta$: Learnable scale and shift parameters

### 2.5.6 Dropout

Dropout is a regularization technique that randomly sets a fraction of input units to zero during training, which helps prevent overfitting:

$$
y = \begin{cases} \frac{x}{1-p} & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases}
$$
(Equation 2.12)

Where $p$ is the dropout rate (typically 0.3 to 0.5 in classification models).

---

## 2.6 YOLO (You Only Look Once) Models

YOLO is a family of real-time object detection models that frame object detection as a regression problem, predicting bounding boxes and class probabilities directly from full images in a single forward pass. Unlike two-stage detectors (R-CNN family), YOLO achieves significantly faster inference speeds while maintaining competitive accuracy.

### 2.6.1 YOLO Architecture Overview

The YOLO architecture consists of three main components:

1. **Backbone:** Feature extraction network (e.g., CSPDarknet, EfficientNet)
2. **Neck:** Feature pyramid network (FPN) that aggregates features at multiple scales
3. **Head:** Detection head that predicts bounding boxes, objectness scores, and class probabilities

The input image is divided into an $S \times S$ grid. Each grid cell predicts $B$ bounding boxes and confidence scores for those boxes. The confidence score reflects how confident the model is that the box contains an object and how accurate the bounding box is:

$$
\text{Confidence} = P(\text{Object}) \times \text{IOU}_{\text{pred}}^{\text{truth}}
$$
(Equation 2.13)

Each bounding box consists of 5 predictions: $(x, y, w, h, \text{confidence})$, where:
- $(x, y)$: Center coordinates of the box relative to grid cell
- $(w, h)$: Width and height relative to whole image
- $\text{confidence}$: Objectness score

### 2.6.2 YOLOv8 Architecture

YOLOv8, released by Ultralytics in 2023, introduced several improvements over previous versions:

**Key Features:**
- **Anchor-free design:** Eliminates the need for predefined anchor boxes
- **CSPDarknet backbone:** Efficient feature extraction with Cross-Stage Partial connections
- **PAN (Path Aggregation Network):** Enhanced feature fusion across scales
- **Decoupled head:** Separate branches for classification and localization

**Architecture Components:**
```
Input (640×640×3)
    ↓
Backbone (CSPDarknet)
    - Conv + BN + SiLU activations
    - C2f modules (CSP bottleneck with 2 convolutions)
    - SPPF (Spatial Pyramid Pooling Fast)
    ↓
Neck (PAN-FPN)
    - Top-down pathway (high-level to low-level features)
    - Bottom-up pathway (low-level to high-level features)
    - Feature fusion at P3, P4, P5 levels
    ↓
Head (Decoupled Detection Head)
    - Classification branch: Conv → Class probabilities
    - Regression branch: Conv → Bounding box coordinates
    ↓
Output: Detections (x, y, w, h, conf, class_probs)
```

**YOLOv8 Model Variants:**
| Variant | Parameters | FLOPs | Input Size | Use Case |
|---------|-----------|-------|------------|----------|
| YOLOv8n | 3.2M | 8.7G | 640×640 | Mobile/Edge devices |
| YOLOv8s | 11.2M | 28.6G | 640×640 | Balanced speed/accuracy |
| YOLOv8m | 25.9M | 78.9G | 640×640 | General purpose |
| YOLOv8l | 43.7M | 165.2G | 640×640 | High accuracy |
| YOLOv8x | 68.2M | 257.8G | 640×640 | Maximum accuracy |

### 2.6.3 YOLOv10 Architecture

YOLOv10 (2024) focused on improving training efficiency and reducing post-processing overhead:

**Key Innovations:**
- **NMS-free training:** Eliminates Non-Maximum Suppression during inference
- **Dual assignments:** One-to-many assignments during training, one-to-one during inference
- **Spatial-channel decoupled downsampling:** Reduces information loss during feature downsampling
- **Rank-guided block design:** Optimizes computational efficiency

**Improvements Over YOLOv8:**
- Faster inference (no NMS required)
- Better localization accuracy
- Reduced computational redundancy
- Improved training stability

### 2.6.4 YOLOv11 Architecture

YOLOv11 (late 2024) is the latest version, introducing state-of-the-art performance:

**Key Features:**
- **C3k2 blocks:** Enhanced CSP blocks with improved gradient flow
- **SPPF enhancement:** Improved spatial pyramid pooling
- **Optimized anchor-free head:** Better bounding box regression
- **Enhanced data augmentation:** Mosaic++, Copy-Paste++

**Performance Improvements:**
- Higher mAP@50 and mAP@50-95 scores
- Faster inference than YOLOv10
- Better small object detection
- Improved handling of occlusions

**YOLOv11 Model Comparison:**

| Variant | Parameters | mAP@50 (RSUD20K) | FPS | Latency (ms) |
|---------|-----------|------------------|-----|--------------|
| YOLOv11n | 2.6M | 72.34% | 432.09 | 2.3 |
| YOLOv11s | 9.4M | 76.21% | 298.45 | 3.4 |
| YOLOv11m | 20.1M | 79.54% | 120.77 | 8.3 |
| YOLOv11l | 25.3M | 80.12% | 89.34 | 11.2 |
| YOLOv11x | 56.9M | **81.85%** | 51.79 | 19.3 |

### 2.6.5 YOLO Loss Function

The YOLO loss function consists of three components:

1. **Localization Loss (Bounding Box Regression):**
$$
\mathcal{L}_{box} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (w_i - \hat{w}_i)^2 + (h_i - \hat{h}_i)^2 \right]
$$
(Equation 2.14)

2. **Confidence Loss (Objectness):**
$$
\mathcal{L}_{obj} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \left[ \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2 \right]
$$
(Equation 2.15)

3. **Classification Loss:**
$$
\mathcal{L}_{cls} = \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
$$
(Equation 2.16)

**Total Loss:**
$$
\mathcal{L}_{total} = \lambda_{coord} \mathcal{L}_{box} + \mathcal{L}_{obj} + \lambda_{class} \mathcal{L}_{cls}
$$
(Equation 2.17)

Where:
- $\mathbb{1}_{ij}^{obj}$: Indicator function (1 if object appears in cell $i$, box $j$)
- $\lambda_{coord}$, $\lambda_{noobj}$, $\lambda_{class}$: Loss weighting hyperparameters

### 2.6.6 YOLO Training Configuration

**Training Hyperparameters:**
```yaml
epochs: 50
batch_size: 16
img_size: 640
optimizer: AdamW
initial_lr: 0.001
final_lr: 0.0001
weight_decay: 0.0005
momentum: 0.937
warmup_epochs: 3
```

**Data Augmentation (YOLO-specific):**
- Mosaic: 1.0 (probability)
- Mixup: 0.1
- HSV-H: 0.015 (hue)
- HSV-S: 0.7 (saturation)
- HSV-V: 0.4 (value)
- Degrees: 0.0 (rotation disabled for street scenes)
- Translate: 0.1 (10% translation)
- Scale: 0.5 (50% scaling range)
- Fliplr: 0.5 (horizontal flip probability)

---

## 2.7 CNN Model (ResNet18)

Convolutional Neural Networks (CNNs) are the foundation of modern computer vision. For classification tasks on the RSUD20K dataset, a ResNet18 architecture was employed as the baseline CNN model.

### 2.7.1 ResNet18 Architecture

ResNet (Residual Network) introduced skip connections that allow gradients to flow directly through the network, solving the vanishing gradient problem in deep networks. ResNet18 has 18 layers with residual blocks.

**Architecture Overview:**
```
Input (224×224×3)
    ↓
Conv1: 7×7 conv, 64 filters, stride 2
BatchNorm + ReLU
MaxPool: 3×3, stride 2
    ↓
Layer 1: 2× Residual Blocks [64 channels]
Layer 2: 2× Residual Blocks [128 channels, downsample]
Layer 3: 2× Residual Blocks [256 channels, downsample]
Layer 4: 2× Residual Blocks [512 channels, downsample]
    ↓
Global Average Pooling
Fully Connected: 512 → 13 classes
Softmax
    ↓
Output: Class probabilities [13 classes]
```

**Residual Block Structure:**
```
Input (x)
    ↓
Conv 3×3 → BatchNorm → ReLU
    ↓
Conv 3×3 → BatchNorm
    ↓
Add shortcut (x) → ReLU
    ↓
Output
```

The residual connection is defined as:
$$
y = \mathcal{F}(x, \{W_i\}) + x
$$
(Equation 2.18)

Where $\mathcal{F}(x, \{W_i\})$ represents the residual mapping to be learned.

### 2.7.2 Custom CNN Implementation

In addition to ResNet18, a custom CNN architecture was implemented for comparison:

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 256 → 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier: 512×14×14 → 512 → 256 → 13
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
```

**Model Statistics:**
- **Total Parameters:** ~50M (ResNet18) / ~100M (Custom CNN)
- **Input Size:** 224×224×3
- **Output:** 13-class probabilities
- **Activation:** ReLU (hidden layers), Softmax (output)

### 2.7.3 CNN Training Configuration

**Hyperparameters:**
```python
optimizer = Adam
learning_rate = 1e-3
batch_size = 32
epochs = 50
loss_function = CrossEntropyLoss
scheduler = ReduceLROnPlateau (factor=0.5, patience=3)
```

**Training Process:**
1. **Forward Pass:** Compute predictions from cropped vehicle images
2. **Loss Calculation:** CrossEntropyLoss between predictions and ground truth
3. **Backward Pass:** Compute gradients via backpropagation
4. **Parameter Update:** Update weights using Adam optimizer
5. **Learning Rate Scheduling:** Reduce LR when validation loss plateaus

---

## 2.8 Vision Transformer (ViT) Model

Vision Transformers (ViT) apply the transformer architecture, originally designed for natural language processing, to image classification tasks. Unlike CNNs that process images through convolutional filters, ViT divides images into patches and processes them as sequences.

### 2.8.1 ViT Architecture

The ViT architecture consists of the following components:

**1. Patch Embedding:**
An input image $X \in \mathbb{R}^{H \times W \times C}$ is divided into $N$ patches of size $P \times P$:
$$
N = \frac{H \times W}{P^2}
$$
(Equation 2.19)

Each patch is linearly projected to a $D$-dimensional embedding:
$$
z_0 = [x_{class}; x_p^1 E; x_p^2 E; \ldots; x_p^N E] + E_{pos}
$$
(Equation 2.20)

Where:
- $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$: Patch embedding matrix
- $E_{pos} \in \mathbb{R}^{(N+1) \times D}$: Position embeddings
- $x_{class}$: Learnable class token

**2. Transformer Encoder:**
The transformer encoder consists of $L$ layers, each containing:
- **Multi-Head Self-Attention (MSA):**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
(Equation 2.21)

- **Multi-Layer Perceptron (MLP):**
$$
\text{MLP}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$
(Equation 2.22)

- **Layer Normalization and Residual Connections:**
$$
z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}
$$
(Equation 2.23)

$$
z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell
$$
(Equation 2.24)

**3. Classification Head:**
The class token at the output of the final encoder layer is passed through an MLP to produce class predictions:
$$
y = \text{LN}(z_L^0)
$$
(Equation 2.25)

![Vision Transformer Architecture](figures/vit_architecture.png)
*Figure 2.9: Vision Transformer (ViT) architecture showing patch embedding, transformer encoder, and classification head.*

### 2.8.2 ViT-Base Configuration

For this research, ViT-Base (ViT-B/16) was used with the following configuration:

**Model Specifications:**
```yaml
Model: ViT-Base/16
Patch Size: 16×16
Image Size: 224×224
Number of Patches: 196 (14×14)
Embedding Dimension: 768
Number of Layers: 12
Number of Attention Heads: 12
MLP Dimension: 3072
Parameters: ~86M
Dropout: 0.1
Attention Dropout: 0.0
```

**Architecture Summary:**
```
Input Image (224×224×3)
    ↓
Patch Embedding (196 patches of 16×16)
    + Position Embedding
    + Class Token
    ↓
Transformer Encoder (12 layers)
    - Multi-Head Self-Attention (12 heads)
    - Layer Normalization
    - MLP (768 → 3072 → 768)
    - Residual Connections
    ↓
Classification Head
    - Extract Class Token
    - Layer Normalization
    - Linear (768 → 13)
    ↓
Output: Class Probabilities (13 classes)
```

### 2.8.3 ViT Training Configuration

**Training Hyperparameters:**
```python
model = create_model('vit_base_patch16_224', 
                     pretrained=True,
                     num_classes=13)
optimizer = Adam(lr=1e-4)
batch_size = 32
epochs = 50
loss_function = CrossEntropyLoss
scheduler = StepLR(step_size=10, gamma=0.5)
```

**Transfer Learning Strategy:**
1. Load pre-trained ViT-Base weights from ImageNet-21k
2. Replace classification head (1000 classes → 13 classes)
3. Fine-tune all layers with lower learning rate
4. Apply data augmentation to prevent overfitting

**Training Process:**
- **Warmup:** First 5 epochs with LR=1e-5
- **Main Training:** 45 epochs with LR=1e-4
- **LR Decay:** Step decay every 10 epochs (γ=0.5)
- **Early Stopping:** Patience=10 epochs on validation loss

---

## 2.9 DINOv2 Model

DINOv2 (Self-Distillation with No Labels v2) is a self-supervised vision transformer trained on large-scale unlabeled image data. It learns robust visual representations without requiring manual annotations, making it highly effective for transfer learning.

### 2.9.1 DINOv2 Architecture

DINOv2 uses the Vision Transformer backbone but is trained using self-supervised learning objectives:

**Key Components:**
1. **Student Network:** ViT architecture processing augmented images
2. **Teacher Network:** Exponential Moving Average (EMA) of student weights
3. **Self-Distillation Loss:** Forces student to match teacher predictions

**Self-Supervised Training Objective:**
$$
\mathcal{L}_{DINO} = -\sum_c P_t^{(c)} \log P_s^{(c)}
$$
(Equation 2.26)

Where:
- $P_t$: Teacher network softmax predictions
- $P_s$: Student network softmax predictions
- $c$: Class index

**Teacher Update (EMA):**
$$
\theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s
$$
(Equation 2.27)

Where $\lambda$ is the momentum parameter (typically 0.996).

### 2.9.2 DINOv2 Architecture Details

**Model Specification (DINOv2-ViT-Base):**
```yaml
Backbone: ViT-Base/14
Patch Size: 14×14
Image Size: 224×224 (resized to 518×518 during inference)
Embedding Dimension: 768
Number of Layers: 12
Number of Heads: 12
Parameters: ~86M
Pre-training Dataset: LVD-142M (142 million images)
```

**Feature Extraction:**
DINOv2 produces highly discriminative features suitable for:
- Image classification (add linear classifier on top)
- Object detection (use as backbone)
- Semantic segmentation
- Image retrieval

### 2.9.3 DINOv2 Fine-tuning for RSUD20K

For classification on RSUD20K, a linear classifier was added on top of frozen DINOv2 features:

```python
# Load pre-trained DINOv2
dinov2_model = torch.hub.load('facebookresearch/dinov2', 
                               'dinov2_vitb14')

# Freeze backbone
for param in dinov2_model.parameters():
    param.requires_grad = False

# Add classification head
classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 13)
)
```

**Training Strategy:**
1. **Phase 1 (10 epochs):** Train only classifier head, freeze DINOv2 backbone
2. **Phase 2 (40 epochs):** Fine-tune last 4 transformer layers + classifier

**Hyperparameters:**
```python
optimizer = Adam
learning_rate_head = 1e-3
learning_rate_backbone = 1e-5
batch_size = 32
epochs = 50
loss_function = CrossEntropyLoss
```

---

## 2.10 DETR (Detection Transformer) Model

DETR (DEtection TRansformer) reformulates object detection as a direct set prediction problem, eliminating the need for hand-designed components like anchor generation and non-maximum suppression.

### 2.10.1 DETR Architecture

DETR consists of three main components:

**1. CNN Backbone:**
- Extracts feature maps from input images
- ResNet-50 typically used
- Output: Feature map of size $H/32 \times W/32$

**2. Transformer Encoder-Decoder:**
- **Encoder:** Processes CNN features with self-attention
- **Decoder:** Generates object predictions using learned object queries

**3. Prediction Heads:**
- **Classification Head:** Predicts class probabilities for each object query
- **Bounding Box Head:** Predicts normalized box coordinates $(x, y, w, h)$

![DETR Architecture](figures/detr_architecture.png)
*Figure 2.10: DETR architecture with CNN backbone, transformer encoder-decoder, and prediction heads.*

### 2.10.2 DETR Loss Function

DETR uses a set-based loss that performs bipartite matching between predictions and ground truth:

**1. Bipartite Matching:**
Find optimal assignment between $N$ predictions and ground truth objects:
$$
\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i}^{N} \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})
$$
(Equation 2.28)

Where $\mathcal{L}_{match}$ combines classification and bounding box losses.

**2. Hungarian Loss:**
$$
\mathcal{L}_{Hungarian}(y, \hat{y}) = \sum_{i=1}^{N} \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{box}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]
$$
(Equation 2.29)

**3. Bounding Box Loss:**
$$
\mathcal{L}_{box}(b_i, \hat{b}_i) = \lambda_{L1} \|b_i - \hat{b}_i\|_1 + \lambda_{GIoU} \mathcal{L}_{GIoU}(b_i, \hat{b}_i)
$$
(Equation 2.30)

Where:
- $\|b_i - \hat{b}_i\|_1$: L1 loss for box coordinates
- $\mathcal{L}_{GIoU}$: Generalized IoU loss

### 2.10.3 DETR Training Configuration

**Model Configuration:**
```yaml
Backbone: ResNet-50
Hidden Dimension: 256
Number of Encoder Layers: 6
Number of Decoder Layers: 6
Number of Attention Heads: 8
FFN Dimension: 2048
Number of Object Queries: 100
Dropout: 0.1
```

**Training Hyperparameters:**
```python
optimizer = AdamW
learning_rate = 1e-4
weight_decay = 1e-4
batch_size = 4 (limited by GPU memory)
epochs = 150 (DETR requires longer training)
lr_scheduler = StepLR(step_size=100, gamma=0.1)
```

**Loss Weights:**
```python
loss_weights = {
    'loss_ce': 1.0,      # Classification loss
    'loss_bbox': 5.0,    # L1 bounding box loss
    'loss_giou': 2.0,    # GIoU loss
}
```

---

## 2.11 Experimental Settings

The experimental environment and training infrastructure are critical for reproducible research. This section details the hardware, software, and training configurations used across all models.

### 2.11.1 Hardware Configuration

**Primary Training System:**
```
GPU: NVIDIA GeForce RTX 3060 (10GB VRAM)
CPU: Intel Core i7-10700K (8 cores, 16 threads)
RAM: 32GB DDR4
Storage: 1TB NVMe SSD
Operating System: Windows 10 Professional
CUDA Version: 11.8
cuDNN Version: 8.6
```

**GPU Memory Utilization:**
| Model | Batch Size | GPU Memory Usage |
|-------|-----------|------------------|
| YOLOv11n | 32 | 4.2 GB |
| YOLOv11m | 16 | 7.8 GB |
| YOLOv11x | 8 | 9.6 GB |
| ResNet18 | 32 | 3.1 GB |
| ViT-Base | 32 | 6.4 GB |
| DINOv2 | 32 | 6.8 GB |
| DETR | 4 | 9.2 GB |

### 2.11.2 Software Environment

**Deep Learning Frameworks:**
```python
PyTorch: 2.0.1
torchvision: 0.15.2
CUDA: 11.8
Python: 3.11.4
```

**Key Libraries:**
```python
ultralytics: 8.0.220  # YOLO training
timm: 0.9.7          # ViT models
transformers: 4.34.0  # DETR
numpy: 1.24.3
opencv-python: 4.8.1
matplotlib: 3.7.2
scikit-learn: 1.3.0
pandas: 2.0.3
albumentations: 1.3.1
```

**Development Environment:**
- **IDE:** Jupyter Notebook / VS Code
- **Version Control:** Git
- **Experiment Tracking:** TensorBoard / CSV logs

### 2.11.3 Training Protocols

**Common Training Settings:**
```python
# Mixed Precision Training
use_amp = True  # Automatic Mixed Precision for faster training

# Gradient Accumulation (for large models with small batch sizes)
accumulation_steps = 4

# Early Stopping
early_stopping_patience = 10
monitor_metric = 'val_loss'

# Model Checkpointing
save_best_only = True
save_frequency = 5  # Save every 5 epochs
```

**Training Time Estimates:**

| Model | Total Training Time | Time per Epoch | Total Epochs |
|-------|---------------------|----------------|--------------|
| YOLOv8n | 4.2 hours | 5 min | 50 |
| YOLOv8x | 18.6 hours | 22 min | 50 |
| YOLOv10m | 9.8 hours | 12 min | 50 |
| YOLOv11x | 21.3 hours | 26 min | 50 |
| ResNet18 | 3.1 hours | 3.7 min | 50 |
| ViT-Base | 6.8 hours | 8.2 min | 50 |
| DINOv2 | 5.4 hours | 6.5 min | 50 |
| DETR | 75-150 hours | 30 min | 150 |

**Total Training Compute:** ~300 GPU-hours across all 18 models

---

## 2.12 Evaluation Metrics

To comprehensively evaluate model performance, multiple metrics were employed for both object detection and classification tasks.

### 2.12.1 Object Detection Metrics (YOLO, DETR)

**1. Intersection over Union (IoU):**
$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}
$$
(Equation 2.31)

**2. Precision:**
$$
\text{Precision} = \frac{TP}{TP + FP}
$$
(Equation 2.32)

**3. Recall:**
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
(Equation 2.33)

**4. F1-Score:**
$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
(Equation 2.34)

**5. Average Precision (AP):**
AP is the area under the Precision-Recall curve:
$$
\text{AP} = \int_0^1 P(R) \, dR
$$
(Equation 2.35)

**6. Mean Average Precision (mAP):**
$$
\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i
$$
(Equation 2.36)

Where $N$ is the number of classes.

**mAP@50:** Mean AP at IoU threshold of 0.5
**mAP@50-95:** Mean AP averaged over IoU thresholds from 0.5 to 0.95 (step 0.05)

**7. Frames Per Second (FPS):**
$$
\text{FPS} = \frac{\text{Number of Images}}{\text{Total Inference Time}}
$$
(Equation 2.37)

### 2.12.2 Classification Metrics (CNN, ViT, DINOv2)

**1. Accuracy:**
$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}
$$
(Equation 2.38)

**2. Per-Class Precision and Recall:**
Calculated for each of the 13 vehicle classes individually.

**3. Confusion Matrix:**
An $N \times N$ matrix where entry $(i, j)$ represents the number of samples of true class $i$ predicted as class $j$.

**4. ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
For multi-class classification, macro-averaged ROC-AUC is computed:
$$
\text{ROC-AUC}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^{N} \text{AUC}_i
$$
(Equation 2.39)

**5. Inference Time:**
Average time (in milliseconds) to classify a single image:
$$
\text{Inference Time} = \frac{\text{Total Processing Time}}{\text{Number of Images}} \times 1000
$$
(Equation 2.40)

### 2.12.3 Statistical Significance Testing

To ensure that performance differences between models are statistically significant, paired t-tests were conducted on per-class mAP scores and accuracy metrics with significance level $\alpha = 0.05$.

---

## 2.13 Hyperparameter Summary

Table 2.2 summarizes the hyperparameters used for training each model architecture.

**Table 2.2: Hyperparameter values for all model architectures**

| Parameter | YOLO | ResNet18 | ViT-Base | DINOv2 | DETR |
|-----------|------|----------|----------|---------|------|
| **Input Size** | 640×640 | 224×224 | 224×224 | 224×224 | 800×1333 |
| **Batch Size** | 16 | 32 | 32 | 32 | 4 |
| **Epochs** | 50 | 50 | 50 | 50 | 150 |
| **Optimizer** | AdamW | Adam | Adam | Adam | AdamW |
| **Initial LR** | 1e-3 | 1e-3 | 1e-4 | 1e-3 | 1e-4 |
| **LR Scheduler** | Cosine | ReduceLR | StepLR | StepLR | StepLR |
| **Weight Decay** | 5e-4 | 0 | 1e-4 | 1e-4 | 1e-4 |
| **Dropout** | - | 0.5 | 0.1 | 0.3 | 0.1 |
| **Loss Function** | YOLOv11 | CE | CE | CE | Hungarian |
| **Mixed Precision** | Yes | Yes | Yes | Yes | Yes |
| **Augmentation** | Mosaic++ | Standard | Standard | Standard | DETR Aug |

**Legend:**
- **CE:** CrossEntropyLoss
- **Hungarian:** Hungarian matching loss (DETR-specific)
- **ReduceLR:** ReduceLROnPlateau
- **StepLR:** StepLR scheduler
- **Cosine:** CosineAnnealingLR

---

## 2.14 Summary of Methodology

This chapter presented a comprehensive methodology for training and evaluating multiple deep learning architectures on the RSUD20K Bangladeshi vehicle dataset. The key contributions of this methodology include:

1. **Comprehensive Data Preprocessing:** YOLO-format label parsing, object cropping, normalization, and extensive augmentation
2. **Multi-Architecture Approach:** Systematic comparison of 18 models (15 YOLO variants + 3 classification models)
3. **Standardized Training Protocol:** Consistent hyperparameters, evaluation metrics, and hardware configuration
4. **Dual-Task Evaluation:** Both object detection (YOLO, DETR) and classification (CNN, ViT, DINOv2) approaches

The next chapter (Chapter 3) presents the quantitative and qualitative results obtained from these experiments, including performance comparisons, per-class analysis, and computational efficiency metrics.

---

**Figure List for Chapter 2:**
- Figure 2.1: Overall workflow block diagram
- Figure 2.2: RSUD20K dataset sample with annotations
- Figure 2.3: Class distribution in training dataset
- Figure 2.4: Object detection class distribution (18,600 images)
- Figure 2.5: Augmentation examples
- Figure 2.6: Max pooling operation
- Figure 2.7: ReLU activation function
- Figure 2.8: Sigmoid activation function
- Figure 2.9: Vision Transformer (ViT) architecture
- Figure 2.10: DETR architecture

**Table List for Chapter 2:**
- Table 2.1: Object detection statistics (18,600 images)
- Table 2.2: Hyperparameter values for all models

---

**End of Chapter 2**

# Convolutional Neural Networks (CNNs)

Before Transformers, CNNs were the undisputed kings of Computer Vision. They remain highly relevant for image processing, edge AI, and specific time-series tasks due to their incredible computational efficiency.

---

## 1. The Core Problem with Standard MLPs on Images
If you feed a 1000x1000 pixel color image into a standard fully connected network (MLP), the first hidden layer requires $1000 	imes 1000 	imes 3 	ext{ (RGB)} = 3,000,000$ input weights *per neuron*. The network would have billions of parameters, take weeks to train, and overfit instantly. Also, MLPs have no concept of 2D spatial structure; a face in the top left looks mathematically different to them than a face in the bottom right.

## 2. Core Mechanisms of CNNs

### A. The Convolution Operation (Feature Extraction)
Instead of looking at the whole image at once, a CNN slides small "filters" or "kernels" (e.g., a 3x3 matrix) across the image.
*   **Dot Product:** At each step, it performs a dot product between the filter and the pixel values it is hovering over.
*   **Parameter Sharing:** The exact same 3x3 filter slides across the *entire* image. This drastically reduces the number of parameters.
*   **Translation Invariance:** Because the same filter looks everywhere, if a filter learns to detect a "horizontal edge," it will find that edge whether it's in the top-left or bottom-right of the image.

### B. Non-Linearity (ReLU)
After the convolution operation, the output (Feature Map) is passed through a ReLU activation function to introduce non-linearity, turning negative pixel calculations to zero.

### C. Pooling (Downsampling)
Pooling reduces the spatial dimensions (width and height) of the feature map, reducing computation and making the network more robust to slight distortions or shifts in the image.
*   **Max Pooling:** Slides a window (e.g., 2x2) and keeps only the maximum value in that window. (Most common).
*   **Average Pooling:** Keeps the average value.

## 3. The Architecture Hierarchy
A standard CNN follows a strict pattern:
1.  **Early Layers:** Extract low-level features (edges, corners, simple textures).
2.  **Middle Layers:** Combine edges to form shapes (circles, squares, parts of a face).
3.  **Deep Layers:** Combine shapes to form high-level semantic concepts (a whole face, a dog, a car).
4.  **Flattening:** The 3D feature maps are flattened into a 1D vector.
5.  **Fully Connected (Dense) Layers:** The flattened vector is passed through standard MLP layers to make the final classification decision (Softmax).

## 4. Famous Architectures (History & Milestones)
*   **ResNet (2015):** The most important CNN breakthrough. Introduced **Skip Connections (Residual Blocks)**. Before ResNet, networks deeper than 20 layers suffered from vanishing gradients. Skip connections allowed researchers to train networks with 150+ layers by allowing gradients a "shortcut" to flow backward unaltered.
*   **MobileNet:** Uses Depthwise Separable Convolutions to drastically reduce parameters and compute, designed specifically for running on mobile phones and edge devices.
*   **YOLO (You Only Look Once):** The state-of-the-art for real-time Object Detection. Instead of running a classifier thousands of times across an image, YOLO frames object detection as a single regression problem, predicting bounding boxes and class probabilities in one incredibly fast forward pass.

## 5. CNNs vs. Vision Transformers (ViTs)
*   **The Trend:** Vision Transformers (cutting images into patches and using Self-Attention) are overtaking CNNs on massive datasets.
*   **The Tradeoff:** ViTs lack the "inductive bias" of CNNs (the innate assumption that pixels close to each other are related). Therefore, ViTs require astronomically more data to train from scratch. For small-to-medium datasets or edge devices, CNNs (like ResNet) are still superior and much faster to train.
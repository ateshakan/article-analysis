# Summary
The paper highlights that traditional AU detection methods used for macro-expressions are not suitable for micro-expressions due to their subtlety. Instead, the authors propose utilizing temporal changes between frames to analyze these subtle facial movements. They mention two techniques, namely motion magnification and optical flow, that can effectively extract motion information from the temporal domain. However, these techniques have limitations such as dependence on parameters and computational complexity.

To overcome these challenges, the paper introduces a new approach called Learnable Eulerian Dynamics (LED) for motion representation extraction. LED differs from Eulerian video magnification by solely extracting motion without magnifying it. The authors make the motion extraction parameters learnable by utilizing automatic differentiation alongside a linearized version of Eulerian video magnification. The motion features extracted by LED are further refined using convolutional layers. This end-to-end training process enables the method to fine-tune the features specifically for the task at hand, ultimately enhancing performance in downstream tasks.

# Related Work
## Micro-Expression Recognition:
### 1. A Comparative Study of Spontaneous Micro-expression Spotting and Recognition Methods:
[In this paper](https://ieeexplore.ieee.org/document/7851001), researchers contributes as creating the first method for spotting spontaneous MEs in long videos (by exploiting feature difference contrast). They present an advanced ME recognition framework which tested on SMIC and CASMEII spontaneous ME databases. Also they proposed the first automatic ME analysis system.
#### Method for ME spotting they proposed on this article:
![[Pasted image 20230717232624.png]]
##### Facial points tracking and block division
![[Pasted image 20230717233906.png]]

In the proposed method, the first frame of the video is used to detect the positions of two inner eye corners and a nasal spine point. These points are then tracked throughout the sequence using the Kanade-Lucas-Tomasi algorithm. The facial area is divided into equal-sized blocks, and the block structure remains fixed based on the coordinates of the three tracked points.

##### Feature Extraction
The proposed method calculates a normalized Local Binary Patterns (LBP) histogram for each block within the facial area. These histograms capture the local texture information. Then, all the histograms are concatenated to form the LBP feature vector for one frame.

In addition to LBP, the method also explores the use of optical flow-based methods. Specifically, Histogram of Optical Flow (HOOF) is calculated by obtaining the flow field for each frame, comparing it to a reference flow frame, and compiling the orientations into a histogram.

Two options are tested for the reference frame: one uses the **first frame of the input video** as the reference frame for the entire video, while the other uses the **first frame within the current time window** as the reference frame, which changes as the time window slides through the video. The experiments compare the two options and discuss their performance.

##### Feature difference (FD) analysis
The basic idea of FD analysis is as follows: for each **current frame(CF)**, its features are compared to the respective **average feature frame(AFF)** by calculating the dissimilarity of the feature vectors. By sliding a time window of N frames, this comparison is repeated for each frame excluding the first k frames from the beginning and the last k frames at the end of the video, where **TF(tail frame)** and **HF(head frame)** would exceed the video boundaries. We define $(k = 1/2 × (N − 1))$. The average feature frame (AFF) is defined as a feature vector representing the average of the features of TF and HF.
![[Pasted image 20230717235719.png]]

*Illustration of terms used in feature difference (FD) analysis. The red curve shows a rapid facial movement (e.g. an ME) which produces a large FD; the blue curve shows a slower facial movement (e.g. an ordinary FE) which produces smaller FD.*

##### Thresholding and peak detection
And then the method calculates FD values for each frame in 36 blocks, selects the M greatest block FD values, and obtains an initial difference vector. Contrast is applied to the difference vector, negative values are set to zero, and thresholding and peak detection techniques are used to identify peaks indicating frames with high intensity of rapid facial movements.

#### Method for ME recognition proposed by this paper:
![[Pasted image 20230718001053.png]]
##### Face alignment:
Face alignment is needed to minimize the differences of face sizes and face shapes across different video samples. 
- A frontal face image with neutral expression, denoted as $I_{mod}$ is chosen on model face. 
- 68 facial landmarks are detected for the first frame of each clip and denoted as $I_{i,1}$. 
- To normalize variations caused by different subjects and movements, image transformation functions are employed to establish correspondence between the landmarks of $I_{i,1}$ and those of $I_{mod}$
- All frames of the ME clip are normalized
- Face area cropped from normalized images
This alignment and normalization process ensures that the frames of each ME clip are registered to the model face, reducing variations in face size and shape across different videos. The resulting normalized images, with cropped face areas, are used for subsequent ME recognition tasks.

##### Motion magnification:
The paper proposes using the Eulerian video magnification method to amplify the subtle motions in videos.
![[Pasted image 20230718003341.png]]
##### Temporal interpolation model
In ME recognition another challenge comes from the short durations of micro-expressions, especially when the videos are recorded at low frame rates. For instance, with a standard recording speed of 25 frames per second (fps), some micro-expressions may **only last for four to five frames.

To address this difficulty, the paper proposes the use of the Temporal Interpolation Model (TIM) introduced by [Zhou et al (Towards a practical lipreading system)](https://www.researchgate.net/publication/221361770_Towards_a_practical_lipreading_system). The TIM method utilizes a path graph to capture the structure of a sequence of frames. It learns a sequence-specific mapping that connects frames and embeds a curve within the path graph. This curve, governed by a single variable t ranging from 0 to 1, represents the temporal relations between frames. It characterizes unseen frames within the continuous process of a micro-expression.

By controlling the variable t at different time points, the TIM method allows for the interpolation of images at arbitrary time positions, even with a small number of input frames. In this work, TIM is used to interpolate all micro-expression sequences to the same length, such as 10, 20, 30 frames, and so on. This serves two purposes: firstly, it upsamples sequences with too few frames, and secondly, it provides a unified clip length, which can lead to more stable performance of feature descriptors.

##### Feature extraction
Several spatial-temporal local texture descriptors have been demonstrated to be effective in tackling the FE recognition problem. In this paper they compared 3 kinds of features in their ME recognition framework.
**- LBP on three orthogonal planes:**
	A video sequence can be thought as a stack of XY planes on T dimension, as well as a stack of XT planes on Y dimension, or a stack of YT planes on X dimension. The XT and YT plane textures can provide information about the dynamic process of the motion transitions. Figure 7(a) presents the textures of XY, XT and YT plane around the mouth corner of one ME clip. *[Dynamic Texture Recognition Using Local Binary Patterns with an Application to Facial Expressions](https://www.researchgate.net/publication/6397809_Dynamic_Texture_Recognition_Using_Local_Binary_Patterns_with_an_Application_to_Facial_Expressions)*
![[Pasted image 20230718005727.png]]

- Histogram of Oriented Gradients (HOG) and Histogram of Intensity Gradient Orientations (HIGO): HOG is a well-known descriptor for capturing gradient orientations in images. HIGO is an extension of HOG that takes into account variations in illumination or contrast. In this work, the 3D spatial-temporal versions of HOG and HIGO are obtained by extending the descriptors from the XY plane to the three orthogonal planes (XY, XT, and YT). Similar to LBP-TOP, histograms are computed from each block/cuboid in the three planes. The histograms from each plane are normalized using local L1 normalization, and the normalized histograms are concatenated to form the final descriptor.

#### Classification
To determine the best parameters for the LSVM, a five-fold cross-validation is performed on the training data. The parameter search is conducted in the range of $10^-1, 1, 2, 10, ..., 10^3$. During the cross-validation, various parameter settings are tested, and the one that yields the best performance is selected for further analysis.

### 2. Two stream Difference Network:
More recently, motion magnification has been used with neural networks. Samples are first magnified before going through a neural network. TSDN ([Two Stream Difference Network](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12030)) performs frame difference between the **onset** and **apex** frame in the hidden representations of an autoencoder. The frame differences are then fed through a neural network for predictions.
![[Pasted image 20230718015253.png]]
*Framework of two‐stream difference network method.*
- Pre‐processing module: it mainly performs face detection, face alignment, and spotting apex frames on the image sequence 
- Two‐stream encoder‐decoder network: encode and decoder the onset apex frame by the identity and micro‐ expression stream through CNN
- Differential network: input the feature‐level difference of the middle layer to CNN for recognising micro‐expression. 
- The micro‐expression image is the **apex** image, and the identity image is the **onset** image in the micro‐expression sequence.

## Action Unit Detection
Most previous work on Action Unit (AU) detection has primarily focused on macro-expressions, where facial morphology observed from static frames provides valuable information. However, with Micro-Expressions (MEs), the facial movements are subtle and require the **analysis of dynamic information across frames**.


# Methodology
In LED, instead of adding the filtered motion to the original image as in traditional motion magnification, only the motion is extracted and used directly for downstream tasks. The motion features can be extracted without learning, but LED introduces learnable parameters in the motion filter to fine-tune the features in an end-to-end manner. This allows LED to learn task-specific features that enhance performance compared to non-learnable motion extraction methods. The learnability is achieved through **automatic differentiation** and using a linearized version of EVM.

To address the issue of non-homogeneous luminance intensity, LED incorporates a normalization scheme. This normalization helps to alleviate inconsistencies in luminance across different frames.

After the motion features have been extracted, they are passed through a Convolutional Neural Network (CNN) for further refinement. The CNN helps to enhance and refine the features extracted by LED.

**LED is designed to be a model-agnostic approach that can be integrated at the beginning of a network. It provides a foundation for ME analysis and can be used as a starting point for subsequent tasks.**

## Motion Representation
![[Pasted image 20230718035942.png]]
As you can see on onset frame and apex frame the difference is extremely small. Therefore, the facial dynamics between frames offers better representation to find subtle differences.  The commonly used methods for motion extraction in ME recognition are optical flow and motion magnification. However, these methods suffer from a separation between the motion representation and the learned features, which prevents an end-to-end approach for fine-tuning the motion representation.

This separation poses a challenge because it limits the ability to optimize and refine the motion representation together with the rest of the network in an end-to-end manner. In other words, the motion extraction process and the subsequent feature learning process are not integrated seamlessly, making it difficult to jointly optimize and fine-tune the motion representation based on the specific task requirements.

In LED, the parameters of the motion extraction process are made learnable, meaning they can be adjusted and optimized during training. This allows the motion representation to be fine-tuned alongside the rest of the network, enabling the model to learn task-specific features that enhance performance in Micro-Expression (ME) recognition.

By integrating the motion extraction process with the overall feature learning process in an end-to-end fashion, LED ensures that the motion representation is optimized specifically for the ME recognition task. This leads to improved performance as the model can effectively capture and utilize the subtle motion dynamics present in MEs.
![[Pasted image 20230718051157.png]]

### Motion Extraction
EVM, first spatially decomposes the frames then uses a temporal bandpass filter to extract the motion information $B_{xy}^t$ , where x and y refer to the coordinates and t to time. It is found that adding the extracted motion $B_{xy}^t$ is unnecessary for detecting AUs. An issue with the filtering approach is that the user must set the hyperparameters of the filter: amplification factor $a$ and the cutoff frequencies r1 and r2. These values depend highly on the magnitude and amount of the input's temporal changes.  

All these parameters could be learned by automatic differentiation with a neural network but EVM algorithm is computationally expensive and this makes it untrainable alongside a neural network. 

To solve these issues, the proposed LED method, involves three techniques: linearization, learnable parameters, and frame difference normalization:

#### Linearization:
EVM algorithm can be sped up by linearizing it using a matrix form. In [A Boost in Revealing Subtle Facial Expressions: A Consolidated Eulerian Framework](https://doi.org/10.1109/FG.2019.8756541) the authors linearize the EVM, in our paper the authors change $w_2$ and  $w_1$ to $r_1$ and $r_2$ 
```python
def calculate_W(T, alpha=20, r1=0.4, r2=0.05):
    W = torch.zeros(T, T, dtype=torch.float).to(device)
    #construct W
    for i in range(T):
        for j in range(T):
            a = j - i
            b = min(1, i)
            if j > i:
                W[i, j] = alpha * (1 - r1) ** a * r1 ** b - alpha * (1 - r2) ** a * r2 ** b
            elif j == i:
                W[i, j] = alpha * (r1 - r2)
    return W
```
![[Pasted image 20230718063000.png]]


![[Pasted image 20230718063047.png]]
*This is the linearization on the original paper*

The reason they modified the original version because they are only interested in the motion, not amplification of it. Here $r_1$ and $r2$ are the bandpass values, a is the motion magnification factor, $a= j - i$, $b = min(l, i - 1)$ and $i = j$ corresponding to the number of frames being used. This drastically reduces the computational load compared to EVM, as the motion representation can now be extracted with a simple tensor contraction.

$$  B_{xy}^t = \left( \sum_{i=0}^t I^i_{xy}W_i^t \right)$$

#### Learnable Parameters
After deriving matrix $W$ matrix now it can be used as a parameter in a neural network that employs automatic differentiation. Instead of directly learning the values of the parameters , we learn the logarithm of each parameter. This ensures that the learned parameter values are always positive.

By incorporating the optimization process as the first layer, the entire network becomes end-to-end. This means that the motion representation and the subsequent convolutional features are learned simultaneously. This joint learning allows the motion representation to be directly learned from the data, which is different from methods like optical flow and motion magnification.

#### Frame Difference Normalization
![[Pasted image 20230718161411.png]]
With linearized formulation we can also think of movement extraction as weighted frame difference. This allows us to apply frame difference normalization which proposed on [DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks]([[1805.07888] DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks (arxiv.org)](https://arxiv.org/abs/1805.07888))
The authors observe an issue when taking differences between frames due to the assumption of a homogeneous stationary luminance intensity. As a result of uneven skin contours and different distances to the light source the luminance intensity has different spatial distribution with different setups. To remove the effect from the difference $B^t_{xy}$ , it is divided by the temporal mean. We generalize the formula for the weighted frame difference as
![[Pasted image 20230718162651.png]]
where Wi are the weights for each frame.

![[Pasted image 20230718162844.png]]
*It can be seen that the learned parameters provide a cleaner representation and the normalization removes some noisy pixels that do not provide value to the classification*

#### Network Architecture
The proposed network architecture based on SSSNet from [Micro-expression recognition with noisy labels](https://oulurepo.oulu.fi/handle/10024/32856) due to it's simplicity and performance of the network architecture. These types of architectures are better for this purpose because of the lack of real world data.
![[Pasted image 20230718163447.png]]

This architecture is modified on the proposed paper:
- To prevent overfitting, dropout layers are added between the hidden layers of the network. Despite its small size, the network still has a tendency to overfit the data.

- Since the input now consists of a sequence of frames instead of a single frame, modifications are made to the network architecture. Two-dimensional convolution, pooling, batch normalization, and dropout operations are replaced with their three-dimensional counterparts to account for the temporal dimension of the input data.

- Given the increased complexity of the task, the number of channels is also increased. The first convolutional layer has 32 channels, the second convolutional layer has 64 channels, and the linear layer has 256 channels.
- ![[Pasted image 20230718165728.png]]
```python
class Net(nn.Module):
    def __init__(self, task_num, dropout=0.5):
        super().__init__()
        self.task_num = task_num
        h1 = 32
        h2 = 64
        h3 = 256
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=h1, kernel_size=(1, 5, 5), stride=1)
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(h1)
        self.drop1 = nn.Dropout3d(dropout)
        
        self.conv2 = nn.Conv3d(in_channels=h1, out_channels=h2, kernel_size=(2, 3, 3), stride=1)
        self.bn2 = nn.BatchNorm3d(h2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.drop2 = nn.Dropout3d(dropout)

        self.fc1 = nn.Linear(9 ** 2 * 2 * h2, h3)
        self.fcs = nn.ModuleList([nn.Linear(h3, 2) for _ in range(self.task_num)])
        self.drop3 = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.log(torch.tensor(10.0)))
        self.r1 = nn.Parameter(torch.log(torch.tensor(0.4)))
        self.r2 = nn.Parameter(torch.log(torch.tensor(0.05)))
```
### Experiments
#### Dataset
CASME II is an updated version of CASME, a dataset used for Micro-Expression (ME) recognition. CASME II offers improvements over its predecessor by increasing the frame rate from 60 frames per second (fps) to 200 fps, providing more detailed temporal information. It consists of a total of 256 samples collected from 26 different subjects.

In CASME II, a total of 19 different Action Units (AUs) have been annotated. However, for the purposes of ME recognition, a subset of eight AUs is commonly used.

These AUs include AU1 (inner brow raiser), AU2 (outer brow raiser), AU4 (brow lowerer), AU7 (lid tightener), AU12 (lip corner puller), AU14 (dimpler), AU15 (lip corner depressor), and AU17 (chin raiser).

SAMM is another dataset used for ME recognition. It contains a total of 159 samples collected from 29 different subjects. In SAMM, a total of 27 different AUs have been annotated. However, for ME recognition purposes, a subset of four AUs is commonly used. These AUs include AU2 (outer brow raiser), AU4 (brow lowerer), AU7 (lid tightener), and AU12 (lip corner puller)

#### Metrics
To address the limited number of samples in ME recognition and account for the imbalance in class distribution, the [LOSO](https://ai.plainenglish.io/leave-one-subject-out-cross-validation-for-machine-learning-model-557a09c7891d) (leave-one-subject-out) protocol is commonly employed. LOSO is a widely used evaluation method in which each subject in the dataset is treated as the testing data, while the remaining subjects are used as the training data.

The LOSO protocol helps to evaluate the generalization performance of the ME recognition model across different subjects by ensuring that the testing data does not overlap with the training data.

To mitigate the issue of class imbalance, the macro F1-score is often used as an evaluation metric instead of accuracy. The macro F1-score considers the performance of each class individually and calculates the average F1-score across all classes. This helps to provide a more balanced assessment of the model's performance, taking into account the varying class frequencies in the dataset
![[Pasted image 20230718171859.png]]

```python
class MultiTaskF1(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskF1, self).__init__()
        self.task_num = task_num
        
    def calc_f1(self, label, prediction):
        _, predicted = torch.max(prediction, 1)
        f1 = f1_score(label.cpu(), predicted.detach().cpu(), average="macro")
        return f1
                        
    def forward(self, preds, labels):
        f1s = [self.calc_f1(labels[:, i], preds[i]) for i in range(self.task_num)]
        return f1s
```

#### Training settings
- **Motion Magnification Parameters**: The parameters for motion magnification (a technique used in LED) are set to a = 10, r1 = 0.4, and r2 = 0.05. These values have been frequently used and shown good performance in prior research.
- **Training Setup**: The training process consists of 400 epochs using the Adam optimizer with a learning rate of 0.0001 and weight decay of 0.001. The LED parameters (a, r1, r2) have a separate learning rate of 0.1, which is changed to 0.01 after 50 epochs. This learning rate scheduling helps stabilize the training process.
- **Ablation Studies**: In ablation studies, when the log transform is not applied, a lower learning rate of 0.01 is used for the cutoff frequencies.
- **Data Augmentation**: Temporal augmentation is applied to augment the data. Spatial augmentation strategies such as cropping, rotations, and color jitter are also experimented with, but no performance improvement is observed.
- **Frame Size**: The frame size is set to 64 x 64. Larger frame sizes (e.g., 128 x 128) show similar results, while smaller frame sizes (e.g., 32 x 32) yield lower performance.
- **Number of Frames**: LED utilizes five frames for its computations.
- **ResNet Backbone**: Among ResNet 18 and 34 models, the ResNet 18 model is chosen as it achieves similar results to ResNet 34 but with a smaller network size.

#### Ablation Studies
![[Pasted image 20230718193829.png]]
Ablation studies refer to conducting experiments to analyze the impact of specific components or techniques by removing or altering them. By adding normalization we can observe an increase of around one percentage point. By incorporating learning into the process, a similar performance increase can be observed. However, when the log transform is not used for the parameters, the values of r1 and r2 become more unstable and often result in negative values. By applying the log transformation, we can observe faster convergence and more stable values for the parameters, eliminating duplicate solutions represented by negative values. This leads to an improvement in performance, as shown in the last row of the table.

![[Pasted image 20230718193026.png]]
- **RGB**: Initial experiment using raw RGB videos shows limited performance, with only AU4 being identified effectively. (*Subtle changes between RGB frames are challenging to learn with limited data and a small CNN.*)
- **MM**: Motion magnification is applied, resulting in significant performance improvement, increasing the average F1-score from 55.3 to 71.7. (*Motion magnification helps identify even the difficult AU7*)
- **FD**: Simple frame difference, using the difference between consecutive frames, shows a slight performance increase compared to motion magnification.
- **LED**: LED method, with automatically learned parameters in an end-to-end fashion, outperforms other approaches, indicating the effectiveness of learning the motion representation from the data.

##### Number of Frames
The best performance was achieved with 5 frames, but there were negligible differences when using 8 or 12 frames. This suggests that many of the frames are redundant when using LED, but relying on a single frame may not provide sufficient information. It's important to note that these findings may not fully generalize to other methods, and LED's preference for a lower number of frames could be attributed to its limitations in learning more complex patterns.
![[Pasted image 20230718200521.png]]
##### Learnability
![[Pasted image 20230718200643.png]]
This table illustrate how the parameters change during the training process on the CASMEII dataset. Specifically, we observe a significant decrease in the magnification value, indicating that it is not necessary for the motion extraction approach. However, it still serves as a scaling factor and takes on different values across datasets and subjects. We notice that the values tend to converge around 100 frames, suggesting that a narrow frequency band is sufficient for the network.

#### Comparison to State of Art


![[Pasted image 20230718201615.png]]
CASME II Results:
- LED outperformed previous methods for detecting action units in CASME II.
- The average F1-Score improved by around 11 compared to the DVASP method when using the SSSNet backbone.
- The LED method also showed promising results with the Resnet18 backbone, indicating its generalizability to other network architectures.

SAMM Results:
- SAMM dataset was observed to be more challenging compared to CASME II.
- LED method achieved better performance than other methods, except for the SCA method in detecting AU12.
- On average, LED achieved an increase of around 9 F1-Score compared to the DVASP method.

CASME Results:
- Results for CASME dataset were not explicitly discussed in the provided information but can be found in the supplementary material.
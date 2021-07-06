# Attention O-Net

**An O-shape neural network alchitecture for junction detection in biomedical iamges.**

<br> 

## Attention O-Net Architecture:
>In this work, we propose an O-shape Network architecture with Attention modules (Attention O-Net) to efficiently detect junctions in original biomedical images without segmentation. Based on the encoder-decoder network, the Attention O-Net has two branches for different tasks. By regressing junction heatmaps, the junction locations are estimated in Junction Detection Branch (JDB), and then the junction coordinates are extracted by choosing the positions with the local highest value. Aiming at the problems that the original biomedical images (especially retinal images) are with extremely low contrast and heavy background noise in the regions with thin branches, a Local Enhancement Branch (LEB) is proposed to enhance the foreground of thin junction regions. The pixels in thin and thick branches are strongly unbalanced, resulting in different heatmap responses in different contrast regions (thin branches usually exhibit low contrast). Therefore, radius-adaptive labels for training LEB are designed to enhance thin branches and reduce the unbalanced heatmap responses of thin and thick junction regions. In addition, the attention module is used to introduce the feature maps from LEB to JDB. It is worth noting that we do not need to segment images during the testing phase. In other words, in this method, an original biomedical image is input, and a heatmap indicating the locations of the junction points is developed. In this sense, the overall architecture is trained in an end-to-end way.

### The Overall Diagram of Attention O-Net Architecture:

![Overall_Diagram](https://github.com/zyqhnu/Images_for_Attention_O-Net/blob/main/fig2.jpg)


### The Detailed Architecture of Attention O-Net:


![detialed_AONet](https://github.com/zyqhnu/Images_for_Attention_O-Net/blob/main/detialed_A_ONet.jpg)

<br> 


## Requirements:

* Tensorflow  == 1.3

* numpy == 1.16

<br> 


## Citation

Accepted by IEEE Journal of Biomedical Health and Informatics (J-BHI).

The full source code have been released at 2021/06/28.

Y. Zhang, M. Liu, F. Yu, T. Zeng and Y. Wang, "An O-shape Neural Network With Attention Modules to Detect Junctions in Biomedical Images Without Segmentation," in IEEE Journal of Biomedical and Health Informatics, doi: 10.1109/JBHI.2021.3094187.

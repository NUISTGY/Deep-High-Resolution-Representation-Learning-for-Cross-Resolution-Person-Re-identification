<div align="center">
  
# Deep-High-Resolution-Representation-Learning-forCross-Resolution-Person-Re-identification
Journal of IEEE Transactions on Multimedia (Under review)

</div>

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction

Person re-identification (re-ID) tackles the problem of matching person images with the same identity from different cameras. In practical applications, due to the differences in camera performance and distance between cameras and persons of interest, captured person images usually have various resolutions. We name this problem as Cross-Resolution Person Re-identification which brings a great challenge for matching correctly. In this paper, we propose a Deep High-Resolution Pseudo-Siamese Framework (PS-HRNet) to solve the above problem. Specifically, in order to restore the resolution of low-resolution images and make reasonable use of different channel information of feature maps, we introduce and innovate VDSR module with channel attention (CA) mechanism, named as VDSR-CA. Then we reform the HRNet by designing a novel representation head to extract discriminating features, named as HRNet-ReID. In addition, a pseudo-siamese framework is constructed to reduce the difference of feature distributions between low-resolution images and high-resolution images. The experimental results on five cross-resolution person datasets verify the effectiveness of our proposed approach. Compared with the state-of-the-art methods, our proposed PS-HRNet improves 3.4%, 6.2%, 2.5%,1.1% and 4.2% at Rank-1 on MLR-Market-1501, MLR-CUHK03, MLR-VIPeR, MLR-DukeMTMC-reID, and CAVIAR datasets, respectively.

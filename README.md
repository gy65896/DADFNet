 # <p align=center> [AAAIW 2022] DADFNet: Dual Attention and Dual Frequency-Guided Dehazing Network for Video-Empowered Intelligent Transportation</p>

<div align="center">
 
[![Paper](https://img.shields.io/badge/DADFNet-Paper-red.svg)](https://arxiv.org/abs/2304.09588)
</div>

---
>**OneRestore: A Universal Restoration Framework for Composite Degradation**<br>  [Yu Guo](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN)<sup>†</sup>, [Ryan Wen Liu](http://mipc.whut.edu.cn/index.html)<sup>* </sup>, Jiangtian Nie, Lingjuan Lyu, Zehui Xiong, Jiawen Kang, Han Yu, Dusit Niyato <br>
(* Corresponding Author)<br>
>AAAI Workshop: AI for Transportation

> **Abstract:** *Visual surveillance technology is an indispensable functional component of advanced traffic management systems. It has been applied to perform traffic supervision tasks, such as object detection, tracking and recognition. However, adverse weather conditions, e.g., fog, haze and mist, pose severe challenges for video-based transportation surveillance. To eliminate the influences of adverse weather conditions, we propose a dual attention and dual frequency-guided dehazing network (termed DADFNet) for real-time visibility enhancement. It consists of a dual attention module (DAM) and a high-low frequency-guided sub-net (HLFN) to jointly consider the attention and frequency mapping to guide haze-free scene reconstruction. Extensive experiments on both synthetic and real-world images demonstrate the superiority of DADFNet over state-of-the-art methods in terms of visibility enhancement and improvement in detection accuracy. Furthermore, DADFNet only takes $6.3$ ms to process a $1,920 \times 1,080$ image on the $2080$ Ti GPU, making it highly efficient for deployment in intelligent transportation systems.*
---

## Requirement ##
* __Python__ == 3.7
* __Pytorch__ == 1.9.1

## Flowchart of Our Proposed Method

We refer to this network as dual attention and dual frequency-guided dehazing network (DADFNet). The framework of our proposed DADFNet is shown in Fig. 1. In particular, this network mainly consists of two parts, named dual attention module (DAM) and high-low frequency-guided sub-net (HLFN). 

![Figure02_Flowchart](https://user-images.githubusercontent.com/48637474/158503605-3200f3dd-ecec-4404-8ee5-04b404a30f66.png)
**The architecture of our proposed dual attention and dual frequency-guided dehazing network (DADFNet). The DADFNet mainly consists of two parts, i.e., dual attention module (DAM) and high-low frequency-guided sub-net (HLFN). Note that LReLU denotes the leaky rectified linear unit function.**

## Test
This code contains two modes, i.e., nonhomogeneous dehazing (not stated in the article) and normal dehazing.
### Normal Dehazing
* Put the hazy image in the "input" folder
* Run "test_real.py". 
* The enhancement result will be saved in the "output" folder.

### Nonhomogeneous Dehazing
* Put the hazy image in the "hazy" folder
* Run "test_real_nonhomogeneous_dehazing.py". 
* The enhancement result will be saved in the "output" folder.

## Citation

```
@article{guo2023dadfnet,
  title={DADFNet: Dual attention and dual frequency-guided dehazing network for video-empowered intelligent transportation},
  author={Guo, Yu and Liu, Ryan Wen and Nie, Jiangtian and Lyu, Lingjuan and Xiong, Zehui and Kang, Jiawen and Yu, Han and Niyato, Dusit},
  journal={arXiv preprint arXiv:2304.09588},
  year={2023}
}
```

#### If you have any questions, please get in touch with me (guoyu65896@gmail.com).

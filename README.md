# Local Reproduction Notes

This fork includes a minimal MTL-AQA score-regression baseline and skeleton-input replacement experiments in addition to the official release files.

Local project documentation has been moved to [docs/](docs/):

- [Project progress summary](docs/PROJECT_PROGRESS_SUMMARY.md)
- [Supervisor report](docs/SUPERVISOR_REPORT.md)
- [Architecture learning guide](docs/PROJECT_ARCHITECTURE_LEARNING_GUIDE.md)
- [Skeleton baseline notes](docs/README_skeleton_baseline.md)

## 📢 Announcement: Workshop on Action Quality Assessment at ICCV 2025! 📢
**April 2025** 📢 📢 📢 We will be organizing **Workshop** on [*Skilled Activity Understanding, Assessment and Feedback Generation* (SAUAFG)](https://sauafg-workshop.github.io/) at **ICCV 2025!** More info on [SAUAFG Website](https://sauafg-workshop.github.io/). **Consider submitting your papers!** See you in Honolulu, Hawaii!

# MTL-AQA
[What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment](https://arxiv.org/abs/1904.04346)

## MTL-AQA Concept:

<p align="center"> <img src="diving_sample.gif?raw=true" alt="diving_video" width="200"/> </p>
<p align="center"> <img src="mtlaqa_concept.png?raw=true" alt="mtl_net" width="400"/> </p>

If you want to train **Pose Estimators** and **Object Detectors** for **AQA/Diving**, we provide that data: [here](https://github.com/laurenok24/NSAQA)

This repository contains MTL-AQA dataset + code introduced in the above paper. If you find this dataset or code useful, please consider citing:
```
@inproceedings{mtlaqa,
  title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
  author={Parmar, Paritosh and Tran Morris, Brendan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={304--313},
  year={2019}
}
```


## 🚀 Also Check Out Our New Approach! 🚀
**Oct 2024:** We have developed a new approach, [NeuroSymbolic AQA](https://arxiv.org/abs/2403.13798), that builds upon this approach, but also analyses and scores using **Professional Rules-based** programs. It is *Comprehensive* and *Explainable* AQA which can generate **Full Performance Reports** for *Actionable* Insights!!! We encourage you to checkout [[Code, Rules-based Programs, Dataset](https://github.com/laurenok24/NSAQA)] [[Demo](https://huggingface.co/spaces/X-NS/NSAQA)] [[Full Paper](https://arxiv.org/abs/2403.13798)]

You are welcome to continue using this project, as it will still be maintained alongside the new approach!

## Check out our other relevant works:

[Fine-grained Exercise Action Quality Assessment](https://github.com/ParitoshParmar/Fitness-AQA): Self-Supervised Pose-Motion Contrastive Approaches for Fine-grained Action Quality Assessment (can be used for Diving as well!) + Fitness-AQA dataset

<b>***</b> <i>Want to know the score of a Dive at the ongoing Olympics, even before the judges' decision?</i> <b>Try out our [AI Olympics Judge](https://share.streamlit.io/gitskim/aqa_streamlit/main/main.py) ***</b>

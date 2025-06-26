# MER 2025 Personality Baseline 

<a href="https://zeroqiaoba.github.io/MER2025-website/">MER25 HOMEPAGE</a> ||  <a href="https://huggingface.co/datasets/MERChallenge/MER2025">MER25 Huggingface</a> 

## Data Download 
<a href="https://huggingface.co/datasets/MDPEdataset/MDPE_Dataset/">Train & Val Data</a> ||  <a href="https://huggingface.co/datasets/MDPEdataset/MER2025_personality/">Test Data</a> ||  <a href="https://codalab.lisn.upsaclay.fr/competitions/23185">Submission Link</a> 

## Introduction

MER2025_personality is the MER25 Challenge @ ACM MM & MRAC25 Workshop @ ACM MM Emotion-enhanced Personality Recognition Track4.

## Requirements

```
pip install -r requirements.txt
```

## Run: Sample 


```
python main.py --feature_set baichuan13B-base  --epochs 200 --batch_size 128 --lr 0.0001 
```

## Suggestion
We suggest that participants can first improve from the following perspectives:

1. Extract richer features.
2. Use more robust model structures. (including pre-trained models, large models, etc.)
3. Use emotional label.



## Citation
For more details about MDPE, please refer to:
[MDPE: A Multimodal Deception Dataset with Personality and Emotional Characteristics](https://arxiv.org/abs/2407.12274)

Please cite our paper if you find our work useful for your research:

```
@article{cai2024mdpe,
  title={MDPE: A Multimodal Deception Dataset with Personality and Emotional Characteristics},
  author={Cai, Cong and Liang, Shan and Liu, Xuefei and Zhu, Kang and Wen, Zhengqi and Tao, Jianhua and Xie, Heng and Cui, Jizhou and Ma, Yiming and Cheng, Zhenhua and others},
  journal={arXiv preprint arXiv:2407.12274},
  year={2024}
}
```

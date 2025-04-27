# IERP 2024 Baseline || <a href="http://www.iscslp2024.com/emotionChallenges">HOMEPAGE</a>

## Baseline model: Linear / GRU Attention Regressor

Participants predict the subjects' eight emotions(sadness, happiness, relaxation, surprise, fear, disgust, anger, neutral) and the intensity score of each emotion (range from 1 to 5 point for each emotion representing intensity, 1 indicating no such emotion, and 5 indicating the strongest emotion);

## Run: Sample 


```
python main.py --feature_set baichuan13B-base --fea_dim 5120 --epochs 200 --batch_size 128 --lr 0.0001 
```
We suggest that participants can first improve from the following perspectives
1. Use personality characteristics.
2. Feature fusion (early or late fusion).
3. When using visual features, it should be considered to integrate the two feature files of "watch" and "description".

If you have any questions, you can contact us through the official email：IEPR2024@iscslp2024.com




MER2025_personality is a subset of MDPE. For more details about MDPE, please refer to [MDPE Dataset](https://huggingface.co/datasets/MDPEdataset/MDPE_Dataset)






This dataset serves as the testing set for MER25 Challenge @ ACM MM & MRAC25 Workshop @ ACM MM Emotion-enhanced Personality Recognition Track, with the MDPE as the training and validation sets. 


More details about the MER2025 competition can be found on the [MER25 Website](https://zeroqiaoba.github.io/MER2025-website/) and [MER25 Huggingface](https://huggingface.co/datasets/MERChallenge/MER2025)




The label_personality.csv remains the same as the original MDPE, except for normalization。




# MDPE Dataset
MDPE is a multimodal deception dataset. Besides deception features, it also includes individual differences information in personality and emotional expression characteristics. MDPE not only supports deception detection, but also provides conditions for tasks such as personality recognition and emotion recognition, and can even study the relationships between them. 



## Dataset Download


The data are passcode protected. Please download and send the signed [EULA](https://drive.google.com/file/d/1A1F8szMOTf9-rK8DYD23GruBArtnYdLl/view?usp=sharing) to [mdpe.contact@gmail.com](mdpe.contact@gmail.com) for access request.

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

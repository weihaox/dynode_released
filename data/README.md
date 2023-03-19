# Datasets

We use four dataset categoreis for our experiments. The latent codes are obtained by inverting each frame into $\mathcal{W^+}$ space of StyleGAN2. Other latent spaces, e.g. $\mathcal{Z}$ or $\mathcal{W}$ space, are also supported. We use [ReStyle](https://github.com/yuval-alaluf/restyle-encoder) for inversion in our experiments.

## dataset categoreis

- Face. StyleGAN2 is trained on LLHQ at the resolution of $1024\times1024$. The real face videos are collected from the publicly available talking-head datasets (e.g. ) or downloaded from YouTube. If using in-the-wild face images, a cropping-and-alignment step would be necessary before inversion. Please refer to the paper for more details of preprocessing. 

- Scene. StyleGAN2 is trained on Place365 at the resolution of $256\times256$.  We use [khrulkov2021latent](https://arxiv.org/abs/2111.14825) to generate temporally-consistent frames by editing the time-varying attributes, e.g. night, dawndusk, and sunrisesunset. We also download real videos of outdoor natural scenes from YouTube and obtain latent codes of sampled frames by direct optimization. 

- Bird. StyleGAN2 is trained on CUB-200-2011 at the resolution of $256\times256$. We collect bird videos from ouTube.

- Isaac3D. StyleGAN2 is trained on Isaac3D with a resolution of $128\times128$. The dataset contains nine factors of variation, such as background color, object shape, robot movement, or camera height. As it is a synthetic dataset, we generate consecutive frames by moving the robot or camera.

## example

An example of inverted codes can be found at [example](https://github.com/weihaox/dynode/tree/main/data/example).
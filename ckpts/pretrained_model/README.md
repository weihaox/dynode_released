# StyleGAN2 Model

We use StyleGAN2 as the pretrained model in our experiments. However, our method is not dependent on StyleGANs and can be applied to any GAN model. The code relies on the [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) pytorch implementation of StyleGAN2. Please download the pretrained model and assign it to this directory. 

The pretrained StyleGAN2 generator can be manually downloaded from [here](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) or by using the following script.


```bash
# pip install gdown

save_path_model='pretrained_models/'

# creat folder if not exist
# mkdir -p $save_path_model

echo "dowloading pretrained models..."

# download pretrained models of StyleGAN2
gdown https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT -O $save_path_model
```
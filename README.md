# LIDA
Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution, CVPR 2026, [arXiv](https://arxiv.org/abs/2603.10583)

## Dataset
We use [GenImage](https://github.com/GenImage-Dataset/GenImage) for evaluation, which can be downloaded online. GenImage is composed of 8 classes of fake images (BigGAN, Midjourney, Wukong, Stable_Diffusion_v1.4, Stable_Diffusion_v1.5, ADM, GLIDE, VQDM) and real images from ImageNet. We construct a registered database by randomly selecting 1, 5, and 10 fake images per class, respectively, from the training set of GenImage. All images in the validating set of GenImage are then used as queries to evaluate the performance.

## Construct the Database
You can construct the referencing database of GenImage (1, 5, 10-shot) and save it in the specified location by running the following command:
```
python feature.py --dataset_path='/path/to/dataset'
                  --weight_path='path/to/weights.pth'
                  --save_dir='path/to/features'
                  --num_gallery_per_class=1/5/10
                  --mode='full/patch'
                  --patch_mode='random/max/min'
                  --patch_size=32
```
The last two parameters are effective only when mode is set to 'patch'. Additionally, we provide the finetuned weights ([Baidu](https://pan.baidu.com/s/1xqP-asn2nuMZHMg2ny0aZA?pwd=k2gq) (code: k2gq) and [OneDrive](https://1drv.ms/u/c/52374438c618b7f2/IQALnU7DV8hqQIru2ff6MnIzAZ_gOjQlKjq2OcloUcVKS3I?e=D1RpuQ)) for evaluation. You can download the weights and easily evaluate LIDA.

## Evaluation
You can evaluate LIDA on GenImage with Rank-1 and mAP reported by running the following command:
```
python attribute.py --dataset_path='/path/to/dataset'
                    --weight_path='path/to/weights.pth'
                    --save_dir='path/to/features'
                    --num_query_per_class=num
                    --mode='full/patch'
                    --patch_mode='random/max/min'
                    --patch_size=32
```
Similarly, the last two parameters are effective only when mode is set to 'patch', and the weights have been given above.

```
@inproceedings{wang2026attribution,
      title={Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution}, 
      author={Hongsong Wang and Renxi Cheng and Chaolei Han and Jie Gui},
      year={2026},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
}
```

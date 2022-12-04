This 


The training and inference notebooks are based on [motono0223's notebook](https://www.kaggle.com/code/motono0223/guie-clip-tensorflow-train-example) and his datasets with my original improvements.

## Model
### for training:
- STEP1 Training without backbone model layers

backbone(CLIP VIT with [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)) + Dropout + Dense(units=64) + Arcface + Softmax (classes=17888)

Dataset of STEP1:
- [products10k](https://www.kaggle.com/datasets/motono0223/guie-products10k-tfrecords-label-1000-10690)  
  This dataset was created from [the product10k dataset](https://products-10k.github.io/).   
  To reduce the dataset size, this dataset has only 50 images per class.  

- [google landmark recognition 2021(Competition dataset)](https://www.kaggle.com/datasets/motono0223/guie-glr2021mini-tfrecords-label-10691-17690)  
  This dataset was created from [the competition dataset](https://www.kaggle.com/competitions/landmark-recognition-2021/data).  
  To reduce the dataset size, this dataset uses the top 7k class images with a large number of images (50 images per class).  

- STEP2 Training with all model layers

backbone(CLIP VIT with [pretrained clip-vit-large-patch14-336(STEP1 model)](https://huggingface.co/openai/clip-vit-large-patch14-336)) + Dropout + Dense(units=64) + Arcface + Softmax (classes=17888)

- [products10k](https://www.kaggle.com/datasets/motono0223/guie-products10k-tfrecords-label-1000-10690)  
- [google landmark recognition 2021(Competition dataset)](https://www.kaggle.com/datasets/motono0223/guie-glr2021mini-tfrecords-label-10691-17690)
- [stanford cars]()

- STEP3 Training with all model layers ()

backbone(CLIP VIT with [pretrained clip-vit-large-patch14-336(STEP2 model)](https://huggingface.co/openai/clip-vit-large-patch14-336)) + Dropout + Dense(units=64) + Arcface + Softmax (classes=17888)

- [products10k](https://www.kaggle.com/datasets/motono0223/guie-products10k-tfrecords-label-1000-10690)  
- [google landmark recognition 2021(Competition dataset)](https://www.kaggle.com/datasets/motono0223/guie-glr2021mini-tfrecords-label-10691-17690)
- [stanford cars]()
- [imagenet-mini with animal classes]()
- [MET Artwork Dataset with 9 images per class]()


- for inference:  

backbone(CLIP) + Dropout + Dense(units=64) + AdaptiveAveragePooling(n=64)


This challenge is supported with Cloud TPUs from Google's TPU Research Cloud (TRC). Thank you so much
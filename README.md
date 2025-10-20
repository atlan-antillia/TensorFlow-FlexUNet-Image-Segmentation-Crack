<h2>TensorFlow-FlexUNet-Image-Segmentation-Crack (2025/10/21)</h2>
<!--
Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
 -->
This is the first experiment of Image Segmentation for <b>Crack </b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512 pixels dataset <a href="https://drive.google.com/file/d/1xAWWpffwOb6CnWNmwtNH4h3XoDaYLd7S/view?usp=sharing">
Crack-ImageMask-Dataset.zip</a> 
 which was derived by us from 
<a href="https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset">
<b>Crack Segmentation Dataset</b>
</a>
<br>
<br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a>,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br><br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/images/CFD_035.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/masks/CFD_035.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test_output/CFD_035.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/images/CFD_068.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/masks/CFD_068.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test_output/CFD_068.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/images/CFD_097.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/masks/CFD_097.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test_output/CFD_097.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<b>Dataset Citation</b><br>
The dataset used here was obtained from 
<br><br>
<a href="https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset">
<b>Crack Segmentation Dataset</b>
</a>
<br><br>
<b>About Dataset</b><br>
This Dataset contains around 11.200 images that are merged from 12 available crack segmentation datasets.
<br>
The name prefix of each image is assigned to the corresponding dataset name that the image belong to. <br>
There are also images with no crack pixel, which could be filtered out by the file name pattern "noncrack*"
<br>
All the images are resized to the size of (448, 448).
<br>
The two folders of images and masks contain all the images. 
The two folders train and test contain training and testing images splitted from the two above folder. 
The splitting is stratified so that the proportion of each dataset in the train and test folder are similar
<br>
<br>
<b>LICENSE</b><br>
Unknown
<br>
<br>
<h3>
2 Crack ImageMask Dataset
</h3>
 If you would like to train this Crack Segmentation model by yourself,
 please download our data <a href="https://drive.google.com/file/d/1xAWWpffwOb6CnWNmwtNH4h3XoDaYLd7S/view?usp=sharing">
 Crack-ImageMask-Dataset.zip
 </a> on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Crack
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Crack Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Crack/Crack_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowUNet Model
</h3>
 We trained Crack TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Crack/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Crack and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Crack 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;                    crack:white
rgb_map = {(0,0,0):0,(255,255,255):1, }
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 20,22,21)</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 40,41,42)</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 42 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Crack/asset/train_console_output_at_epoch42.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Crack/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Crack/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Crack/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Crack/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Crack</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Crack.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Crack/asset/evaluate_console_output_at_epoch42.png" width="720" height="auto">
<br><br>Image-Segmentation-Crack

<a href="./projects/TensorFlowFlexUNet/Crack/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Crack/test was not low, and dice_coef not high as shown below.
<br>
<pre>
categorical_crossentropy,0.05
dice_coef_multiclass,0.9731
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Crack</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Crack.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Crack/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Crack/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Images of 512x512 pixels </b><br>
<b>rgb_map (Hemorrhagic:red, Ischemic:green)</b> <br>

<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/images/CFD_068.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/masks/CFD_068.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test_output/CFD_068.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/images/CFD_078.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/masks/CFD_078.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test_output/CFD_078.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/images/CFD_101.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/masks/CFD_101.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test_output/CFD_101.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/images/CRACK500_20160222_081031_1_721.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/masks/CRACK500_20160222_081031_1_721.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test_output/CRACK500_20160222_081031_1_721.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/images/CRACK500_20160222_081839_1921_361.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/masks/CRACK500_20160222_081839_1921_361.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test_output/CRACK500_20160222_081839_1921_361.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/images/CRACK500_20160222_114759_1281_1081.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test/masks/CRACK500_20160222_114759_1281_1081.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Crack/mini_test_output/CRACK500_20160222_114759_1281_1081.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. TensorFlow-FlexUNet-Image-Segmentation-Model</b>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>2. Multi-temporal crack segmentation in concrete structures using deep learning approaches</b><br>
Said Harb, Pedro Achanccaray, Mehdi Maboudi, Markus Gerke<br>
<a href="https://arxiv.org/pdf/2411.04620">https://arxiv.org/pdf/2411.04620</a>
<br>
<br>
<b>3. An efficient semantic segmentation method for road crack based on EGA-UNet</b><br>
Li Yang, Jingwei Deng, Hailong Duan & Chenchen Yang <br>
<a href="https://www.nature.com/articles/s41598-025-01983-3">https://www.nature.com/articles/s41598-025-01983-3<</a>
<br>
<br>
<b>4. EfficientCrackNet: A Lightweight Model for Crack Segmentation</b><br>
Abid Hasan Zim, Aquib Iqbal, Zaid Al-Huda, Asad Malik, Minoru Kuribayash<br>
<a href="https://arxiv.org/pdf/2409.18099">https://arxiv.org/pdf/2409.18099</a>
<br>
<br>
<b>5. Fine-grained crack segmentation for high-resolution images via a multiscale cascaded network</b><br>
Honghu Chu, Pang-jo Chun<br>
<a href="https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13111">https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13111</a>
<br>


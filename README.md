# LaneNet-Lane-Detection

使用tensorflow主要基于IEEE IV会议论文“走向端到端的车道检测：实例分割方法”，实现用于实时车道检测的深度神经网络。有关详细信息，请参阅他们的论文 [https：// arxiv .org / abs / 1802.05591](https://arxiv.org/abs/1802.05591)。该模型由编码器-解码器阶段，二进制语义分割阶段和使用区分性损失函数的实例语义分割组成，用于实时车道检测任务。

主要的网络架构如下：

`Network Architecture`
![2019-11-27_212604](README\2019-11-27_212604-1575372052560.jpg)

## Installation

该软件仅在带有GTX-1070 GPU的ubuntu 16.04（x64），python3.5，cuda-9.0，cudnn-7.0上进行了测试。要安装此软件，您需要tensorflow 1.10.0，并且尚未测试其他版本的tensorflow，但我认为它可以在版本1.10以上的tensorflow中正常工作。其他必需的软件包，您可以通过以下方式安装它们

```
pip3 install -r requirements.txt
```

## Test model

在这个仓库中，我上传了一个在tusimple车道数据集[Tusimple_Lane_Detection](http://benchmark.tusimple.ai/#/)上训练的模型。深度神经网络推理部分可以达到大约50fps，这与本文中的描述类似。但是我现在实现的输入管道需要改进，以实现实时车道检测系统。

训练有素的车网模型权重文件存储在 [new_lanenet_model_file中](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0)。您可以下载模型并将其放在文件夹model / tusimple_lanenet /中。

您可以按以下步骤在训练后的模型上测试单个图像

```
python tools/test_lanenet.py --weights_path ./model/tusimple_lanenet_vgg/tusimple_lanenet.ckpt 
--image_path ./data/tusimple_test_image/0.jpg
```

结果如下：

`Test Input Image`

![a](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\a.jpg)

`Test Lane Mask Image`

![lanenet_mask_result](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\lanenet_mask_result.png)

`Test Lane Binary Segmentation Image`

![lanenet_binary_seg](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\lanenet_binary_seg.png)

`Test Lane Instance Segmentation Image`

![lanenet_instance_seg](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\lanenet_instance_seg.png)

如果要在整个tusimple测试数据集上评估模型，可以调用如果设置save_dir参数，结果将保存在该文件夹中，或者结果将不会保存，而是在推理过程中显示（每张图像3秒钟）。我在整个tusimple车道检测数据集上测试了该模型，并将其制作为视频。您可能会瞥见它。

```
python tools/evaluate_lanenet_on_tusimple.py --image_dir /home/DataSet/CV/lanenet_data/dataset/clips --weights_path ./model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt  --save_dir ./test_set/test_output
```

`Tusimple test dataset gif`
![lanenet_batch_test](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\lanenet_batch_test.gif)

## Trainyour own model

#### 数据准备

# 数据准备

1. 首先按照链接下载 [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3) 数据集：train_set.zip  test_set.zip test_label.json
2. 调用 tools/generate_tusimple_dataset.py 将原始数据转换格式
   ![2019-11-28_100446](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\2019-11-28_100446.jpg)

这里会生成 train.txt 和 val.txt，调整格式如下：

```
testing/gt_image/0000.png testing/gt_binary_image/0000.png testing/gt_instance_image/0000.png
testing/gt_image/0001.png testing/gt_binary_image/0001.png testing/gt_instance_image/0001.png
testing/gt_image/0002.png testing/gt_binary_image/0002.png testing/gt_instance_image/0002.png
```

3. 调用 data_provider/lanenet_data_feed_pipline.py 转换标注成 TFRecord 格式

```
python data_provider/lanenet_data_feed_pipline.py 
--dataset_dir ./data/training_data_example
--save_dir ./data/training_data_example/tfrecords
```

#### Train model

在我的实验中，训练时期为80010，批量大小为4，初始学习速率为0.001，并使用幂为0.9的多项式衰减。关于训练参数，您可以检查global_configuration / config.py了解详细信息。您可以切换--net参数来更改基本编码器阶段。如果选择--net vgg，则vgg16将用作基本编码器阶段，并且将加载预训练的参数。您可以修改训练脚本以加载自己的预训练参数，或者可以实现自己的基本编码器阶段。您可以调用以下脚本来训练自己的模型

```
python tools/train_lanenet.py 
--net vgg 
--dataset_dir ./data/training_data_example
-m 0
```

您还可以通过以下方式从快照继续训练过程：

```
python tools/train_lanenet.py 
--net vgg 
--dataset_dir data/training_data_example/ 
--weights_path path/to/your/last/checkpoint
-m 0
```

您可以使用**张量板**工具监视训练过程：

在我的实验过程中 `Total loss` 如下:  
![total_loss](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\total_loss.png)

该`Binary Segmentation loss` 下降如下:  
![binary_seg_loss](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\binary_seg_loss.png)

该 `Instance Segmentation loss` 下降如下:  
![instance_seg_loss](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\instance_seg_loss.png)

## Experiment

训练过程中的准确性提高如下：
![accuracy](H:\real_work\LanNet_车道检测\lanenet-lane-detection-master\README\accuracy.png)

## Recently updates 2018.11.10

根据新的tensorflow API调整一些基本的cnn op。使用传统的SGD优化器而不是原始文件中使用的原始Adam优化器来优化整个模型。我发现SGD优化器将导致更稳定的训练过程，并且不会轻易陷入使用原始代码时经常发生的严重损失。

我已经在此处使用新代码[new_lanenet_model_file](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0)上传了一个在tusimple数据集上训练的新的Lanenet模型。您可以下载新的模型权重并更新新的代码。要更新新代码，您只需要

```
git pull origin master
```

其余与上面提到的相同。最近，我将发布一个在culane数据集上训练的新模型。

## Recently updates 2018.12.13

由于许多用户希望使用自动工具从Tusimple数据集生成训练样本。我上传了用于生成训练样本的工具。您需要首先下载Tusimple数据集并将文件解压缩到本地磁盘。然后运行以下命令以生成训练样本和train.txt文件。

```angular2html
python tools/generate_tusimple_dataset.py --src_dir path/to/your/unzipped/file

```

该脚本将创建训练文件夹和测试文件夹。原始rgb图像，二进制标签图像，实例标签图像的训练样本将自动在training / gt_image，training / gt_binary_image，training / gt_instance_image文件夹中生成。您可以在开始训练过程之前自行检查一下。

请注意，该脚本仅处理训练样本，并且您需要从train.txt中选择几行以生成自己的val.txt文件。为了获得测试图像，您可以自己修改脚本。

## 最近更新2019.05.16

新型号的砝码可以在[这里](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0)找到

## MNN项目

添加工具以将Lanenet Tensorflow CKPT模型转换为MNN模型并在移动设备上部署该模型

###### 冻结您的Tensorflow CKPT模型权重文件

```
cd LANENET_PROJECT_ROOT_DIR
python mnn_project/freeze_lanenet_model.py -w lanenet.ckpt -s lanenet.pb

```

###### 将PB模型转换为MNN模型

```
cd MNN_PROJECT_ROOT_DIR/tools/converter/build
./MNNConver -f TF --modelFile lanenet.pb --MNNModel lanenet.mnn --bizCode MNN

```

###### 将Lanenet源代码添加到MNN项目中

将Lanenet源代码添加到MNN项目中，并修改CMakeList.txt以编译可执行二进制文件。

## TODO

- [x] 添加嵌入可视化工具以可视化嵌入特征图
- [x] 分别添加详细培训Lanenet组件的详细说明。
- [x] 在不同的数据集上训练模型
- ~~[ ] Adjust the lanenet hnet model and merge the hnet model to the main lanenet model~~
- ~~[ ] Change the normalization function from BN to GN~~

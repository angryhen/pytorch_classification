# pytorch_classification

### 重构

&emsp;&emsp;整体流程（数据准备，传参，训练，测试）基本不变，主要丰富了框架的功能：

- [x] 简单的模型替换
- [x] APEX混合精度训练
- [x] 分布式训练
- [x] 参数的统一管理
- [x] 功能模块化



### 环境搭建

新框架的环境与之前保持一致，请查看主目录[README](https://github.com/FirminSun/TBox-server-dh/blob/master/README.md#%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)!



### train

#### 1. 参数修改

所有默认参数，通过yacs统一管理，存放在[config/config.py](config/config.py)里，针对每一项训练任务，额外通过yml文件管理参数（和以前一致），参考[c15.yml](https://github.com/FirminSun/TBox-server-dh/blob/master/classification/Cls_torch/configs/c15.yml)

#### 2. 训练

>```bash
># example
>python train.py --config configs/c15.yml
>--config： 只需传入你所用到的yml文件的名称
>
>请保证yml配置文件的参数正确
>```



### eval

### evaluation

eval.py 为预测/验证的代码，提供了两种方法

1. 单张图片, 结果会直接打印

    ```bash
    python eval.py --config {yml_file} --image  {image_path}  
    
    # example
    python eval.py --config configs/c15.yml --image hhl_lans_hz/2019-03-19_more_10038_3.jpg 
    ```


2. 整个文件夹，保持与制作数据集时一样的目录结构，利用make_csv.py生成test.csv

   测试的模型地址对应修改yml的checkpoint参数
   
   ```bash
   python eval.py --config {yml_file}
   
   # example
   python eval.py --config configs/c15.yml
   ```




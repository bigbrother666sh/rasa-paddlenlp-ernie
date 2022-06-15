# Rasa NLU Components using PaddleNLP

[![](https://img.shields.io/pypi/v/rasa_paddlenlp.svg)](https://pypi.python.org/pypi/rasa_paddlenlp)


## Features

- 兼容paddlenlp2.3 以及 Ernie3.0
- by bigbrother666

## Usage
#### 0、安装rasa3.x

```shell
pip install rasa
```

#### 1、安装paddlenlp 2.3.4

```shell
pip install paddlenlp==2.3.4
```
注：这一步之前需要安装PaddlePaddle，请自行查阅paddle官网，

如果需要使用gpu，请注意安装paddle的gpu版本

#### 2、建立文件夹，结构如下

```
.\
|
|--configs
|--data
```

#### 3、把本repo deploy分支下的rasa-paddlenlp文件夹整个拷贝到 .\下

#### 4、配置config.yml，可以多配置几个，以便比较哪个效果最好，所有配置文件都扔到configs里面

repo中提供三个configs示例，config为官方自带的jieba方案，config1为SimonLiang的bert方案（已兼容paddlenlp2.3)，config5为Ernie3.0-base-zh版本

但是要注意：**config1和config5不能直接比较，因为程序会有不同**

#### 5、准备nlu训练数据文件，按格式要求，可以准备多份，扔到 data里面，训练时程序会自动合并，

配置文件和训练文件不懂，看这里： [https://rasa.com/docs/rasa/nlu-training-data](https://rasa.com/docs/rasa/nlu-training-data)

#### 6、在.\中运行full-test，遴选出最优的config

```shell
rasa test nlu --config configs --percentages 20 50 80
```

注意：这一步config1和config5只能保留一个，把另一个暂时移出configs文件夹

这一步需要漫长的等待…………

#### 7、在.\results文件夹下查阅nlu_model_comparison_graph.pdf文件

注意两个信息：

1、分值最高的曲线对应的configs就是你对应这批数据该用的；

2、如果你发现曲线到最后还处于上升姿态，说明你还可以通过追加nlu训练数据提高模型表现。

除了以上信息外也可以进入具体文件下查看细节

#### 8、把表现最好的config文件拷贝到.\中，并命名为 config.yml 然后在.\目录执行

```shell
rasa train nlu
```

#### 9、自己试用

```shell
rasa shell nlu
```

#### 10、api访问

```shell
rasa run --enable-api
```

注意：这一步可以通过 -i 参数指定侦听地址，，默认是 http://0.0.0.0(linux下) 或者 http://localhost(windows下)

通过 -p 指定侦听端口

另外 rasa shell 命令也支持 --enable-api参数

#### 11、如果数据有更新，更新量不大的话，建议先test下现有模型，看要不要finetune或者重新训练

```shell
rasa test nlu --nlu 新数据文件 --cross-validation
```

训练好也是到results文件夹下，主要看 intent_histogram.png，如果左边蓝条都集中在上部，右侧红条都集中在下部（当然没有更好），说明模型对新数据泛化的也不错，可以不训练。

更加详细的，可以看同名目录下的两个json文件，一个列出了所有识别错误的数据，一个列出个各个class的recall 精度 特异以及F1分值。

（其实这个也可以用来比较两个模型的表现，注意通过--model参数指定测试的模型，然后如果测试数据也是放在data文件夹下，可以不指定--nlu参数，不过不建议这样，容易最后搞混）

#### 12、上面用的是交叉验证，也可以用传统的训练方法，但你的数据要足够多

先拆分测试集和训练集

```shell
rasa data split nlu
```

之后用训练集训练，测试集测试……

个人感觉没啥必要，都是小模型，你也没那么多数据，用官方默认的方法挺好

另外最后注意，rasa shell和rasa run都会默认调起同级models文件夹下最新的模型，这两个命令不看config.yml的，模型会按照训练时的pipeline进行predict。

-------------

### 有关PaddleNLP2.3兼容和Ernie调起的问题（本项目主要代码更改）：

paddlenlp2.3跟2.2相比在tokenizer的方法上有比较大的变化，所以我们主要需要更改paddlenlp_tokenizer.py，主要是更改126行，增加两个入参即可，分别是：

```
return_dict=False
max_length=512
```

然后Ernie3.0的返回跟bert不太一样，所以为了调起ernie3.0还需要更改130行为：

```
if e['special_tokens_mask'][0] == 1:
```

**这里其实我不是很有把握是否妥当**，主要是我没有研究paddlenlp这部分代码，只是这里不把i改成0，哪怕是改成 i-1 也会报out of range，所以我怀疑Ernie3.0这里的返回值只有1个……

但实测下来这样改是可以调起ernie3.0正常使用的，并且实测效果还好于jieba方案和bert方案……

但是这一句改了，paddlenlp的bert就调不起来了，这也就是本项目没有办法直接同时比较bert和ernie3.0的原因

具体见：https://github.com/senses-chat/rasa-paddlenlp/issues/3

## Credits

This package took inspiration from the following projects:

- [Rasa](https://github.com/rasahq/rasa)
- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)

This package was created with Cookiecutter and the audreyr/cookiecutter-pypackage project template.

- [Cookiecutter](https://github.com/audreyr/cookiecutter)
- [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage)

## License

[MIT](./LICENSE)

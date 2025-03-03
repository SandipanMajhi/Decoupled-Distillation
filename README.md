# Decoupled-Distillation
Tested the paper on Decoupled Disitillation on ResNet 8 as student and Resnet 18 as teacher model. Check the test files for usage.

### Results : 
| Experiment/Model                                      | Dataset | Num Epochs | Optimizer | BatchSize | Val Acc  | Test Acc | Comments                           |
|-------------------------------------------------------|---------|------------|-----------|-----------|----------|----------|------------------------------------|
| Pretrained + Finetuned ResNet18 (Teacher)             | CIFAR10 | 50         | Adam      | 128       | 0.9122   | 0.911    |                                    |
| Finetuned from Scratch ResNet8 (Student)              | CIFAR10 | 50         | Adam      | 128       | 0.8014   | 0.8049   |                                    |
| Distilled ResNet8 from ResNet18 (DKD)                 | CIFAR10 | 100        | -         | -         | 0.8766   | 0.8721   | Clear increase over the original   |

### Ablation Study:
##### Alpha - 
| Experiment/Model                                      | Dataset | Num Epochs | Mode      | alpha       | Val Acc  | Test Acc | Comments                           |
|-------------------------------------------------------|---------|------------|-----------|-------------|----------|----------|------------------------------------|
| Distilled ResNet8 from ResNet18 (tckd)                 | CIFAR10 | 100       | tckd      | 1.0         | 0.8688   | 0.8422   | 4% increase                        |
| Distilled ResNet8 from ResNet18 (tckd)                 | CIFAR10 | 100       | tckd      | 0.5         | 0.8476   | 0.8482   | 4% increase                        |
| Distilled ResNet8 from ResNet18 (tckd)                 | CIFAR10 | 100       | tckd      | 0.25        | 0.8418   | 0.8302   | 3% increase                        |

##### Beta - 
| Experiment/Model                                      | Dataset | Num Epochs | Mode      | beta        | Val Acc  | Test Acc | Comments                           |
|-------------------------------------------------------|---------|------------|-----------|-------------|----------|----------|------------------------------------|
| Distilled ResNet8 from ResNet18 (tckd)                 | CIFAR10 | 100       | nckd      | 2.0         | 0.8506   | 0.8387   | 3% increase                        |
| Distilled ResNet8 from ResNet18 (tckd)                 | CIFAR10 | 100       | nckd      | 4.0         | 0.8476   | 0.8482   | 4% increase                        |
| Distilled ResNet8 from ResNet18 (tckd)                 | CIFAR10 | 100       | nckd      | 6.0         | 0.8418   | 0.8302   | 3% increase                        |

### Citation: 

```
@article{Zhao2022DecoupledKD,
  title={Decoupled Knowledge Distillation},
  author={Borui Zhao and Quan Cui and Renjie Song and Yiyu Qiu and Jiajun Liang},
  journal={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
  pages={11943-11952},
  url={https://api.semanticscholar.org/CorpusID:247476179}
}
```

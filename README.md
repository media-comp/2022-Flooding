# 2022-Flooding


This is tensorflow implementation of *"Takashi Ishida, Ikko Yamane, Tomoya Sakai, Gang Niu, and Masashi Sugiyama. 2020. Do we need zero training loss after achieving zero training error? In Proceedings of the 37th International Conference on Machine Learning (ICML'20). JMLR.org, Article 428, 4604–4614"*

Scripts for MNIST, Fashion-MNIST and KMNIST are present in the models folder. This work can be extended in numerous ways namely *"Implementing Flooding algorithm for Deep CNNs"* or *"Training while varying Flooding constant"*.

## Algorithm

Proposed Objective Function:

![equation](https://latex.codecogs.com/svg.image?\tilde{J}\left&space;(&space;\theta&space;\right&space;)&space;=&space;\left|&space;J\left&space;(&space;\theta&space;&space;\right&space;)&space;-&space;b&space;&space;\right|&space;&plus;&space;b&space;)  
where, b = Flooding constant

## Results

Following figures represent the performance of MLP Classifier for the Fashion-MNIST dataset when trained with and without __Flooding__. We notice that while for the model trained without Flooding the testing loss diverges away but, by using __Flooding__ the testing loss stablizes. 

Comparing Accuracies             |  Comparing Losses
:-------------------------:|:-------------------------:
![](https://github.com/anubhav2901/Flooding/blob/main/figures/accuracy.png)  |  ![](https://github.com/anubhav2901/Flooding/blob/main/figures/loss.png)

## Requirements

> - keras==2.8.0
> - matplotlib==3.5.2
> - numpy==1.22.3
> - tensorflow==2.8.0


## Demo
Run the following command for a demo of Flooding on MNIST dataset

> python demo.py

## References
- Takashi Ishida, Ikko Yamane, Tomoya Sakai, Gang Niu, and Masashi Sugiyama. 2020. Do we need zero training loss after achieving zero training error? In Proceedings of the 37th International Conference on Machine Learning (ICML'20). JMLR.org, Article 428, 4604–4614
- https://www.tensorflow.org/

# Visual BigData Algorithm
# 大数据算法实践和可视化

---
# 算法
### K-Means
对于给定的样本集，按照样本之间的距离大小，将样本集划分为K个簇。让簇内的点尽量紧密的连在一起，而让簇间的距离尽量的大。

实现：
* 对旧金山878049条犯罪记录的地点进行K-Means聚类，以确定犯罪中心，数据集来自[https://www.kaggle.com/c/sf-crime/data](https://www.kaggle.com/c/sf-crime/data), 仓库内未包含训练数据
* 使用轻量化的数据集加快运行速度，使聚类的效果和计算速度得到提升，可用于介绍K-Means算法
* 使用均匀分布的数据集来分布演示K-Means算法的过程，可清楚看到每次迭代的过程和中心质点的变化

---
# Python依赖
K-Means.py
```shell
pip install pandas
pip install sklearn
pip install matplotlib
```
---
# 使用方法
K-Means.py
```python
def criminal_kmeans()  # 执行此函数用于跑旧金山罪犯记录
def slim_kmeans()  # 执行此函数用于演示K-Means最佳分类情况
def iter_kmeas()  # 执行此函数用于分步演示K-Means
```
---
# 截图
旧金山犯罪数据(放大了)

![criminal_kmeans](https://github.com/50Death/Visual-BigData-Algorithm/blob/master/screenshots/criminal_kmeans.jpg)
演示K-Means

![slim_kmeans](https://github.com/50Death/Visual-BigData-Algorithm/blob/master/screenshots/slim_kmeans.png)
分步演示GIF

![iter_kmeas](https://github.com/50Death/Visual-BigData-Algorithm/blob/master/screenshots/iter_kmeans.gif)

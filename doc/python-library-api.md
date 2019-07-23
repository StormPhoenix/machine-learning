#NumPy
---
##numpy.array
&emsp;&emsp;numpy库自定义的数组类型,支持多种运算操作:
* **array** ** **x** 对array中的每个值做乘方运算
* **array.sum()** 其中值得说明axis参数,axis的值指定从哪一维度开始相加
* **array.argsort()** 默认从低到高排序,返回排序后的下标数组
* **array.shape** 类型为 **tuple**,表示numpy.array各个维度的大小
##numpy.array(list)
&emsp;&emsp;**参数:** 数组

&emsp;&emsp;**用法:** 初始化一个numpy.array类型的数组返回

##numpy.tile(a, reps)
&emsp;&emsp;**参数:** a数组,reps表示从哪些方向扩展

&emsp;&emsp;**用法:** 对传入的a做复制扩展,扩展大小由reps规定

##numpy.zeros(shape)
&emsp;&emsp;**参数:** shape元组

&emsp;&emsp;**用法:** 用0初始化一个矩阵,维度由shape指定

##numpy.linspace(a, b, count)
&emsp;&emsp;**参数:** 数字

&emsp;&emsp;**用法:** 在a和b之间等间距取count个数,并返回array

##numpy.empty(shape)
&emsp;&emsp;**参数:** shape元组,表示数组的维度,以及各个维度的长度

&emsp;&emsp;**用法:** 初始化一个空array,shape指定维度

##numpy.append(array_a, array_b, axis)
&emsp;&emsp;**参数:** array_a, array_b表示待合并的数组, axis指定从哪些方向合并

&emsp;&emsp;**用法:** array_a 与 array_b合并后返回,axis制定在哪个维度上合并.但是numpy.array不适合用于**大规模,经常性**的动态扩展.

&emsp;&emsp;[关于numpy初始化空数组,以及如何扩展的说明](https://vimsky.com/article/3717.html)

##numpy.nonzero(array)
&emsp;&emsp;**参数:** array 传入的矩阵

&emsp;&emsp;**用法:** 返回 array 非零的索引

&emsp;&emsp;**返回值:** 返回二值元组 (row_index, col_index), row_index 是非零元素的行索引, col_index 是非零元素列索引.

##numpy.multiply, numpy.dot, * 
[区别参考链接](https://blog.csdn.net/zenghaitao0128/article/details/78715140)
#os
---
##os.path.isfile(filename)
&emsp;&emsp;**参数:** filename 文件名

&emsp;&emsp;**用法:** 判断filename文件是否存在

#pickle
---
简单理解:对象序列化到磁盘,或从磁盘文件反序列化成对象
```
import pickle

def storeTree(inputTree, fileName):
    fw = open(fileName, "wb")
    pickle.dump(inputTree, fw)
    fw.close()


def loadTree(fileName):
    fr = open(fileName, "rb")
    return pickle.load(fr)
```


#Matplotlib
---
##figure()
&emsp;&emsp;**pyplot**模块中的方法,返回一个**figure**对象.此方法调用之后的所有绘制操作都在这个**figure**上进行.

##figure对象
* add_subplot(row, col, index) 将figure分成 row x col 的格子,在第index格子位置进行绘图.
* ax = add_subplot(), ax可调用scatter()方法绘制散点图

##show()
&emsp;&emsp;展示绘制的图

#Python
---
##object[]与object.get()
获取python对象某个属性的值,如果对象没有这个属性,前者会报错,后者会返回默认值

##sorted(object, key, reverse)
&emsp;&emsp;**参数:** 可迭代对象,用于比较的键,比较结果是否要翻转
&emsp;&emsp;**用法:** reverse=True返回从低到高的比较结果,形式为列表,每个元素用元组表示

##list的深复制与浅复制

```
a = [1, 2, 3]
b = a # 浅复制
b = a[:] # 深复制
```

##type()
&emsp;&emsp;**参数:** 任意对象
&emsp;&emsp;**用法:** 返回对象的类型

```
a = {}
type(a) # output: dict
type(a).__name__ # output: 'dict'
```

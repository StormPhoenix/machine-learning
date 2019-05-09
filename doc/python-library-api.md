#NumPy
---
##numpy.array
&emsp;&emsp;numpy库自定义的数组类型,支持多种运算操作:
* **array** ** **x** 对array中的每个值做乘方运算
* **array.sum()** 其中值得说明axis参数,axis的值指定从哪一维度开始相加
* **array.argsort()** 默认从低到高排序,返回排序后的下标数组
* **array.shape** 类型为 **tuple**,表示numpy.array各个维度的大小
##numpy.array(list)
&emsp;&emsp;参数:数组
&emsp;&emsp;初始化一个numpy.array类型的数组返回

##numpy.tile(tuple)
&emsp;&emsp;参数:数组,元组
&emsp;&emsp;对传入的数组做扩展,扩展大小由元组规定

##numpy.zeros(tuple)
&emsp;&emsp;参数: 元组
&emsp;&emsp;用0初始化一个矩阵,维度由参数指定

##numpy.linspace(a, b, count)
&emsp;&emsp;参数: 数字
&emsp;&emsp;在a和b之间等间距取count个数,并返回array

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

##sorted()
&emsp;&emsp;参数:可迭代对象,用于比较的键,比较结果是否要翻转
&emsp;&emsp;reverse=True返回从低到高的比较结果

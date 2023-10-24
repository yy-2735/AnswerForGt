# 是什么
1.运行在客户端的脚本语言
2.脚本语言：不需要编译，运行过程中由js解释器逐一解释并执行
3.部分作用：
>表单动态
*网页特效*
服务端开发
桌面程序
控制硬件
游戏开发

4.区分：
* HTML和CSS：描述类语言；
* JS：编程类语言；
5.浏览器执行JS的简介
* 浏览器本身并不会执行JS代码，而是通过内置的JS解释器
* JS引擎逐行解释每一句源码，转换为机器语言
# 组成
1.ECMAscript（语法）
2.DOM（页面文档对象）：如各元素
3.BOM（浏览器对象）：如弹出框，控制浏览器跳转，获取分辨率等

# 基本书写
1.书写位置
>行内:直接写到元素的内部
内嵌：在head处写
外部：新建文件，后缀为.js，再在HTML页面中引入即可
ps.大致规律同CSS
2.引号的使用：
HTML中一般用双引号而JS中一般用单引号
3.外部引用时两个script间不能写代码
4.注释方式
* 单行注释（快捷键）
* 多行注释（shift+alt+a）
5.不写；不算错

# 输入输出语句
1.alert():浏览器弹出**警示框**，**输出**展示给用户看的
2.console.log(''):浏览器控制台打印**输出**信息
3.prompt（）：浏览器弹出输入框，用户可以**输入**

# 变量
1.本质是在内存中一块用来存放数据的空间。
2.声明变量：*var*
3.变量的初始化
4.变量的命名规范：
* 大致规则同C++
* 驼峰命名法
5.交换两个变量的值
* 再创建一个临时变量即可


# 数据类型
1.注意：JavaScript是一种动态语言，即可以不用**提前声明**变量类型，只有当**程序运行的时候**才能被确定
2.意味着相同的变量可以用作不同的类型

## Number
1.进制
>0：表八进制
0X:表十六进制
2.最大值，最小值
3.三个特殊值

## isNaN()
1.用这个判断是否是**非数字**
2.如果是非数字，返回true，反之则返回false

## String(字符串型)
1.var str=''
ps.引号复合就近匹配的原则
2.转义符与C++的基本相同
3.length：获取字符串的长度，写为console.log(str.length)
ps.空格等也算长度
4.字符串的拼接，用+号
* 其的加强使用：可以用来代换字符串中变量

## 布尔型
1.基本规律同C++

## undefined
1.一个变量声明后未赋值

## null
1.给变量赋null则变为空值

## typeof
1.用来检测变量的数据类型
2.如console.log(typeof num)
# DOM
## 简要介绍
1.标准编程接口
2.通过接口可以改变网页的内容，结构，样式
3.文档：一个页面就是一个文档
4.元素：页面里面所有的标签都是元素
**5.网页中的所有内容都是节点（如标签，属性，文本，注释等），用node表示**
5.记忆：dom树

## 获取元素
1.根据ID获取
* 匹配特定ID的元素
ps.ID是大小写敏感的字符串，且是唯一的
* 返回值：返回一个匹配到ID的对象
2.根据标签名获取
3.通过HTML
4.



# PC端网页特效
## Offset
1.意即偏移量，使用其相关属性可以动态的得到该元素的位置
2.获得元素自身的大小
3.获得
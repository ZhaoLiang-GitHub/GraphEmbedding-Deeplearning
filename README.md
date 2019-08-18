# GraphEmbedding-Deeplearning
采用深度学习的图表征学习，包括Deepwalk\line\node2vec\sdne\struc2vec等—截止201908

该仓库为修改自[GitHub](https://github.com/shenweichen/GraphEmbedding)，应该该仓库的代码逻辑我个人并不是十分认可，所以在此将其重构并添加注释与相关文档（原始代码中啥注释和文档也没有，我有理由怀疑原始仓库地址的作者不是自己手写的也是搬运的），所以耗时比较长，会逐步添加更新，如果有做相关工作的同学可以与我邮件联系1318525510@qq.com

1. config.py

  该文件为参数设置文件，所有的参数均在该文件内设置，在模型与主文件中均不需要填写，所有的默认值均在模型中写好，如果有默认值时，在参数文件中可以设置为None
  
2. setup.py

  安装所需要的一些依赖库，注：其中一部分库（例如TensorFlow）的版本更新后会出现函数名或者参数名的修改，如果您在使用中出现了该问题请提交request
  
3. main.py 

  主文件，设置好参数文件及数据后，运行该文件即可
  
4. data
  
  data/knowledgegraph_name/ 中应该含有两个文件，一个关系二元组,一个实体ID映射文件 
  
5. code

  模型文件，该文件是不对外暴露的，其内部参数也不需要修改，
  

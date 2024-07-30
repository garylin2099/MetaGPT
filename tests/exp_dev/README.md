## 经验实验跑测流程

实验入口：`tests/exp_dev/exp_with_knowledge.py`  
参数设置：`tests/exp_dev/exp_with_knowledge.py`中的`ExpConfig`类，含义如下：  
```python
class ExpConfig(BaseModel):
    use_tools: bool = True  # 是否使用工具
    plan_with_knowledge: bool = True  # 是否在plan生成阶段使用知识
    task_with_knowledge: bool = True  # 是否在代码生成阶段使用知识
    optimize_with_knowledge: bool = False  # 是否在迭代优化方案生成阶段使用知识
    with_data_info: bool = False  # 是否在plan生成阶段引入数据集信息
    knowledge_rerank: bool = False  # 是否使用知识重排挑选top知识
    max_optimize_iter: int = 10  # 优化迭代次数
```

## 经验实验分步骤详解

1. 下载Kaggle Top方案  
   账号配置：需要在`config/config2.yaml`中配置Kaggle账号信息如下：  
   ```yaml
   kaggle:
     username: "YOUR_USERNAME"
     key: "YOUR_KAGGLE_KEY"
   ```  
   代码位置：`metagpt/utils/kaggle_client.py`  
   存储位置：代码运行后会在`data/kaggle/competitions`下生成每个比赛的文件夹，文件夹下存储了该比赛的Top方案
2. 根据Kaggle Top方案生成知识库  
   代码位置：`metagpt/actions/di/knowledge_extraction.py`
3. 迭代优化实验  
   实验入口即为前文`tests/exp_dev/exp_with_knowledge.py`

In this repository we readapt [MIND](https://arxiv.org/abs/2312.02916) paper, which is a replay free task increamental learning paradigm, to the federated learning task.

Federated Learning is a distributed machine learning paradigm in which multiple nodes (clients) collaboratively train a shared global model without sharing their raw data. Instead of centralizing data from different sources, each client keeps its local data private and performs training independently.

After local training, clients send only model updates such as learned weights or gradients to a central server. The server aggregates these updates to produce a new global model, which captures knowledge from all clients as if it had been trained on the combined dataset. The updated global model is then redistributed to clients for the next training round.

This approach enables collaborative learning while preserving data privacy and reducing the risks associated with centralized data storage. 

![Classical federated learning architecture](/images/FL_Scheme.png)

When different clients are exposed to different learning tasks or class subsets over time, the training setting is referred to as Federated Class-Incremental Learning (FCIL). In this scenario, each client learns new classes sequentially while having no access to data from previously learned tasks, nor to data belonging to other clients.

At each incremental step, clients train a local model using only the data of their current task. Once local training is completed, the updated model parameters are sent to a central server, where they are aggregated to form a new global model. This global model is then redistributed to the clients and used as the starting point for training on the next novel task. The process repeats as new classes become available over time.

![New architecture](/images/Federated_scheme.png)

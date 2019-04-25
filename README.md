# Privacy Protected Federated Learning on Blockchain
Federated learning convolution neural network (FL-CNN) to detect mode of transport from Itinerum GPS trajectories.

Federated learning on blockchain is a modeling approach to learn collaboratively a shared mode (of transport) inference model while the data is stored on different users' smartphones, each one is registered on a blockchain node (referred to as worker nodes). In such framework, a single shared model is trained locally on each node's data and the changes to the model are summarized and submitted to the master node of the blockchain[1]. Finally, the summarized updates from different nodes is transferred to the master node, and will be averaged to improve the single shared model[1].

The repository containing the codes for federated convolutional neural network to detect the mode of transport from user's smartphone GPS trajectories. The federated learning allows to protect the privacy of the users while analyzing their GPS trajectories gathered by their smartphones. Figure1 demonstrates how the federated learning on blockchain works.
<br/>

![](https://github.com/Ali-TRIPLab/Privacy_Protected_Federated_Learning_on_Blockchain/blob/master/images/FL_Diagrams.jpg?raw=true)
<p align="center"><b>Figure.1 Federated Learning implemented on Blockchain</b></p>
 <br/>



**References** <br/>

[1] Federated learning: Collaborative machine learning without  centralized  training  data,https://ai.googleblog.com/2017/04/federated-learning-collaborative.html, accessed:  2019-01-22.

### Two-Stream-RNN for skeleton recognition

Reimplementation in **PyTorch** of **CVPR2017** *'Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks'*

****
This repository is final examination of the course *"Deep learning and Computer vison"* in SCUT 
which is contributed by (18信息工程6班) 杨东杰 李哲 吴嘉杰

**PS**: 如果同学们要**直接**使用这个代码交作业，我们希望你能在作业中说明对我们小组的代码进行了参考。
（如果是同学们只是参考了这个代码，但是**代码是自己写**的，就不必说明了）
****
#### How to use

- Download NTU-RGBD dataset
  
  <https://github.com/shahroudy/NTURGB-D>  
<br >

- Convert NTU-RGBD dataset to h5 file
    ```
    cd data
    python load_ntu_rgbd.py
    ```
    Some configurations before you running:
    1. You can choose **cross-view** or **cross-subject** evaluation.
    Details about the evaluation on NTU-RGBD dataset  in paper *'NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis'*.  
    2. **missing_skeleton.txt** should be put under directory of "data".
    <br >

- Start training
    ```
    python train.py --task_name  train_task1 --temp_rnn_type hierarchical --spatial_seq_type traversal --batch_size 512 --eval_batch_size 512 --epochs 2000 --lr 0.001 --eval_period 5 --lr_decay_gamma 0.75 --lr_decay_step 50 
    ```
    More options can be selected in train.py.
    The checkpoint and log information will be saved every time the model has been evaluated.  
    <br >

- Resume training
    ```
    python train.py --task_name  train_task2 --temp_rnn_type hierarchical --spatial_seq_type traversal --batch_size 512 --eval_batch_size 512 --epochs 2000 --lr 0.001 --eval_period 5 --lr_decay_gamma 0.75 --lr_decay_step 50 --resume [last checkpoint path]
    ```
    Because of some bugs not fixed, the learning rate shoud be restored by inputing the value manually.
    

### Our results
Results for running for 100 epochs. 
Accuracies on testset won't change after 60 epochs.

|model type|evaluation|temporal rnn type|spatial sequence type|accurry|
|----|----|----|----|----|
|original paper|cross subject|hierarchical|traversal|0.647|
|modified by us|cross subject|hierarchical|traversal|0.589|

we modify the model to make the balance between temporal rnn and spatial rnn to be able to learn. However it seems to perform not so well.

### Contact us
Contact any of the contributors is OK.

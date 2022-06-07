# DL based DNA fast decoding

_Data and Code Adopted from the paper "Fast decoding cell type–specific transcription factor
binding landscape at single-nucleotide resolution"_

Abstract:

    "Decoding the cell type–specific transcription factor (TF) binding landscape at single-nucleotide resolution is crucial for understanding the regulatory mechanisms underlying many fundamental biological processes and human diseases. However,
    limits on time and resources restrict the high-resolution experimental measurements of TF binding profiles of all possible
    TF–cell type combinations. Previous computational approaches either cannot distinguish the cell context–dependent TF
    binding profiles across diverse cell types or can only provide a relatively low-resolution prediction. Here we present a novel
    deep learning approach, Leopard, for predicting TF binding sites at single-nucleotide resolution, achieving the average area
    under receiver operating characteristic curve (AUROC) of 0.982 and the average area under precision recall curve
    (AUPRC) of 0.208. Our method substantially outperformed the state-of-the-art methods Anchor and FactorNet, improving the predictive AUPRC by 19% and 27%, respectively, when evaluated at 200-bp resolution. Meanwhile, by leveraging a
    many-to-many neural network architecture, Leopard features a hundredfold to thousandfold speedup compared with current many-to-one machine learning methods."

Rather than modeling all transcription factors, in this challenge, we will just focus on one transcription factor: **CTCF** (a highly conserved zinc finger protein). Figure 1 below demonstrates the input and target output of the deep learning model.

![Figure1](/figure/fig1.png?raw=true "Title")

The input of the model consists of two parts: 
1. DNA sequence: one-hot encoding of the DNA sequences(A[1,0,0,0],T[0,1,0,0],C[0,0,1,0],G[0,0,0,1]), which have input length equal 
to 10240bp.
2. DNase-seq: real value of DNase-seq filtered alignment signal, which contain chromatin accessibility information of the corresponding DNA sequence. A location with higher
value indicates such location is more likely to be a transcription factor's binding site.

The output of the model is the predicted transcription factor's binding site of the corresponding DNA sequence where 
'0' indicated current nucleotide is not a binding site and '1' indicated current nucleotide is a binding site.

Thus,

    the input shape: (batch_size, 10240, 5)
    the output shape: (batch_size, 10240, 1) 

For the construction of the dataset, we separate the human chromosomes into three groups: where we randomly select 16 chromosomes for training and 4
chromosomes for validation, and 2 chromosomes for the test. And from the Training Chromosome, we randomly sample 20,000 DNA sequences and their corresponding
DNase-seq data. For validation and test datasets, we randomly sample 4,000 datapoint from their chromosome sets respectively.

Since data preprocessing is not the main focus of this competition and will take extra computation resources, we have preprocessed and saved the data in the tf.data.Dataset format. To load the dataset, 

    dataset = tf.data.experimental.load(path)

For the evaluation of the model performance, we use the following metrics PR_AUC, Binary IOU, Dice Coefficient, and model training time. The table below shows the benchmark performance for a relatively naive multi-layer CNN model training over two Nvidia V100 GPUs.
 

model          | loss          | pr_auc        | dice_coef    | binary_iou | training time(s) 
| -------------| ------------- |:-------------:| ------------:|-----------:| -------------:
|   cnn       |    0.000850   | 0.456871      | 0.330022     |0.309475    |  1356.7

To setup conda environment:

    conda env create --name leopard --file environment.yml
    conda activate leopard


We have provided the framework to load the data and train the model.  To make a copy of the preprocessed data and model, please find instruction here (TBD). By default, it will use a naive multi-layer CNN network to predict CTCF binding sites. 

To train the model:

    python deep_tf.py 

To specify a different model architecture, first, implement your model in models.py, and then use -m option to pass in your selected model for training.
    
    python deep_tf.py -m cnn

Once you successfully train your model, it will generate an output folder that contains model checkpoints, saved model, model's performance result, and horovod timeline file. If you are satisfied with your model's performance, please submit the folder of the saved model. We will run your model in reserved data and rank your model based on the metrics above.

For model selection, please find the orignal paper using [UNet](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8015851/). Since the transciption binding site problem is similar to image segmentation problem. some recently emerged architecture such as [Vision Transformer](https://arxiv.org/abs/2010.11929) and [Swin Transformer](https://arxiv.org/abs/2103.14030) might be useful.


For training using NCI Gadi, we have provided the following job submission script **leopard.sh** and **distributed_leopard.sh**. For more details on NCI Gadi and PBS Job, please see [link](https://opus.nci.org.au/display/Help/Gadi+User+Guide)

To train the model with one GPU on Gadi:

    qsub leopard.sh

for distributed training using horovod:

    qsub distributed_leopard.sh




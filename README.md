# midl2021_Pyramid Stream Networks
This is a simple implementation for "Semantic Segmentation of Lung Adenocarcinoma Growth Patterns using Pyramid Stream Networks" submitted to MIDL2021 short paper track.

* Superpixel pooling using a classical SLIC method, in the paper, the number of superpixel is set to 500 and 2000
* img_to_npy.py: dividing images into patches with a uniform size, 384$\times$384
* train.py: training a model
* predict_loop.py: generating a segmentation mask

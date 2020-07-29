
# Using Custom Datasets

Here is the class that must be implemented in order to use custom datasets

    from abc import ABC, abstractmethod

    class BaseDataset(ABC):

        @abstractmethod
        def __init__(self,
                        dataset_dir,
                        subset,
                        flip_ratio,
                        pad_mode,
                        mean,
                        std,
                        preproc_mode,
                        scale,
                        train,
                        debug):
            pass

        @abstractmethod
        def _filter_images(self):
            pass

        @abstractmethod
        def get_ann_info(self, index):
            pass

        @abstractmethod
        def num_classes(self):
            pass

        @abstractmethod
        def get_labels(self):
            pass

        @abstractmethod
        def __len__(self):
            pass

        @abstractmethod
        def __getitem__(self, index):
            pass

Feel free to change the constructor definition as you see fit. If you do so, make sure to adjust the the dataset part in the configuration file used for training as well.

For an example of how to implement custom datasets, see [pascal.py](pascal.py), which implements the [Pascal VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). 

In order to use the custom dataset, you must instantiate your custom dataset instead of the default COCO dataset in the configuration file used for training. This is how the Pascal dataset would be used:

    # dataset settings
    dataset_type = 'PascalDataset'
    data_root = '/deep-learning/models/models/vision/detection/data/pascal/VOCdevkit/VOC2012/'
    data = dict(
        imgs_per_gpu=4,
        train=dict(
            type=dataset_type,
            train=True,
            dataset_dir=data_root,
            subset='train',
            flip_ratio=0.5,
            pad_mode='fixed',
            preproc_mode='caffe',
            mean=(123.675, 116.28, 103.53),
            std=(1., 1., 1.),
            scale=(800, 1333)),
        val=dict(
            type=dataset_type,
            train=False,
            dataset_dir=data_root,
            subset='val',
            flip_ratio=0,
            pad_mode='fixed',
            preproc_mode='caffe',
            mean=(123.675, 116.28, 103.53),
            std=(1., 1., 1.),
            scale=(800, 1333)),
        test=dict(
            type=dataset_type,
            train=False,
            dataset_dir=data_root,
            subset='val',
            flip_ratio=0,
            pad_mode='fixed',
            preproc_mode='caffe',
            mean=(123.675, 116.28, 103.53),
            std=(1., 1., 1.),
            scale=(800, 1333)),
    )

Lastly, you will need to switch out the evaluation hook used from the COCO specific evaluation hook, to a generic evaluation hook. In [train.py](../apis/train.py) in the _dist_train function, where this line occurs: 

    runner.register_hook(CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))

change it to:

    runner.register_hook(DistEvalmAPHook(val_dataset_cfg, **eval_cfg))

Below is some information that might help with the implementation.

 - When building a mapping between class names and integer labels, make sure to start labels at 1 instead of 0 otherwise there will be overlap with the background class (0) produced from the model.
 - The ImageTransform and BboxTransform functions are useful for making transformations to the images and bboxes in the getitem function. 
 - Make sure that the bboxes are returned in the following format: [ymin, xmin, ymax, xmax]. This is the format our model expects. 
 - filter_images: depending on how you choose to store your data, this function might not be required. In [pascal.py](pascal.py), we filter based on the width of the bounding boxes. Some additional things to possibly filter on are include removing crowd bounding boxes or remove images that are too easy/hard should you have this information. If no filtering is required, just pass and never call filter function. 
 - get_ann_info: must return a **dictionary** with "bboxes", "labels", and ""bboxes" as keys and numpy arrays as values
 - get_labels: must return a **list** of class names
 - getitem: must return the image, image metadata, bboxes, and labels. Lines 157-172 in pascal.py show how to construct the image metadata array.  

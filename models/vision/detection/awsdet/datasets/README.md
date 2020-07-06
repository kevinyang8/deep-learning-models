
# Using Custom Datasets

For an example of how to implement custom datasets, see pascal.py, which implements the [Pascal VOC 2012 dataset]([http://host.robots.ox.ac.uk/pascal/VOC/voc2012/](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)). 

Below is some information that might help with the implementation.

 - when building a mapping between class names and integer labels, make sure to start labels at 1 instead of 0 otherwise there will be overlap with the background class (0) produced from the model.
 - the ImageTransform and BboxTransform functions are useful for making transformations to the images and bboxes in the getitem function. 
 - make sure that the bboxes are returned in the following format: [ymin, xmin, ymax, xmax]. This is the format our model expects. 
 - filter_images: depending on how you choose to store your data, this function might not be required. In pascal.py, we filter based on the width of the bounding boxes. Some additional things to possibly filter on are include removing crowd bounding boxes or remove images that are too easy/hard should you have this information. If no filtering is required, just pass and never call filter function. 
 - get_ann_info: must return a **dictionary** with "bboxes", "labels", and ""bboxes" as keys and numpy arrays as values
 - get_labels: must return a **list** of class names
 - getitem: must return the image, image metadata, bboxes, and labels. Lines 157-172 in pascal.py show how to construct the image metadata array.  

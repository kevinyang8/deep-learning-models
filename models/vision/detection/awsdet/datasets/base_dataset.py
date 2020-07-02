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
        
    

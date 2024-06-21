from .jasper_ridge import JasperRidgeDataset
from .urban import UrbanDataset

dataset_factory = {
    'jasper_ridge': JasperRidgeDataset,
    'urban': UrbanDataset
}
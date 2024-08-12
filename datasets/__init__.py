from .jasper_ridge import JasperRidgeDataset, MotionCodeJasperRidge
from .urban import UrbanDataset

dataset_factory = {
    'jasper_ridge': JasperRidgeDataset,
    'jasper_ridge_pixel': MotionCodeJasperRidge,
    'urban': UrbanDataset
}
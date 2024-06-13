from .basic_model import CNN
from .ca_siamese_unet import CASiameseUNet
from .unet import UNet
from .sam_siamese_unet import SamSiameseUNet

model_factory = {
    'ca_siamese': CASiameseUNet,
    'unet': UNet,
    'sam_siamese': SamSiameseUNet,
    'cnn': CNN
    
}
model_args = {
    'ca_siamese': (6, 3, 256, 4),
    'unet': (3, 256, 4),
    'sam_siamese': (6, 3, 256, 4),
    'cnn': (3, 4)
}
from einops import rearrange, reduce, repeat, pack, unpack
from datasets.cave_dataset import CAVEDataset
from mogptk.data import Data
import mogptk
data_path = "./datasets/data/CAVE"


# def prepare_point_ds(dataset):
#     Xt = []
#     Yt = []
#     for items in dataset:
#         y, z, x_gt, _  = items
#         xt = rearrange(z, 'c h w -> (h w) c')
#         yt = rearrange(x_gt, 'c h w -> (h w) c')
#         Xt.append(xt)
#         Yt.append(yt)

#     Xt, ps = pack(Xt, "* c") 
#     Yt, ps = pack(Yt, "* c")
#     return Xt, Yt


def prepare_point_ds(dataset):
    Xt = []
    Yt = []
    for items in dataset:
        y, z, x_gt, _  = items
        xt = rearrange(z, 'c h w -> (h w) c')
        yt = rearrange(x_gt, 'c h w -> (h w) c')
        Xt.append(xt)
        Yt.append(yt)

    Xt, ps = pack(Xt, "* c") 
    Yt, ps = pack(Yt, "* c")
    ds = mogptk.DataSet()
    for c in range(Yt.shape[1]):
        print(Xt.shape, Yt[:, c].shape)
        d = Data(Xt, Yt[:, c])
        ds.append(d)
    
    return ds




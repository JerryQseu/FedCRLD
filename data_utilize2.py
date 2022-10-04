import os
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
from skimage.transform import resize
import SimpleITK as sitk
import random
from scipy import ndimage,misc
from aug_tool import  Crop, MirrorTransform, SpatialTransform
from batchgenerators.transforms import GammaTransform

class PreQuantilePercent():
    """Quantile(分位数) normalization class."""
    name = 'quantile'

    def __init__(self, percent=96):
        """Initializer.

        Args:
            percent: int, Above this percent the data are considered outliers.
        """
        self.percent = percent

    def __call__(self, data):
        """ Processing the percentile normalization.
        Set all outlier values higher than a given percentage to the highest
        acceptable value.

        Args:
            data: ndarray, numpy array data to apply the preprocessing.

        Returns:
            ndarray data without outlier
        """
        tresh = np.percentile(data, self.percent)
        idx = data > tresh
        data[idx] = data.min()
        data[idx] = data.max()
        return data
def read_image(image,spacing,spacing_target):
    new_shape = (int(np.round(spacing[0] / spacing_target[0] * float(image.shape[0]))),
                 int(np.round(spacing[1] / spacing_target[1] * float(image.shape[1]))),
                 int(np.round(spacing[2] / spacing_target[2] * float(image.shape[2]))))
    image = resize(image, new_shape, order=3, mode='edge')
    image = image.astype(np.float32)#[d,w,h]
    return image

def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')

def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res

def read_label(image,spacing,spacing_target):

    tem = convert_to_one_hot(image)
    vals = np.unique(image)
    result = []
    for i in range(len(tem)):
        result.append(resize_image(tem[i].astype(float), spacing, spacing_target, 1)[None])
    image = vals[np.vstack(result).argmax(0)]
    return image

def normor(image):
    image -=image.mean()
    image /=image.std()
    return image

Path_4d = './data2/image/'
Path_lab = './data2/label/'

img_3D = []

for root,dirs,files in os.walk(Path_4d):
    for file in files:
        fourd_path = os.path.join(root, file)

        img_3D.append(fourd_path)

class Data(Dataset.Dataset):
    def __init__(self,im_3D):
        self.im_3D = im_3D

    def __len__(self):
        return len(self.im_3D)

    def __getitem__(self, index):

        img3 = sitk.ReadImage(self.im_3D[index])
        spacing = np.array(img3.GetSpacing())[[2, 1, 0]]  ###[z,x,y]
        image3 = sitk.GetArrayFromImage(img3).astype(float)
        image3 = image3[:, :, :]

        # a = self.im_3D[index][14:-7]+ '_gt.nii.gz'

        label = sitk.ReadImage(Path_lab + self.im_3D[index][14:-7] + '_gt.nii.gz')
        label = sitk.GetArrayFromImage(label).astype(float)
        label = label[:, :, :]

        spacing_target = (10, 1.25, 1.25)
        spacing_target = list(spacing_target)
        spacing_target[0] = spacing[0]

        image3 = read_image(image3, spacing, spacing_target)
        label = read_label(label, spacing, spacing_target)

        d, w, h = image3.shape
        shape = (w, h)



        mirror_transform = MirrorTransform()
        spatial_transform = SpatialTransform(patch_center_dist_from_border=np.array(shape) // 2,
                                             do_elastic_deform=True,
                                             alpha=(0., 1500.),
                                             sigma=(30., 50.),
                                             do_rotation=True,
                                             angle_x=(-np.pi, np.pi),
                                             angle_y=(0, 0),
                                             angle_z=(0, 0),
                                             do_scale=True,
                                             scale=(0.75, 1.25),
                                             border_mode_data='constant',
                                             border_cval_data=0,
                                             order_data=3,
                                             random_crop=False)

        mirror_code = mirror_transform.rand_code()
        label = mirror_transform.augment_mirroring(label, mirror_code)
        image3 = mirror_transform.augment_mirroring(image3, mirror_code)

        coords = spatial_transform.rand_coords(shape)
        le = convert_to_one_hot(label)
        le1 = le
        vale = np.unique(label)
        resultlabel = []
        for i in range(len(le)):
            for j in range(d):
                lae = le[i,j,:,:].astype(float)
                le1[i,j,:,:] = spatial_transform.augment_spatial(lae, coords, is_label=True)[None]

            resultlabel.append(le1[i:i+1,:,:,:])

        label = vale[np.vstack(resultlabel).argmax(0)]

        for j in range(d):
            image3[j, :, :] = spatial_transform.augment_spatial(image3[j, :, :], coords)


        d, w, h = image3.shape
        if w > h:
            beishu = w / 200
        else:
            beishu = h / 200

        image = resize(image3, (d, w // beishu, h // beishu), order=3, mode='edge')
        image3 = image.astype(np.float32)  # [d,w,h]

        tem_label = convert_to_one_hot(label)
        vals_label = np.unique(label)
        result_vals_label = []
        for i in range(len(tem_label)):
            result_vals_label.append(
                resize(tem_label[i].astype(float), (d, w // beishu, h // beishu), order=1, mode='edge')[None])
        label = vals_label[np.vstack(result_vals_label).argmax(0)]



        label = convert_to_one_hot(label)

        fenwe = PreQuantilePercent()
        image3 = fenwe(image3)

        image3_norm = normor(image3)
        image3_norm = image3_norm[np.newaxis, :, :, :]
        name = self.im_3D[index]

        return image3_norm, label, name

train_data = Data(img_3D)
import logging
from dataclasses import dataclass

import bg_space as bg
import numpy as np
import skimage.transform
from bg_atlasapi import BrainGlobeAtlas
from brainglobe_utils.general.system import get_num_processes
from tqdm import tqdm
import multiprocessing as mp
from functools import partial 
from scipy import ndimage as ndi 
import glob, os 


def initialise_brainreg(atlas_key, data_orientation_key, voxel_sizes):
    scaling_rounding_decimals = 5
    n_free_cpus = 2
    atlas = BrainGlobeAtlas(atlas_key)
    source_space = bg.AnatomicalSpace(data_orientation_key)

    scaling = []
    for idx, axis in enumerate(atlas.space.axes_order):
        scaling.append(
            round(
                float(voxel_sizes[idx])
                / atlas.resolution[
                    atlas.space.axes_order.index(source_space.axes_order[idx])
                ],
                scaling_rounding_decimals,
            )
        )

    n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)
    load_parallel = n_processes > 1

    logging.info("Loading raw image data")
    return (
        n_free_cpus,
        n_processes,
        atlas,
        scaling,
        load_parallel,
    )


def downsample_and_save_brain(img_layer, scaling):
    first_frame_shape = skimage.transform.rescale(
        img_layer.data[0], scaling[1:2], anti_aliasing=True
    ).shape
    preallocated_array = np.empty(
        (img_layer.data.shape[0], first_frame_shape[0], first_frame_shape[1])
    )
    print("downsampling data in x, y")
    for i, img in tqdm(enumerate(img_layer.data)):
        down_xy = skimage.transform.rescale(
            img, scaling[1:2], anti_aliasing=True
        )
        preallocated_array[i] = down_xy

    first_ds_frame_shape = skimage.transform.rescale(
        preallocated_array[:, :, 0], [scaling[0], 1], anti_aliasing=True
    ).shape
    downsampled_array = np.empty(
        (first_ds_frame_shape[0], first_frame_shape[0], first_frame_shape[1])
    )
    print("downsampling data in z")
    for i, img in tqdm(enumerate(preallocated_array.T)):
        down_xyz = skimage.transform.rescale(
            img, [1, scaling[0]], anti_aliasing=True
        )
        downsampled_array[:, :, i] = down_xyz.T
    return downsampled_array

def _resample_img_slice(img, scaling=(1,1)):
    return ndi.zoom(img, scaling, order=1, mode='reflect')

def downsample_brain_fast(img_layer, scaling, num_workers):
    if not num_workers:
        return downsample_and_save_brain(img_layer, scaling)
    else:
        ## Method 1: subsampling in z
        # z_dim = img_layer.data.shape[0]
        # zs = np.round(np.arange(0,z_dim,1/scaling[0])).astype('int')
        # imgs = [img_layer.data[z] for z in zs]
        # f = partial(_resample_img_slice, scaling = scaling[1:])
        # p = mp.Pool(num_workers)
        # final_img = list(tqdm(p.imap(f,imgs), total = len(imgs)))
        # p.close()
        # p.join()
        # return np.asarray(final_img)

        ## Method 2: downsample in x,y, then in z 
        print("Downsample in x,y...")
        z_dim = img_layer.data.shape[0]
        zs = np.arange(z_dim)
        imgs = [img_layer.data[z] for z in zs]
        f = partial(_resample_img_slice, scaling = scaling[1:])
        p = mp.Pool(num_workers)
        final_img = list(tqdm(p.imap(f,imgs), total = len(imgs)))
        p.close()
        p.join()
        final_img = np.asarray(final_img)

        # Now downsample in z 
        print("Downsample in z...")
        x_dim = final_img.shape[2]
        xs = np.arange(x_dim)
        imgs = [final_img[:,:,x] for x in xs]
        f = partial(_resample_img_slice, scaling = ((scaling[0],1))
        p = mp.Pool(num_workers)
        final_imgs = list(tqdm(p.imap(f,imgs), total=len(imgs)))
        p.close()
        p.join()

        final_img = np.dstack(final_imgs)
        return np.asarray(final_img)


@dataclass
class NiftyregArgs:
    """
    Class for niftyreg arguments.
    """

    affine_n_steps: int
    affine_use_n_steps: int
    freeform_n_steps: int
    freeform_use_n_steps: int
    bending_energy_weight: float
    grid_spacing: float
    smoothing_sigma_reference: float
    smoothing_sigma_floating: float
    histogram_n_bins_floating: float
    histogram_n_bins_reference: float
    debug: bool

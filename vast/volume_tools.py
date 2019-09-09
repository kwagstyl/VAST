import h5py
import numpy as np
import pyminc.volumes.factory as pyminc
from skimage.transform import rescale
import re
import os

def crop_and_resample(arr, crop, scale):
    """Crop and resize the data"""
    cropped_array=np.zeros((crop[2]-crop[0],crop[3]-crop[1]))
    #crop to ensure the box is inside the image. Catches margins outside bigbrain vol.

    arr_crop = arr[np.max([0,crop[0]]):np.min([arr.shape[0],crop[2]]), 
                  np.max([0,crop[1]]):np.min([arr.shape[1],crop[3]])]
    cropped_array[np.max([0,-crop[0]]):arr_crop.shape[0]+np.max([0,-crop[0]]),
                 np.max([0,-crop[1]]):arr_crop.shape[1]+np.max([0,-crop[1]])] = arr_crop
    result=rescale(cropped_array, 1/scale, order=0,anti_aliasing=False,preserve_range=True,multichannel=False)
    return result

def blockify_from_hdf5(hdf5_filenames,output_dir, margin=20, resolution=100):
    scale = np.round(resolution/20).astype(int)
    h5file=h5py.File(hdf5_filenames[0],'r+')
    file_labels=h5file['mask'].attrs['labels']
    h5file.close()
    for area_label in file_labels[1:]:
        print('generating block for area: '+area_label.decode())
        area_index=np.where(file_labels==area_label)[0][0]
        generate_block_for_area(hdf5_filenames, output_dir, area_label, area_index, margin, scale)
    return


def generate_block_for_area(hdf5_filenames, output_dir, area_label, area_index, margin=20, scale =5):
    """create a minc block from the hdf5 files"""
    block,limits,sections=calculate_bounding_box(hdf5_filenames,area_index,margin=margin,scale=scale)
    template_filename=re.sub(r'\d\d\d\d','{}',hdf5_filenames[0])
    for section in range(block.shape[0]):
        im= h5py.File(template_filename.format(sections[section*scale]))['mask']['pyramid']['00'][:]
    
        im = (im==area_index).astype(int)
        #crop and rescale image
        scaled=crop_and_resample(im,limits,scale)
        block[section]=scaled
    
    height=im.shape[0]
    #invert_y as minc reads from bottom
    block=np.flip(block,1)
    ## pad in the section direction
    block = np.pad(block,((np.round(margin/scale).astype(int),np.round(margin/scale).astype(int)),
                          (0,0),(0,0)), 'constant',constant_values=0)
    original_origin = np.array([-70.02,-58.6777778, -70.5666667])
    start_section=int(sections[0])
    y_offset= (start_section-margin)*0.02
    z_offset = (height- limits[2]) * 0.021166667
    x_offset = limits[1]*0.021166667

    crop_shift = np.array([y_offset,z_offset,x_offset])
    new_origin= original_origin + crop_shift
    steps=(0.02*scale, 0.021166667*scale, 0.021166667*scale)
    out_vol = pyminc.volumeFromData(os.path.join(output_dir,area_label.decode()+"_block.mnc"), block, dimnames=("yspace", "zspace", "xspace"), starts=tuple(new_origin), steps=steps, volumeType="uint")
    out_vol.writeFile()    
    return

def save_mnc_block(block_filename,block,origin=(0,0,0),resolution=(1,1,1),dtype="ubyte", dimnames=("xspace", "yspace", "zspace")):
    """save block to volumetric filename with correct resolution etc
        currently only supports minc, could be extended"""
    
    out_vol = pyminc.volumeFromData(block_filename, block, dimnames=dimnames, starts=tuple(origin), steps=tuple(resolution), volumeType=dtype)
    out_vol.writeFile()    
    return

def calculate_bounding_box(hdf5_filenames,area_index,margin=20, scale=5):
    """ calculate bounding box for a given label.
    returns empty box, limits in x/y dimensions and sections to be imported.
    a margin is added to mitigate clipping - margin is pixels at 20um
    scale determines resolution of bbox. 5 >> 100um"""
    #downsampling scale indicates the level of the pyramid. 07 is 128 times downsampled to 00
    downsampling_scale=128
    overall_xmin,overall_ymin,overall_xmax,overall_ymax = [np.inf,np.inf,0,0]
    #create grid for indices
    h5file=h5py.File(hdf5_filenames[0],'r+')
    image_mask=h5file['mask']['pyramid']['07'][:]
    height, width = image_mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(width),np.arange(height))
    h5file.close()
    
    area_sections=[]
    for file_name in hdf5_filenames:
        
        h5file=h5py.File(file_name,'r+')
        image_mask=h5file['mask']['pyramid']['07'][:]
        #check if area is in the section
        if area_index in image_mask:
            area_sections.append(re.findall(r'\d\d\d\d',file_name)[0])
            xmin,ymin,xmax,ymax=bbox2(image_mask==area_index)
            overall_xmin = min(overall_xmin, xmin)
            overall_ymin = min(overall_ymin, ymin)
            overall_xmax = max(overall_xmax, xmax)
            overall_ymax = max(overall_ymax, ymax)
        h5file.close()
    #rescale to 20 micron and add margin
    limits=np.array([overall_xmin,overall_ymin,overall_xmax,overall_ymax])
    limits = limits* downsampling_scale + np.array([-margin,-margin,margin,margin])
    
    y_length=len(area_sections)
    #create limits of the block with margins in ALL directions
    dimensions_full_res = np.array([y_length, limits[2]-limits[0], limits[3]-limits[1]])
    #downscaled dimensions, rounded down
    scaled_dimensions = np.round(dimensions_full_res /scale).astype(int)
    #create empty block to fill with data
    block=np.zeros((scaled_dimensions.astype(int))).astype(int)
    return block,limits, area_sections
    
    
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return rmin,  cmin, rmax, cmax

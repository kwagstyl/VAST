import numpy as np
#import wget
import os
#import pyminc.volumes.factory as pyminc
from vast.surface_volume_mapper import SurfaceVolumeMapper  as svm
import matplotlib.pyplot as plt


class SectionPlotter():
    def __init__(self, world_coordinate, axis='y', resolution = np.array([0.0211667, 0.02, 0.0211667]),
                max_dimensions=np.array([6572, 7404, 5711]), origin=np.array([ -70.66667,-70.02,-58.7777]),
                hemisphere='left', data_dir=None, coordinates_dir = None):
        """class for generating, fetching and cropping 2D slices
        initialised with the world coordinate (i.e. 3D space) and axis (xyz) of the desired slice
        Given a hemisphere, will calculate"""
        self.axis = axis
        self.world_coordinate = world_coordinate
        self.origin = origin
        self.max_dimensions = max_dimensions
        self.resolution = resolution
        self.calculate_section_params()
        self.hemisphere = hemisphere
        self.coordinates_dir = coordinates_dir
        self.data_dir = data_dir
        self.initialise_svm_coordinates()
        
        
        
    def get_indices_section(self):
        """returns array of vertex indices of profiles needed to fill the section"""
        return np.unique(self.sv_map.volume_surf_coordinates['triangles'])
        
    def expand_indexed_profiles(self,indices,profiles):
        """Given a set of profiles indices and matching profiles, expand the profiles to be a matrix of profiles
        with zeros in the gaps"""
        profiles_full = np.zeros((655362,np.shape(profiles)[1]))
        profiles_full[indices]=profiles
        return profiles_full
        
    def plot_indexed_profiles(self, indices, profiles, interpolation='linear'):
        """plots indexed profiles back to 2D histology using surface to volume mapper"""
        expanded_profiles=self.expand_indexed_profiles(indices,profiles)
        block = self.orient_local_mncfile(np.squeeze(self.sv_map.map_profiles_to_block(expanded_profiles, interpolation=interpolation)))
        return block
        
    def initialise_svm_coordinates(self):
        """check if coordinates file has been calculated. If not, then calculate using surface_volume mapper"""
        
        
        self.coordinates_filename = os.path.join(self.coordinates_dir,'{}_{:04d}_{}_coordinates.hdf5'.format(
           self.axis_str, self.section_number, self.hemisphere,))
        print('looking for {}'.format(self.coordinates_filename))
        #generate or import the mapping here.
#        print(self.coordinates_filename,
#                     os.path.join(self.data_dir,'raw','white_{}.obj'.format(self.hemisphere)),
#                     os.path.join(self.data_dir,'raw','gray_{}.obj'.format(self.hemisphere)),
#                     self.resolution,
#                     self.section_dimensions,
#                     self.section_origin)
        self.sv_map = svm(filename = self.coordinates_filename,
                      white_surf= os.path.join(self.data_dir,'raw','white_{}.obj'.format(self.hemisphere)),
                      gray_surf= os.path.join(self.data_dir,'raw','gray_{}.obj'.format(self.hemisphere)),
                     resolution = self.resolution,
                      mask = None,
                      dimensions = self.section_dimensions,
                      origin = self.section_origin,
                      save_in_absence = True)
               
        
    def import_histology(self, section_dir):
        """import histology file and orientate correctly or download if doesn't exist"""
        filename = os.path.join(section_dir,'{}_{:04d}_histo.mnc'.format(self.axis_str,self.section_number))
        print(filename)
        if not os.path.isfile(filename):
            self.download_histology(filename)
        array_data = self.load_2D_mnc(filename)
        #squeeze down to 2D array
        #TODO if resolutions are not 
        array_data = self.orient_histology(array_data)
        return array_data
    
    def download_histology(self, filename):
        import wget
        """download histological section from the BigBrain ftp"""
        url = 'ftp://bigbrain.loris.ca/BigBrainRelease.2015/2D_Final_Sections/{}/Minc/pm{:04d}o.mnc'.format(
            self.axis_str,self.section_number)
        print(url)
        if not os.path.isfile(filename):
            wget.download(url,filename)
    
    def orient_histology(self,array_data):
        """Flipping images according to axis"""
        if self.axis=='y':
            return np.flipud(array_data)
        elif self.axis=='z':
            return np.rot90(array_data)

    def import_results_mncfile(self, filename):
        """load locally generated mnc file
        Checks, loads and orients correctly for plotting"""
        if not os.path.isfile(filename):
            print("Can't find {}".format(filename))
            print("Consider generating one with the .generate_results_mncfile")
        array_data = self.load_2D_mnc(filename)
        array_data = self.orient_local_mncfile(array_data)
       # array_data = self.filter_zeros(array_data)
        return array_data

    def orient_local_mncfile(self,array_data):
        """Flipping images according to axis"""
        if self.axis=='y':
            return np.rot90(array_data)
        elif self.axis=='z':
            return np.flipud(array_data)
        
    def filter_zeros(self,array_data):
        """set zeros to nan for transparent"""
        array_data[array_data==0]=np.nan
        return array_data

    def load_2D_mnc(self,filename):
        """loads and squeezes a 2D mncfile"""
        import pyminc.volumes.factory as pyminc
        mncfile=pyminc.volumeFromFile(filename)
        array_data = np.squeeze(np.array(mncfile.data))
        return array_data
    
#TODO generate results block, given profiles, coordinates etc.


#     def generate_results_mncfile(self, filename,experiment_folder=None,  results_file=None, coordinates_filename=None, test_hemisphere=None,
#                                 sal_type=None,mask_regions=None):
#         """Based on the plot_saliencies script, will create the volume to surface mapping file and
#         fill with specified data"""
#         if os.path.isfile(filename):
#             block = self.import_results_mncfile(filename)
#             #TODO check dimensions match expected
#             return block
#         #TODO implement for both hemispheres. 
#         mncless_filename = filename.replace('.mnc','')
#         sp.plot_saliency(experiment_folder,  results_file, coordinates_filename, test_hemisphere, mask_regions,
#                          self.resolution, self.section_dimensions,self.section_origin, sal_type, mncless_filename)
#         return self.import_results_mncfile(filename)



    
    def calculate_section_params(self):
        """calls a range of functions to set up the self class"""
        self.get_axis_info()
        self.get_section_number()
        #reset from whole brain block to section block dimensions and origin
        self.section_dimensions = self.max_dimensions.copy()
        self.section_dimensions[self.axis_index]=1
        self.section_origin = self.origin.copy()
        self.section_origin[self.axis_index] = self.world_coordinate

    def get_axis_info(self):
        """Get string and index describing axis"""
        axes=['x','y','z']
        axis_strings=['Sagittal','Coronal','Axial']
        self.axis_index=axes.index(self.axis)
        self.axis_str = axis_strings[self.axis_index]


    def get_section_number(self):
        """calculate section number"""
        self.section_number = np.round((self.world_coordinate-self.origin[self.axis_index])/self.resolution[self.axis_index]).astype(int)
        
    def autocrop_to_mask(self, all_images,mask, thr=0):
        """takes a set of images and crops each to the mask"""
        mask = mask>thr
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        for image in all_images.keys():
            all_images[image]= all_images[image][rmin:rmax,cmin:cmax]
        return all_images

    def plot_images_grid(self, all_images, image_keys=None,labels=False, crop=True):
        axes={}
        if crop:
            all_images = self.autocrop_to_mask(all_images, all_images['labels'])

        colourmaps=['viridis_r','rainbow','viridis','viridis','viridis','viridis','viridis','viridis','viridis']
        #if not specified, arbitary order
        if image_keys is None:
            image_keys=list(all_images.keys())
        a=plt.figure(figsize=(10,10))
        for i, sal in enumerate(image_keys):
            if i>0:
                axes[sal] = plt.subplot(3,3,i+1, sharex=axes[image_keys[0]],sharey=axes[image_keys[0]])
            else:
                axes[sal] = plt.subplot(3,3,i+1)
            axes[sal].set_title(sal)
            if 'diff' in sal:
                vmin = np.quantile(all_images[sal][all_images[sal]>0],.01)
                vmax = np.quantile(all_images[sal][all_images[sal]>0],.99)
                limits = np.max([np.abs(vmin),np.abs(vmax)])
                axes[sal].imshow(all_images[sal], cmap='RdBu',vmin=-limits,vmax=limits)
            elif 'labels' in sal or 'predictions' in sal:
                vmin=0.1
                vmax=3
                axes[sal].imshow(all_images[sal], cmap='rainbow',
                             vmin=vmin, vmax=vmax)
            else:
                vmin=np.quantile(all_images[sal][all_images[sal]>0],.001)
                vmax=np.quantile(all_images[sal][all_images[sal]>0],.999)
                axes[sal].imshow(all_images[sal], cmap=colourmaps[i],
                             vmin=vmin, vmax=vmax)
            axes[sal].axis('off')
        plt.tight_layout()
        return a





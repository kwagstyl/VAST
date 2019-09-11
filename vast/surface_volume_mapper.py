
import vast.io_mesh as io
import numpy as np
import h5py
import vast.volume_tools as vt
import vast.surface_tools as st
from functools import partial
import concurrent.futures
import os
import vast.math_helpers as mh

import time
class SurfaceVolumeMapper(object):
    
    def __init__(self, white_surf=None, gray_surf=None, resolution=None, mask=None, dimensions=None,
                  origin=None, filename=None, save_in_absence=False ):
        """Class for mapping surface data to voxels
        Always assumes axis order is xyz
        Args:
        
        white_surf/gray_surf : surface files to map to the volume.
        
        resolution:            specify voxel size in mm, 
                               either a float for uniform voxels or 3 floats for non-uniform
        
        mask:                  used to create bounding box which determines block. 
                               If dimesions and origins are specified, these are used instead.
                               dimensions: size of block to create in voxel steps.
                               
        dimensions:            size of volume block in voxel steps.
        
        origin:                Origin of volume (mm), 
                               i.e. location of one corner in real world coordinates
        
        filename:              path to hdf5 file containing precomputed coordinates. 
                               block attributes are checked against specified resolution,
                               dimensions and origin.
        
        """
        #initialise block coordinates dictionary
        self.volume_surf_coordinates={'voxel_coordinates':[],
                               'triangles':[],
                                'depths':[],
                               'triangle_coordinates':[]}
        
        if filename is not None:
            if os.path.isfile(filename):
                print('loading precomputed coordinates from {}'.format(filename))
                self.load_precomputed_coordinates(filename)
                print('loaded header info:')
                print('resolution: {}'.format(self.resolution))
                print('dimensions: {}'.format(self.dimensions))
                print('origin: {}'.format(self.origin))
                print('We hope this file matches your expectations')
                return
            else:
                print('precomputed coordinates file not found, recomputing...')
        
        #check resolution is 1D or 3D
        if resolution is not None:
            if isinstance(resolution, float):
                self.resolution = np.array([resolution,resolution,resolution])
            elif len(resolution)==3:
                self.resolution = resolution
            else:
                NotImplementedError
                
        print('loading surface meshes')
        if white_surf is not None:
            self.white_surface = io.load_mesh_geometry(white_surf)
            self.triangles_to_include = self.white_surface['faces']
        if gray_surf is not None:
            self.gray_surface = io.load_mesh_geometry(gray_surf)
        
        
        
        #check if mask. Calculate dimensions and origins from mask, unless these are specified.
        print('masking triangles')
        if mask is not None:
            self.mask=mask
            self.triangles_to_include = self.surface_mask_triangles(self.mask)
            if np.logical_and(type(dimensions) is not np.ndarray,type(dimensions )is not list) or np.logical_and(type(origin) is not np.ndarray,
                                                                                                        type(origin) is not list):
                          
                
                block_box = SurfaceVolumeMapper.bounding_box(np.vstack((self.gray_surface['coords'][mask],
                                                                        self.white_surface['coords'][mask])))
                self.dimensions = np.ceil((block_box[1]-block_box[0])/self.resolution).astype(int)
                self.origin = block_box[0]
                
            else:
                self.dimensions = np.array(dimensions)
                self.origin = np.array(origin)
        # if no mask, use block to filter triangles down
        else:
            self.dimensions = np.array(dimensions)
            self.origin = np.array(origin)
        
        print(self.dimensions, self.resolution)
        self.max_dimension = self.origin + self.dimensions * self.resolution
        print('Number of triangles in surface mask: {}'.format(len(self.triangles_to_include)))
        #return
        self.triangles_to_include = self.volume_mask_triangles(self.triangles_to_include)
        print('Number of triangles after masking to volume block: {}'.format(len(self.triangles_to_include)))
        
        #main function
        print('calculating coordinates')
        
        t1=time.time()
       # self.calculate_volume_surf_coordinates()
        t2=time.time()
        self.volume_surf_coordinates={'voxel_coordinates':[],
                               'triangles':[],
                                'depths':[],
                               'triangle_coordinates':[]}
       # print('non-parallel tool: ',t2-t1)
        t2=time.time()
        self.calculate_volume_surf_coordinates_parallel()
        t3=time.time()
        print('parallel tool: ',t3-t2)
        #save file if filename was give, file did not exist and save_in_absence is True
        if filename is not None and save_in_absence:
            print('saving out coordinates to {}'.format(filename))
            self.save_coordinates(filename)
        
    #functions

    def save_coordinates(self, filename):
        """save coordinates as hdf5 file
        """
        f=h5py.File(filename, 'w')
        f.attrs['resolution'] = self.resolution
        f.attrs['origin'] = self.origin
        f.attrs['dimensions'] = self.dimensions
        coords_files=['voxel_coordinates','triangles', 'depths','triangle_coordinates']
        for coords_file in coords_files:
            dset = f.require_dataset( coords_file ,
                                 shape = self.volume_surf_coordinates[coords_file].shape ,
                                 dtype = self.volume_surf_coordinates[coords_file].dtype,
                                 compression = "gzip", compression_opts = 9)
            dset[:] = self.volume_surf_coordinates[coords_file]
        f.close()
        
    def load_precomputed_coordinates(self, filename):
        """load coordinates from hdf5 file"""
        f=h5py.File(filename, 'r')
        self.resolution = f.attrs['resolution'] 
        self.origin = f.attrs['origin'] 
        self.dimensions = f.attrs['dimensions'] 
        coords_files=['voxel_coordinates','triangles', 'depths','triangle_coordinates']
        for coords_file in coords_files:
            self.volume_surf_coordinates[coords_file] = f[coords_file][:]
        f.close()
        
    def map_vector_to_block(self, vector_file, interpolation='linear'):
        """map values from vector file to voxel coordinates
        interpolation between vertices can either be:
        nearest neighbour or trilinear (weighted by barycentric)"""
        block = np.zeros(self.dimensions)
        tri_coords = self.volume_surf_coordinates['triangle_coordinates']
        triangles=self.volume_surf_coordinates['triangles']
        vc=self.volume_surf_coordinates['voxel_coordinates']
        if interpolation == 'linear':
            # interpolation
            block[vc[:,0],vc[:,1],vc[:,2]] = np.einsum('ij,ij->i', tri_coords, vector_file[triangles])
        elif 'nearest' in interpolation:
            #nearest is the maximum of the 3 coordinates
            nearest_index=triangles[tri_coords.max(axis=1,keepdims=1) == tri_coords]
            block[vc[:,0],vc[:,1],vc[:,2]] = vector_file[nearest_index]    
        return block
    
    def map_profiles_to_block(self, profiles, interpolation='linear'):
        """map values from vector file to voxel coordinates
        interpolation between vertices can either be:
        nearest neighbour or trilinear (weighted by barycentric)"""
        block = np.zeros(self.dimensions)
        tri_coords = self.volume_surf_coordinates['triangle_coordinates']
        triangles=self.volume_surf_coordinates['triangles']
        vc=self.volume_surf_coordinates['voxel_coordinates']
        depths=np.round((profiles.shape[1]-1)*self.volume_surf_coordinates['depths']).astype(int)
        print('reading depths')
        triangle_values=np.array([profiles[triangles[:,0],depths[:]],profiles[triangles[:,1],depths[:]],profiles[triangles[:,2],depths[:]]]).T
        print('writing to block')
        if interpolation == 'linear':
            block[vc[:,0],vc[:,1],vc[:,2]] = np.einsum('ij,ij->i', tri_coords, triangle_values)
        elif 'nearest' in interpolation:
            #nearest is the maximum of the 3 coordinates
            nearest_index=triangles[tri_coords.max(axis=1,keepdims=1) == tri_coords]
            block[vc[:,0],vc[:,1],vc[:,2]] = profiles[nearest_index,depths]    
        return block
    
    def save_block(self, filename,block, dtype="ubyte"):
        """calls save block from volume tools"""
        vt.save_mnc_block(filename, block,
                          origin=self.origin, resolution=self.resolution,
                         dtype=dtype)
        return
        
    
    
        
        
    def surface_mask_triangles(self, mask):
            """return triangles with all vertices in mask only"""
           
            return self.triangles_to_include[np.any(mask[self.triangles_to_include],axis=1)]
            
    def volume_mask_triangles(self, triangles_to_include):
            """return triangles with all vertices in block only"""
            vertex_indices = np.unique(triangles_to_include)
            #check if vertices inside block.+1 if above max, -1 if below origin
            g_include=(self.gray_surface['coords'][vertex_indices] > self.max_dimension).astype(int) - (self.gray_surface['coords'][vertex_indices] < self.origin).astype(int)
            w_include=(self.white_surface['coords'][vertex_indices] > self.max_dimension).astype(int) - (self.white_surface['coords'][vertex_indices] < self.origin).astype(int)
            #exclude if both are either 1 or -1, so if multiplied +1.
            exclude=np.any((g_include*w_include)==1,axis=1)
            
            #include if either grey or white is inside
            surface_mask_indices = vertex_indices[np.logical_not(exclude)]
            surface_mask=np.zeros(len(self.gray_surface['coords'])).astype(bool)
            surface_mask[surface_mask_indices] = True
            np.all(surface_mask[self.triangles_to_include],axis=1)
            #mask triangles
            return self.surface_mask_triangles(surface_mask)
        
    def calculate_volume_surf_coordinates(self):
        """calculate cortical depths and barycentric coordinates for voxels and triangles in volume 
        and store in data dictionary"""
        print('{} triangles included'.format(len(self.triangles_to_include)))
        for k,triangle in enumerate(self.triangles_to_include):
            if k % 10000 ==0:
                print('{}% done'.format(100*k/len(self.triangles_to_include)))
            prism=self.generate_prism(self.gray_surface['coords'],self.white_surface['coords'],triangle)
            bbox = SurfaceVolumeMapper.prism_bounding_box(prism)
            world_coords, voxel_coords= SurfaceVolumeMapper.voxel_world_coords_in_box(bbox,self.origin, self.resolution, self.dimensions)
            wc, vc, depths, tri_coords=SurfaceVolumeMapper.get_depth_and_barycentric_coordinates_for_prism(world_coords,voxel_coords,prism)
            if len(vc)>0:
                self.volume_surf_coordinates['voxel_coordinates'].extend(vc.tolist())
                self.volume_surf_coordinates['depths'].extend(depths.tolist())
                self.volume_surf_coordinates['triangles'].extend(np.tile(triangle,(len(depths),1)).tolist())
                self.volume_surf_coordinates['triangle_coordinates'].extend(tri_coords.tolist())
        lv=len(self.volume_surf_coordinates['voxel_coordinates'])
        ld=len(self.volume_surf_coordinates['depths'])
        for key in self.volume_surf_coordinates.keys():
            self.volume_surf_coordinates[key] = np.array(self.volume_surf_coordinates[key])
        assert lv==ld,'lengths dont match depths={}voxel_coords{}'.format(ld,lv)
        return
    
    def calculate_volume_surf_coordinates_parallel(self):
        """calculate depths and barycentric coordinates for voxels and triangles in volume
        in parallel"""
        num_process=3
        volume_surf_coordinates={'voxel_coordinates':[],
                               'triangles':[],
                                'depths':[],
                               'triangle_coordinates':[]}
        
        subsets = np.array_split(np.arange(len(self.triangles_to_include)),num_process)
        func = partial(SurfaceVolumeMapper.calculate_volume_surf_coordinates_one_prism,
                       self.gray_surface['coords'],self.white_surface['coords'],
                       self.triangles_to_include,
                       self.origin, self.resolution, self.dimensions, subsets)
        t1=time.time()
        #Threading doesn't work here but process pool does.
        with concurrent.futures.ProcessPoolExecutor(num_process) as pool:
            store = list(pool.map(func,range(len(subsets))))
        for pool_output in store:
            for key in self.volume_surf_coordinates.keys():
                self.volume_surf_coordinates[key].extend(pool_output[key])
        for key in self.volume_surf_coordinates.keys():
            self.volume_surf_coordinates[key] = np.array(self.volume_surf_coordinates[key])
        t2=time.time()
        print('function time: ',t2-t1)
        #for key in volume_surf_coordinates.keys():
        #    volume_surf_coordinates[key] = [x for x in volume_surf_coordinates[key] if x]
        #    self.volume_surf_coordinates2[key]=np.array([item for sublist in volume_surf_coordinates[key] for item in sublist])
        #t3=time.time()
        #print('sorting time: ',t3-t2)
        return

            
    @staticmethod        
    def calculate_volume_surf_coordinates_one_prism(
        gray_surface_coords,white_surface_coords,
        triangles,
        origin, resolution, dimensions, subset_triangles,k):
        """calculate on subset of triangles"""
        store_surf_coordinates={'voxel_coordinates':[],
                               'triangles':[],
                                'depths':[],
                               'triangle_coordinates':[]}
        percentage_divider=np.round(len(subset_triangles[k])/10).astype(int)
        for counter,tri_index in enumerate(subset_triangles[k]):
            if counter % percentage_divider ==0:
                print('Process {} is {}% done'.format(k,np.round(100*counter/len(subset_triangles[k]))))
            prism = SurfaceVolumeMapper.generate_prism(gray_surface_coords, white_surface_coords, triangles[tri_index])
            bbox = SurfaceVolumeMapper.prism_bounding_box(prism)
            world_coords, voxel_coords= SurfaceVolumeMapper.voxel_world_coords_in_box(bbox,origin, resolution, dimensions)
            wc, vc, depths, tri_coords=SurfaceVolumeMapper.get_depth_and_barycentric_coordinates_for_prism(world_coords,voxel_coords,prism)
            #if some coordinates are returned, then store these
            if len(vc)>0:
                store_surf_coordinates['voxel_coordinates'].extend(vc.tolist())
                store_surf_coordinates['depths'].extend(depths.tolist())
                store_surf_coordinates['triangles'].extend(np.tile(triangles[tri_index],(len(depths),1)).tolist())
                store_surf_coordinates['triangle_coordinates'].extend(tri_coords.tolist())
        return store_surf_coordinates
        
            
    @staticmethod
    def generate_prism(gray_surface_coords,white_surface_coords,triangle):
        """return coordinates for prism in a dictionary
        with two triangles
        ordering is g1,g2,g3 - w1,w2,w3"""
        prism_coordinates={'g_triangle':gray_surface_coords[triangle],'w_triangle':white_surface_coords[triangle]}
        return prism_coordinates
    
    
   # def generate_prism(self,triangle):
   #     """return coordinates for prism in a dictionary
   #     with two triangles
   #     ordering is g1,g2,g3 - w1,w2,w3"""
   #     prism_coordinates={'g_triangle':self.gray_surface['coords'][triangle],'w_triangle':self.white_surface['coords'][triangle]}
   #     return prism_coordinates
    
    
    @staticmethod
    def bounding_box(coords):
        """calculate bounding box for input coordinates"""
        mins=np.min(coords,axis=0)
        maxs=np.max(coords,axis=0)
        return mins, maxs





    
    @staticmethod
    def prism_bounding_box(prism):
        """returns the two defining corners of a box enclosing the prism.
        i.e. the minimum and maximum values in the 3 dimensions."""
        return SurfaceVolumeMapper.bounding_box(np.vstack((prism['g_triangle'],prism['w_triangle'])))
    
    @staticmethod
    def voxel_world_coords_in_box(box, origin_offset, voxel_resolution, dimensions):
        """calculate which voxels from a block/slice/volume are located a box (world coordinates)
        returns coordinates of voxels and voxel indices
        Assumes axis orderings of box, origin_offset, voxel resolution and dimensions are all the same
        Usually xyz"""
        #calculate box corners in voxel indices. Ensure voxel coordinates are non-negative and do not
        #exceed volume limits
        indices_min = np.min((np.max((np.floor((box[0] - origin_offset)/voxel_resolution),[0,0,0]),axis=0),dimensions),axis=0).astype(int)
        indices_max = np.min((np.max((np.ceil((box[1]- origin_offset)/voxel_resolution), [0,0,0]), axis=0),dimensions), axis=0).astype(int)
        if (indices_min == indices_max).all():
            #box not in volume block.
            return None, None
        #get a grid of coordinates
        voxel_coordinates=np.mgrid[indices_min[0]:indices_max[0],
             indices_min[1]:indices_max[1],
            indices_min[2]:indices_max[2]].T
        voxel_coordinates = np.reshape(voxel_coordinates,(voxel_coordinates.size//3,3))
        #convert to world coordinates
        world_coordinates=origin_offset+voxel_coordinates*voxel_resolution
        #mask out those not in block to speed up calculations on slices
        
        return world_coordinates, voxel_coordinates.astype(int)
    
    @staticmethod
    def get_exact_depth_multiple_coordinates(voxel_coords,prism,decimals=5):
        """returns exact coortical depth of point

        due to imprecisions in estimating roots of the cubic, it is advisable to round to desired accuracy.
        for 3mm cortex, decimals=5 gives an accuracy of 30 nanometers"""
        #solve for depth
        connecting_vectors = prism['w_triangle']-prism['g_triangle']
        connecting_inplane_vectors = np.array([connecting_vectors[2]-connecting_vectors[0],
                                               connecting_vectors[1]-connecting_vectors[0]])
        #k2 term of cp
        cross_product_connecting_vectors = np.cross(connecting_inplane_vectors[0],connecting_inplane_vectors[1])

        gray_inplane_vectors = np.array([prism['g_triangle'][2]-prism['g_triangle'][0],
                                         prism['g_triangle'][1]-prism['g_triangle'][0]])
        #const term of cp
        cross_product_gray_inplane_vectors = np.cross(gray_inplane_vectors[0],gray_inplane_vectors[1])

        #k term of cp
        cross_prod_gray_connecting1 = np.cross(gray_inplane_vectors[1], connecting_inplane_vectors[0])
        cross_prod_gray_connecting2 = np.cross(gray_inplane_vectors[0], connecting_inplane_vectors[1])
        cross_prod_gray_connecting_sum = -cross_prod_gray_connecting1+cross_prod_gray_connecting2

        g3 = prism['g_triangle'][2]
        v3 = connecting_vectors[2]
        g3_voxel_coords = g3-voxel_coords
        
        
        #precalculate fixed parts
        k3 = np.dot(cross_product_connecting_vectors,v3)
        k2_fixed=np.dot(v3,cross_prod_gray_connecting_sum)
        k2 = k2_fixed+ np.dot(cross_product_connecting_vectors,g3_voxel_coords.T) 
       #print(k2.shape)
        k1_fixed=np.dot(v3, cross_product_gray_inplane_vectors)
        k1 = k1_fixed + np.dot(cross_prod_gray_connecting_sum,g3_voxel_coords.T)       
        
        
        k0 = np.dot(cross_product_gray_inplane_vectors,g3_voxel_coords.T)
        ## TODO adapt real cubic solve so that the outputs work and match solve.
       # all_depths_c = mh.real_cubic_solve(k3, k2,k1,k0)
       # all_depths_c[np.logical_or(all_depths_c<0,all_depths_c>1)]=float('NaN')
        
        
        all_depths=np.zeros(len(voxel_coords))
        for k, voxel_coord in enumerate(voxel_coords):          
            #TODO replace with matrix roots function

            #depths = np.roots([k3,k2[k],k1[k],k0[k]])
            depths = mh.solve(k3, k2[k], k1[k], k0[k])
            are_real = np.isreal(depths)
            depths = np.round(np.real(depths[are_real]),decimals=decimals)
            depths = depths[np.logical_and(depths>=0,depths<=1.0)]
            if len(depths)==0:
                all_depths[k]=float('NaN')
            else:
                all_depths[k]=depths[0]

        #print(np.vstack((all_depths_c,all_depths)))
        return all_depths

    @staticmethod
    def barycentric_coordinates(p,tri):
        #solve to return coordinates as barycentric from 3 vertices of triangle.
        #Use outputs for linear interpolation
        a = (np.square(tri[0,0]-tri[2,0]) + np.square(tri[0,1]-tri[2,1]) + np.square(tri[0,2]-tri[2,2]))
        b = (tri[1,0]-tri[2,0])*(tri[0,0]-tri[2,0]) + (tri[1,1]-tri[2,1])*(tri[0,1]-tri[2,1]) + (tri[1,2]-tri[2,2])*(tri[0,2]-tri[2,2])
        c = b
        d = (np.square(tri[1,0]-tri[2,0]) + np.square(tri[1,1]-tri[2,1]) + np.square(tri[1,2]-tri[2,2]))
        f = (p[0] - tri[2,0])*(tri[0,0]-tri[2,0]) + (p[1]-tri[2,1])*(tri[0,1]-tri[2,1]) + (p[2]-tri[2,2])*(tri[0,2]-tri[2,2])
        g = (p[0] - tri[2,0])*(tri[1,0]-tri[2,0]) + (p[1]-tri[2,1])*(tri[1,1]-tri[2,1]) + (p[2]-tri[2,2])*(tri[1,2]-tri[2,2])
        chi = (d*f - b*g)/(a*d - b*c)
        eta = (-c*f + a*g)/(a*d - b*c)
        lambda1 = chi
        lambda2 = eta
        lambda3 = 1 - chi - eta
        return lambda1, lambda2, lambda3
    
    def barycentric_coordinates_matrix(p,tri):
    #solve to return coordinates as barycentric from 3 vertices of triangle.
        #Use outputs for linear interpolation
        a = (np.square(tri[:,0,0]-tri[:,2,0]) + np.square(tri[:,0,1]-tri[:,2,1]) + np.square(tri[:,0,2]-tri[:,2,2]))
        b = (tri[:,1,0]-tri[:,2,0])*(tri[:,0,0]-tri[:,2,0]) + (tri[:,1,1]-tri[:,2,1])*(tri[:,0,1]-tri[:,2,1]) + (tri[:,1,2]-tri[:,2,2])*(tri[:,0,2]-tri[:,2,2])
        c = b
        d = (np.square(tri[:,1,0]-tri[:,2,0]) + np.square(tri[:,1,1]-tri[:,2,1]) + np.square(tri[:,1,2]-tri[:,2,2]))
        f = (p[:,0] - tri[:,2,0])*(tri[:,0,0]-tri[:,2,0]) + (p[:,1]-tri[:,2,1])*(tri[:,0,1]-tri[:,2,1]) + (p[:,2]-tri[:,2,2])*(tri[:,0,2]-tri[:,2,2])
        g = (p[:,0] - tri[:,2,0])*(tri[:,1,0]-tri[:,2,0]) + (p[:,1]-tri[:,2,1])*(tri[:,1,1]-tri[:,2,1]) + (p[:,2]-tri[:,2,2])*(tri[:,1,2]-tri[:,2,2])
        chi = (d*f - b*g)/(a*d - b*c)
        eta = (-c*f + a*g)/(a*d - b*c)
        lambda1 = chi
        lambda2 = eta
        lambda3 = 1 - chi - eta
        return np.vstack((lambda1, lambda2, lambda3)).T
   
    
    @staticmethod
    def get_depth_and_barycentric_coordinates_for_prism(world_coords,voxel_coords,prism):
        """calculate the precise depth and barycentric coordinates within a prism
        of all world coordinates
        depth - fractional depth from gray to white surface
        barycentric - fractional distance from each vertex in triangle"""
        depths = SurfaceVolumeMapper.get_exact_depth_multiple_coordinates(world_coords,prism)
        #filter out coordinates not in the right depth
        world_coords = world_coords[~np.isnan(depths)]
        voxel_coords = voxel_coords[~np.isnan(depths)]
        depths=depths[~np.isnan(depths)]
        #calculate barycentric coordinates for remaining voxels
        vector=prism['w_triangle']-prism['g_triangle']
        barycentric_coords = np.zeros((len(depths),3))
        #for k, (world_coord, depth) in enumerate(zip(world_coords,depths)):
        #    barycentric_coords[k] = SurfaceVolumeMapper.barycentric_coordinates(world_coord, depth*vector +prism['g_triangle'])
        barycentric_coords = SurfaceVolumeMapper.barycentric_coordinates_matrix(world_coords, vector*np.tile(depths,(3,3,1)).T +np.tile(prism['g_triangle'],(len(depths),1,1)))
        #filter out coordinates outside of triangle
        exclude=np.logical_or(np.any(barycentric_coords<0,axis=1),np.any(barycentric_coords>1,axis=1))
        world_coords = world_coords[~exclude]
        voxel_coords = voxel_coords[~exclude]
        depths=depths[~exclude]
        barycentric_coords=barycentric_coords[~exclude]
        return world_coords, voxel_coords, depths, barycentric_coords


    

    

    
        

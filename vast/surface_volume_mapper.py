
import vast.io_mesh as io
import numpy as np
import h5py
import vast.volume_tools as vt
import vast.surface_tools as st

class SurfaceVolumeMapper(object):
    
    def __init__(self, white_surf=None, gray_surf=None, resolution=None, mask=None, dimensions=None,
                  origin=None, filename=None ):
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
        
        #check resolution is 1D or 3D
        print('loading surface meshes')
        if white_surf is not None:
            self.white_surface = io.load_mesh_geometry(white_surf)
            self.triangles_to_include = self.white_surface['faces']
        if gray_surf is not None:
            self.gray_surface = io.load_mesh_geometry(gray_surf)
        
        if resolution is not None:
            if len(resolution)==3:
                self.resolution = resolution
            elif len(resolution) == 1:
                self.resolution = [resolution,resolution,resolution]
            else:
                NotImplementedError
        
        #check if mask. Calculate dimensions and origins from mask, unless these are specified.
        print('masking triangles')
        if mask is not None:
            self.mask=mask
            self.triangles_to_include = self.surface_mask_triangles(self.mask)
            if dimensions is None or origin is None:
                block_box = SurfaceVolumeMapper.bounding_box(np.vstack((self.gray_surface['coords'][mask],
                                                                        self.white_surface['coords'][mask])))
                self.dimensions = np.ceil((block_box[1]-block_box[0])/self.resolution).astype(int)
                self.origin = block_box[0]
                
            else:
                self.dimensions = dimensions
                self.origin = origin
        # if no mask, use block to filter triangles down
        else:
            self.dimensions = dimensions
            self.origin = origin
            self.triangles_to_include = self.volume_mask_triangles(self.triangles_to_include)
        
        #initialise block coordinates dictionary
        self.volume_surf_coordinates={'voxel_coordinates':[],
                               'triangles':[],
                                'depths':[],
                               'triangle_coordinates':[]}
        
        #main function
        print('calculating coordinates')
        self.calculate_volume_surf_coordinates()
        

    def save_coordinates(self, filename):
        """save coordinates as hdf5 file
        """
        f=h5py.File(filename, 'w')
        f.attrs['resolution'] = self.resolution
        f.attrs['origin'] = self.origin
        f.attrs['dimensions'] = self.dimensions
        coords_files=['voxel_coordinates','triangles', 'depths','triangle_coordinates']
        for coords_file in coords_files:
            dset = f.require_dataset( 'voxel_coordinates' ,
                                 shape = self.volume_surf_coordinates[coords_file].shape ,
                                 dtype = type(self.volume_surf_coordinates[coords_file]),
                                 compression = "gzip", compression_opts = 9)
            dset[:] = self.volume_surf_coordinates[coords_file]
        f.close()
        
    def map_vector_to_block(self, vector_file, interpolation='linear'):
        """map values from vector file to voxel coordinates
        interpolation between vertices can either be:
        nearest neighbour or trilinear (weighted by barycentric)"""
        block = np.zeros(self.dimensions)
        tri_coords = np.array(self.volume_surf_coordinates['triangle_coordinates'])
        triangles=np.array(self.volume_surf_coordinates['triangles'])
        vc=np.array(self.volume_surf_coordinates['voxel_coordinates'])
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
        tri_coords = np.array(self.volume_surf_coordinates['triangle_coordinates'])
        triangles=np.array(self.volume_surf_coordinates['triangles'])
        vc=np.array(self.volume_surf_coordinates['voxel_coordinates'])
        depths=np.round((profiles.shape[1]-1)*np.array(self.volume_surf_coordinates['depths'])).astype(int)
        triangle_values=np.zeros((len(depths),3))
        print('reading depths')
        for k,d in enumerate(depths):
            profiles_triangle=profiles[triangles[k]]
            triangle_values[k,:]=profiles_triangle[:,d]
        print('writing to block')
        if interpolation == 'linear':
            block[vc[:,0],vc[:,1],vc[:,2]] = np.einsum('ij,ij->i', tri_coords, triangle_values)
        elif 'nearest' in interpolation:
            #nearest is the maximum of the 3 coordinates
            nearest_index=triangles[tri_coords.max(axis=1,keepdims=1) == tri_coords]
            block[vc[:,0],vc[:,1],vc[:,2]] = profiles[nearest_index,depths]    
        return block
    
    def save_block(self, filename,block, dtype="uint"):
        """calls save block from volume tools"""
        vt.save_mnc_block(block, filename,
                          origin=self.origin, resolution=self.resolution,
                         dtype=dtype)
        return
        
    
    
        
        
    def surface_mask_triangles(self, mask):
            """return triangles with all vertices in mask only"""
            return self.triangles_to_include[np.all(mask[self.triangles_to_include],axis=1)]
            
    def volume_mask_triangles(self, triangles_to_include):
            """return triangles with all vertices in block only"""
            vertex_indices = np.unique(triangles_to_include)
            max_dimension = self.origin + self.dimensions * self.resolution
            #check if vertices inside block
            g_include=np.logical_and(np.all(gray_surface['coords'][vertex_indices] > self.origin,axis=0),
                           np.all(gray_surface['coords'][vertex_indices] < max_dimension,axis=0))
            w_include=np.logical_and(np.all(white_surface['coords'][vertex_indices] > self.origin,axis=0),
                           np.all(white_surface['coords'][vertex_indices] < max_dimension,axis=0))
            #include if either grey or white is inside
            surface_mask = vertex_indices[np.logical_or(g_include,w_include)]
            #mask triangles
            return surface_mask_triangles(vertices_to_include)
        
    def calculate_volume_surf_coordinates(self):
        """calculate depths and barys for all triangles in volume 
        and store in data dictionary"""
        for triangle in self.triangles_to_include:
            prism=self.generate_prism(triangle)
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
        assert lv==ld,'lengths dont match depths={}voxel_coords{}'.format(ld,lv)
        return
            
    def generate_prism(self,triangle):
        """return coordinates for prism in a dictionary
        with two triangles
        ordering is g1,g2,g3 - w1,w2,w3"""
        prism_coordinates={'g_triangle':self.gray_surface['coords'][triangle],'w_triangle':self.white_surface['coords'][triangle]}
        return prism_coordinates
    
    
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

        k3 = np.dot(cross_product_connecting_vectors,v3)
        all_depths=np.zeros(len(voxel_coords))
        for k, voxel_coord in enumerate(voxel_coords):
            k2 = (np.dot(v3,cross_prod_gray_connecting_sum)+
             np.dot(cross_product_connecting_vectors,g3-voxel_coord))
            k1 = (np.dot(v3, cross_product_gray_inplane_vectors) +
             np.dot(cross_prod_gray_connecting_sum,g3-voxel_coord))
            k0 = (np.dot(cross_product_gray_inplane_vectors,g3-voxel_coord))
            depths = np.roots([k3,k2,k1,k0])
            depths = np.round(depths[np.isreal(depths)],decimals=decimals)
            depths = depths[np.logical_and(depths>=0,depths<=1.0)]
            if len(depths)==0:
                all_depths[k]=float('NaN')
            else:
                all_depths[k]=depths[0]
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
        for k, (world_coord, depth) in enumerate(zip(world_coords,depths)):
            barycentric_coords[k] = SurfaceVolumeMapper.barycentric_coordinates(world_coord, depth*vector +prism['g_triangle'])
        #filter out coordinates outside of triangle
        exclude=np.logical_or(np.any(barycentric_coords<0,axis=1),np.any(barycentric_coords>1,axis=1))
        world_coords = world_coords[~exclude]
        voxel_coords = voxel_coords[~exclude]
        depths=depths[~exclude]
        barycentric_coords=barycentric_coords[~exclude]
        return world_coords, voxel_coords, depths, barycentric_coords


    

    
        
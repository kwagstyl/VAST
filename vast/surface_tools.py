import numpy as np
from vast import io_mesh
from scipy.stats import mode
from vast.constants import civet_path
import os
import subprocess


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def normal_vectors(vertices,faces):
    norm = np.zeros( vertices.shape, dtype=vertices.dtype )
    tris = vertices[faces]
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    n=normalize_v3(n)
    norm[ faces[:,0] ] += n
    norm[ faces[:,1] ] += n
    norm[ faces[:,2] ] += n
    return normalize_v3(norm)

def spherical_np(xyz):
    pts=np.zeros(xyz.shape)
    xy=xyz[:,0]**2 + xyz[:,1]**2
    pts[:,0]=np.sqrt(xy+xyz[:,2]**2)
    pts[:,1]=np.arctan2(np.sqrt(xy),xyz[:,2])
    pts[:,2]=np.arctan2(xyz[:,1], xyz[:,0])
    return pts


def get_neighbours_from_tris(tris, label=None):
    """Get surface neighbours from tris
        Input: tris
         Returns Nested list. Each list corresponds 
        to the ordered neighbours for the given vertex"""
    n_vert=np.max(tris+1)
    neighbours=[[] for i in range(n_vert)]
    for tri in tris:
        neighbours[tri[0]].extend([tri[1],tri[2]])
        neighbours[tri[2]].extend([tri[0],tri[1]])
        neighbours[tri[1]].extend([tri[2],tri[0]])
    #Get unique neighbours
    for k in range(len(neighbours)):      
        if label is not None:
            neighbours[k] = set(neighbours[k]).intersection(label)
        else :
            neighbours[k]=f7(neighbours[k])
    return np.array(neighbours);


def get_neighbours(surfname):
    """return neighbours from surface filename"""
    surf=io_mesh.load_mesh_geometry(surfname)
    neighbours = get_neighbours_from_tris(surf['faces'])
    return neighbours



def f7(seq):
    #returns uniques but in order to retain neighbour triangle relationship
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))];

def compute_islands(area_txtfile,neighbours):
    """calculates islands with same label value"""
    islands=np.zeros_like(area_txtfile).astype(int)
        #array for quick indexing
    indices=np.arange(len(area_txtfile))
    #k is island counter
    k=0
    #while some vertices haven't been assigned
    while np.sum(islands==0)>0:
        k+=1
        #start with lowest unassigned vertex
        cluster=[np.min(np.where(islands==0)[0])]
        #set vertex to island value
        islands[cluster]=k
        #get label value (i.e. inside or outside label, 1/0)
        v_seed_label=area_txtfile[cluster[0]]

        old_cluster=0
        #while size of island k increases
        while np.sum(islands==k)>np.sum(old_cluster):
            #calculate new vertices
            added_vertices=islands==k-old_cluster
            #store for next comparison
            old_cluster=islands==k
            # for the new vertices
            for v in indices[added_vertices]:
                #get the neighbours
                neighbourhood=np.array(neighbours[v])
                #of the neighbours, which are in the same label and not yet part of the island.
                #set these to the island value
                islands[neighbourhood[np.logical_and(area_txtfile[neighbourhood]==v_seed_label,
                                                     islands[neighbourhood]!=k)]]=k
    return islands

##TODO tidy any label file??

def tidy_holes_binary(area_txtfile, neighbours,threshold_area=50, iterations=2):
    """fills small holes in binary surface annotation file.
    threshold_area is maximum area to be switched.
    iterations is the number of times looped through in case of nested holes"""
    #create empty surf file to be filled with island indices
    for i in range(iterations):
        islands=compute_islands(area_txtfile,neighbours)
        #then fill the holes
        new_area=np.copy(area_txtfile).astype(int)
        island_index,counts=np.unique(islands,return_counts=True)
        ordered=np.argsort(counts)
        for ordered_index in ordered:
            if counts[ordered_index]<threshold_area:
                island_i=island_index[ordered_index]
                new_area[islands==island_i]=(area_txtfile[islands==island_i][0]-1)*-1
        area_txtfile=new_area
    return new_area

def tidy_combined_atlas(combined_areas,overlaps,neighbours,threshold=100):
    """fill holes in combined_atlas, including overlapping areas
    fills values are the most frequent values on the border of the island"""
    all_areas_islands = compute_islands(combined_areas,neighbours)
    island_index,counts=np.unique(all_areas_islands,return_counts=True)
    ordered=np.argsort(counts)
    overlap_values=np.unique(combined_areas[overlaps])
    vertex_indices=np.arange(len(combined_areas)).astype(int)
    new_combined_areas=combined_areas.copy()
    for ordered_index in ordered:
        island_of_interest=all_areas_islands==island_index[ordered_index]
        island_value=combined_areas[island_of_interest][0]
        #only replace islands of 0s or overlaps
        if np.logical_and(counts[ordered_index]<threshold, island_value==0 or (overlaps[island_of_interest]).any()):        
            neighbours_island = get_ring_of_neighbours(island_of_interest, neighbours, vertex_indices, ordered=False)
            new_combined_areas[island_of_interest]=mode(new_combined_areas[neighbours_island])[0][0]
    return new_combined_areas


def get_ring_of_neighbours(island, neighbours, vertex_indices=None, ordered=False):
    """Calculate ring of neighbouring vertices for an island of cortex
    If ordered, then vertices will be returned in connected order"""
    if not vertex_indices:
        vertex_indices=np.arange(len(island))
    if not ordered:

        neighbours_island = neighbours[island]
        unfiltered_neighbours = []
        for n in neighbours_island:
            unfiltered_neighbours.extend(n)
        unique_neighbours = np.setdiff1d(np.unique(unfiltered_neighbours), vertex_indices[island])
        return unique_neighbours
            
def get_neighbouring_roi(area, neighbours,mask=None, scale = 1, uncertainty_zone_steps =0):
    """generate region of interest of same size as area from neighbouring vertices
    currently no buffer zone or mask is included.
    scale : size of neighbouring area relative to roi. 1 means number of vertices = roi
    uncertainty_zone_steps : int, number of vertex steps to take when expanding ring.
    """
    roi=area.astype(bool)
    mask=mask.astype(bool)
    target_number_of_vertices = np.round(np.sum(roi) *scale).astype(int)
    bool_uncertainty = np.zeros_like(roi)
    if uncertainty_zone_steps:
        uncertainty_zone_verts = []
        for k in range(uncertainty_zone_steps):
            ring_vertices = get_ring_of_neighbours(roi, neighbours)
            #alter roi so that neighbouring ring extends from outside the uncertainty zone
            roi[ring_vertices] = True
            uncertainty_zone_verts.extend(ring_vertices)
            if mask is not None:
                uncertainty_zone_verts=np.array(uncertainty_zone_verts)
                uncertainty_zone_verts=uncertainty_zone_verts[~mask[uncertainty_zone_verts]].tolist()
        bool_uncertainty[uncertainty_zone_verts]= True
    
    neighbouring_roi=[]
    #ring of uncertainty, to allow for inexact borders to be adjusted.
    while len(neighbouring_roi) < target_number_of_vertices:
        ring_vertices = get_ring_of_neighbours(roi, neighbours)
        #extend region of interest to get next neighbours in subsequent loop
        roi[ring_vertices] = True
        #store ring vertices
        neighbouring_roi.extend(ring_vertices)
        #mask vertices that you want to exclude eg medial wall. 
        if mask is not None:
            neighbouring_roi=np.array(neighbouring_roi)
            neighbouring_roi=neighbouring_roi[~mask[neighbouring_roi]].tolist()
            
    #shrink to correct length, equal to original roi
    neighbouring_roi = neighbouring_roi[:target_number_of_vertices]
    bool_roi = np.zeros_like(roi)
    bool_roi[neighbouring_roi]= True
    return bool_roi, bool_uncertainty


def load_printed_intensities(intensity_file):
    with open(intensity_file,'r') as f:
            for n in range(4): 
                f.readline()
            lst=f.readline().split('\t')[1:-1]
            vector=[]
            for i in lst:
                try:
                    vector.append(float(i))
                except ValueError:
                    vector.append(0)          
    return vector 




def create_weighted_midsurface(gray, white, surfdir):
    import tempfile
    tmpdir=os.path.join(tempfile.gettempdir(),'tmp_{}'.format(str(np.random.randint(1000))))
    os.mkdir(tmpdir)
    from vast import io_mesh
    """ create mid surface weighted by the curvature so that it is on the outside of 
         gyri and inside of sulci to prevent self-intersection and correspondence problems
    inputs : gray_surface filename
              white surface filename
              """
    if 'left' in gray:
        hemi = 'left'
    else :
        hemi = 'right'
    if 'rsl' in gray:
        flag='rsl'
    else:
        flag='native'
    if os.path.isfile(os.path.join(surfdir, 'weighted_mid_'+ hemi+'_'+flag+'.obj')):
        m = io_mesh.load_mesh_geometry(os.path.join(surfdir, 'weighted_mid_'+ hemi+'_'+flag+'.obj'))
        return m['coords']
    else:
        try:
            subprocess.call('{} {} {}'.format(os.path.join(civet_path,'average_objects'),os.path.join(tmpdir, 'mid.obj') , gray ,white), shell=True)
            subprocess.call('{} -mean_curvature -alpha 0.05 {} {}'.format(os.path.join(civet_path,'depth_potential'),os.path.join(tmpdir, 'mid.obj '), os.path.join(tmpdir,'curvature.txt')), shell=True)
            subprocess.call('{} -smooth 3 {} {} {}'.format(os.path.join(civet_path,'depth_potential'), os.path.join(tmpdir,'curvature.txt '), os.path.join(tmpdir, 'mid.obj '), os.path.join(tmpdir,'smcurvature.txt')), shell=True)
#normalise values between 0 and 1
            curv = np.loadtxt(os.path.join(tmpdir,'smcurvature.txt'))
            min_value=np.mean(curv)-2*np.std(curv)
            curv = (curv - min_value)
            max_value=np.mean(curv)+2*np.std(curv)
            curv = np.array([curv/max_value]).T
            np.clip(curv, 0,1,out=curv)
#load in surfaces
            g=io_mesh.load_mesh_geometry(gray)
            w=io_mesh.load_mesh_geometry(white)
            mid = g['coords']*(curv) + w['coords']*(1-curv)
            g['coords'] = mid
            io_mesh.save_mesh_geometry(os.path.join(surfdir, 'mid_'+ hemi+'_'+flag+'.obj'), g)
        except OSError:
            print('Error in creating create_weighted_midsurface, likely due to difficulties finding CIVET')
            raise
        return mid



def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def get_nearest_indices(surf1,surf2):
    """find nearest indices in surf2 for each vertex in surf1"""
    ListIndices=np.arange(len(surf2))
    RealIndices=np.zeros(len(surf1))
    checkdist=0.6
    for k, coord in enumerate(surf1):
        #reduce search to only nearby coordinates
        checkdist=0
        #if can't find a vertex, expand the search radius by 0.5
        indices=None
        while indices is None or len(indices) <2:
            checkdist+=0.5
            indices=ListIndices[np.logical_and(surf2[:,0]>coord[0]-checkdist,
                       np.logical_and(surf2[:,0]<coord[0]+checkdist,
                       np.logical_and(surf2[:,1]>coord[1]-checkdist,
                       np.logical_and(surf2[:,1]<coord[1]+checkdist,
                       np.logical_and(surf2[:,2]>coord[2]-checkdist,
                                      surf2[:,2]<coord[2]+checkdist)))))]
        tmpsurf2 = surf2[indices]
        #run nearest triangle, nearest edge and nearest vertex checking whether eta and chi are 0<x<1.
        Index = closest_node(coord, tmpsurf2)
        RealIndices[k]=indices[Index]
        if k % 50000 ==0:
            print(str(100.0*k/float(len(surf1))) + '% done')
    return RealIndices;




def get_windows(surfname):
    """generate indices of 19 ordered windows for each vertex in surface
        Central vertex is at position window[9] i.e. the middle of the window
           Either side and moving outwards for 3 steps are the nearest neighbours (5 neighbours have one neighbour repeated twice)
         Beyond these are the outer ring for 6 steps on either side.
         This arrangement mirrors the topology of windows in the 2D plane"""
    #     _ _
    #   /_\/_\
    #   \/_\/
    neighbours = get_neighbours(surfname)
    windows=np.zeros((len(neighbours),19),dtype='int')
    print(len(neighbours))
    for k in range(len(neighbours)):
        central=[k]
        inner_ring=neighbours[k]
        inner_patch=central+inner_ring
        outer_ring = []
        for v in inner_ring:
            extras = neighbours[v]
            outer_ring.extend(ordered_new_neighbours(inner_patch, extras))
        outer_ring = f7(outer_ring)
        if len(inner_ring) == 5:
            inner_ring.append(inner_ring[0])
        counter=-1
        while len(outer_ring) < 12:
            counter += 1
            outer_ring.append(outer_ring[counter])
            
        windows[k,:] = outer_ring[:6][::-1]+inner_ring[:3][::-1] + central + inner_ring[3:] + outer_ring[6:]
    return windows



def ordered_new_neighbours(patch, neighbourhood):
    """ for a vertex in a hexagonal patch, find the neighbours not already in the patch 
        return them in a consecutive order. Used to find the outer ring of a window and retain
         topology."""
    switch1 = False
    switch2 = False
    switch3 = False
    n_neighbours=len(neighbourhood)
    k=-1
    while switch3 == False:
        n = neighbourhood[k%n_neighbours]
        k+=1
        if switch1 == True and switch2 == True:
            if n in patch:
                return new_neighbours
            else:
                new_neighbours.append(n)
        elif switch1 == True:
            if n not in patch:
                new_neighbours= [n]
                switch2 = True
        else :
            if n in patch:
                switch1=True



import sys
sys.path.insert(0,'/Users/aswinchari/Documents/GitHub/vast')

import vast.io_mesh as io
import vast.surface_tools as st
import numpy as np
import nibabel as nb
import csv
import argparse
import os

flatten = lambda l: [item for sublist in l for item in sublist]
#load in data

def main(args):

    neighbours=st.get_neighbours(args.combined_surface)
    lh_cluster=nb.load(args.left_clusters)
    lh_cluster=np.asanyarray(lh_cluster.dataobj).ravel()
    rh_cluster=nb.load(args.right_clusters)
    rh_cluster=np.asanyarray(rh_cluster.dataobj).ravel()
    lh_cortex=nb.freesurfer.io.read_label(os.path.join(args.subject_id,'label','lh.cortex.label'))
    rh_cortex=nb.freesurfer.io.read_label(os.path.join(args.subject_id,'label','rh.cortex.label'))
    lh_area=nb.freesurfer.io.read_morph_data(os.path.join(args.subject_id,'surf','lh.area'))
    rh_area=nb.freesurfer.io.read_morph_data(os.path.join(args.subject_id,'surf','rh.area'))

    #get cortical vertices
    rh_cortex=rh_cortex+len(lh_cluster)
    cortical_vertices = np.hstack((lh_cortex,rh_cortex))

    #get clusters (add max left )
    rh_cluster[rh_cluster>0]+=np.max(lh_cluster)
    clusters= np.hstack((lh_cluster, rh_cluster))
    cluster_indices=np.unique(np.round(clusters))[1:]

    areas = np.hstack((lh_area,rh_area))
    n_vertices=len(neighbours)

    for cluster_index in cluster_indices:
        print(cluster_index)
        n_clusters=100
        print(np.sum(clusters==cluster_index))
        cluster_size=np.sum(clusters==cluster_index)
        
        for cluster in np.arange(n_clusters):

            if cluster==0:
              #  random_cluster_matrix[cluster,clusters==cluster_index]=1
                random_cluster_lists=[list(np.where(clusters==cluster_index)[0])]
            else:
                seed_vertex = np.random.choice(cortical_vertices)
                random_cluster_area=0
                old_cluster=neighbours[seed_vertex]
                while cluster_size > random_cluster_size:
                    new_cluster = st.f7(flatten(neighbours[old_cluster]))
                    random_cluster_area = len(new_cluster)
                    old_cluster=new_cluster
            random_cluster_lists.append(new_cluster[:cluster_size])

        with open(os.path.join(args.subject_id,'surf','random_clusters_{}.csv'.format(cluster_index)),'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(random_cluster_lists)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write cluster files to outputs')
    parser.add_argument('--subject-id', dest='subject_id', type=str, help='Subject id for loading files')
    parser.add_argument('--combined-surface', dest='combined_surface', type=str, help='Combined mesh surface')
    parser.add_argument('--left-clusters', dest='left_clusters', type=str, help='Left clusters mgh file')
    parser.add_argument('--right-clusters', dest='right_clusters', type=str, help='Left clusters mgh file')
    
    args = parser.parse_args()
    
    main(args)

    #get clusters (add max left )
    clusters= np.hstack((lh_cluster, rh_cluster+np.max(lh_cluster)))
    cluster_indices=np.unique(np.round(clusters))

    areas = np.hstack((lh_area,rh_area))
    n_vertices=len(neighbours)

    for cluster_index in cluster_indices:
        n_clusters=100
        #cluster_area=np.sum(areas[clusters==cluster_index])

        for cluster in np.arange(n_clusters):
            if cluster==0:
              #  random_cluster_matrix[cluster,clusters==cluster_index]=1
                random_cluster_lists=[list(np.where(clusters==cluster_index)[0])]
                base_cluster_length=np.sum(clusters==cluster_index)
            else:
                seed_vertex = np.random.choice(cortical_vertices)
                random_cluster_area=0
                old_cluster=neighbours[seed_vertex]
                while base_cluster_length > len(new_cluster):
                    new_cluster = st.f7(flatten(neighbours[old_cluster]))
                    #random_cluster_area = np.sum(areas[new_cluster])
                    old_cluster=new_cluster
                random_cluster_lists.append(new_cluster[:base_cluster_length])

        with open(os.path.join(subject_id,'surf','random_clusters_{}.csv'.format(cluster_index)),'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(random_cluster_lists)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write cluster files to outputs')
    parser.add_argument('--subject-id', dest='subject_id', type=str, help='Subject id for loading files')
    parser.add_argument('--combined-surface', dest='combined_surface', type=str, help='Combined mesh surface')
    parser.add_argument('--left-clusters', dest='left_clusters', type=str, help='Left clusters mgh file')
    parser.add_argument('--right-clusters', dest='right_clusters', type=str, help='Left clusters mgh file')
    
    args = parser.parse_args()
    
    main(args)

import nibabel as nib
import os
import numpy as np
import dipy
import pickle

from math import sin, cos, pi
from warnings import warn

class fib_upsampling:

	def __init__( self, tracks = 'data/fibers_connecting.trk', weights = '', nrRand = 1000 ):
		# declarations
		self.tracks = tracks
		self.weights = weights

		self.samplePoints = 50
		self.cutOff = 20
		self.pca_trafo = 'saved_pca_trafo.dat'
		self.pca_pts = 'saved_pca_pts.dat'
		self.clustering = 'saved_clustering.dat'
		self.nrClusters = 100
		self.nrRand = nrRand
		#self.minFibLength = 3

	def prepareTrafoClustering( self ):
		# Trafo and Clustering could be precomputed if streamlines are at least
		# rigidly aligned and reused for other subjects, otherwise it has to
		# be performed for each subject separately or another clustering scheme
		# needs to be used! (KMeans can be resource and time intensive for a
		# large number of streamlines)

		# run if pcaTrafo does not exist
		self.pcaTrafo()
		# run if clustering does not exist
		self.cluster()


	def pcaTrafo( self ):
		from dipy.tracking.streamline import set_number_of_points
		from matplotlib.mlab import PCA
		#use sklearn instead, matplotlib pca is depricated
		#from sklearn.decomposition import PCA

		#w = np.loadtxt( self.weights )

		#streams = nib.streamlines.load( self.tracks )
		#fibs = streams.streamlines#[w > 0]

		# load mrtrix streamlines ( subject space )
		streams = nib.streamlines.load( self.tracks )
		fibs = streams.streamlines

		fibs_resampled = self.resample( fibs )

		# calculate PCA transformation
		pcaResult = PCA( fibs_resampled, standardize = False )

		# trafo
		pcaTrafo = pcaResult.Wt

		# summed variance of dimensions 10 to 90
		sum( pcaResult.fracs[self.cutOff+1:] )  / sum( pcaResult.fracs )

		pca_trafo = self.pca_trafo

		# store pca-points for clustering
		np.savetxt( self.pca_pts, pcaResult.Y[:, :self.cutOff] )
		#np.savetxt( pca_w, w[w>0] )

		# remove points for storage purposes
		pcaResult.a = []
		pcaResult.Y = []

		# save pca trafo to file
		with open( pca_trafo, 'wb+' ) as fid:
		    pickle.dump( pcaResult, fid, -1 )

	def cluster( self ):
		from sklearn.cluster import KMeans

		# load pca trafo
		with open( self.pca_trafo, 'rb' ) as fid:
			pcaResult = pickle.load( fid )

		#load pca points
		pcaFibs = np.loadtxt( self.pca_pts )

		clust = KMeans( init = 'k-means++', n_clusters = self.nrClusters, n_init = 10 )
		clust.fit( pcaFibs )

		# save clustering scheme to file
		with open( self.clustering, 'wb+' ) as fid:
		    pickle.dump( clust, fid, -1 )

	def resample( self, fibs ):
		from dipy.tracking.streamline import set_number_of_points

		# resample streamlines
		strResample = set_number_of_points( fibs, self.samplePoints )

		# flip fibers with wrong start / end point
		fibLen = np.zeros( ( len( fibs ), 1 ), dtype = float )

		for i in range( len( strResample ) ):
			fib = strResample[i]
			diffFib = sum( fib[1:, :]-fib[:-1, :] )
			stdFib = ( np.std( fib[:, 0] ), np.std( strResample[0][:, 1] ), np.std( fib[:, 2] ) )
			fibLen[i] = sum( sum( ( ( fib[1:, :]-fib[:-1, :] )**2 ).T ) );
			if diffFib[stdFib.index( max( stdFib ) )] < 0:
				# flip fiber
				strResample[i][:] = fib[::-1, :]

		# from array to vector
		fibs = map( lambda f: f.ravel(), strResample )
		fibs = np.vstack( fibs )

		return fibs

	def upsample( self, tracks = '' ):
		### UNDER CONSTRUCTION ###
		# I already started writing some code here, but I didn't test it yet!!!
		if not tracks:
			tracks = self.tracks
		# load pca trafo
		with open( self.pca_trafo, 'rb' ) as fid:
			pcaResult = pickle.load( fid )

		# load clustering
		with open( self.clustering, 'rb' ) as fid:
			clust = pickle.load( fid )

		#streams = nib.streamlines.load( self.tracks )
		#fibs = streams.streamlines[w > 0]

		# load mrtrix streamlines ( subject space )
		streams = nib.streamlines.load( tracks )
		fibs = streams.streamlines

		rFibs = self.resample( fibs )

		pcaTrafo = pcaResult.Wt
		pcaFibs = ( rFibs-pcaResult.mu ).dot( np.linalg.inv( pcaTrafo ) )

		# classify
		clustInd = clust.predict( pcaFibs[:,:self.cutOff] )

		upFibs = []

		# loop through n_clusters
		for i in range(clust.n_clusters):
			ind = np.where( clustInd == i )
			clustFibs = np.squeeze(pcaFibs[ ind, :self.cutOff])

			# The matlab code provides some more functionality by filtering / removing
			# new random fibers with a too large distance from the cluster "mean
			# fiber" and cropping / resampling fibers back to the FOV. Some of the
			# new random fibers might end up outside of the FOV depending on the
			# cluster fiber distribution
			# I justed created multiple sets of random fibers until the required
			# nr of new streamlines is met.
			newPCAFibs = np.random.multivariate_normal( np.mean( clustFibs, 0 ), np.cov( clustFibs.T ), self.nrRand )

			randFibs = pcaResult.mu+np.concatenate((newPCAFibs,np.zeros((newPCAFibs.shape[0],pcaTrafo.shape[0]-self.cutOff),dtype=newPCAFibs.dtype)), axis=1).dot(pcaTrafo)

			# appending each bundle to the list might not be a fast way to do this!
			upFibs.append(randFibs)

		# save upsampled fibers
		upFibs = [s.reshape((int(pcaTrafo.shape[0]/3),3)) for s in np.concatenate(upFibs)]
		newFibs = [dipy.tracking.streamline.apply_affine(np.eye(4),s) for s in upFibs]
		newFibs = nib.streamlines.Tractogram(newFibs)
		newFibs.affine_to_rasmm = streams.affine

		header = streams.header
		header['nb_streamlines'] = len(upFibs)

		nib.streamlines.save(newFibs, 'data/fibsUp.trk', header=header)

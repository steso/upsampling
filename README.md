upsampling
----------

Fiber up‐sampling according to this [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5256175/)

dependencies:
* numpy
* dipy
* nibabel
* matplotlib
* sklearn

parameters:
* nrClusters: number of clusters (k) used during KMeans clustering in the cluster() function
* nrRand: number of new streamlines which are generated per cluster
* samplePoints: number of points per streamline during resampling
* cutOff: dimensionality after the PCA transformation

todo:
* replace matplotlib PCA with sklearn PCA (matplotlib PCA is depricated!)
* add ability to load fiber_assignment.txt instead of using k-means clustering
* crop or remove streamlines with points outside of the FOV
* remove streamlines with a too large distance to the bundle mean fiber (e.g. max distance of initial streamlines as threshold)
* further regularize location of streamline start and end points
* explore streamline distribution (bundlewise and total) in PCA-space
* try spline representation instead of resampled points

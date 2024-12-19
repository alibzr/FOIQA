#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: all-in-one file to get the GrandChallenge models ranking per track (saliency data).
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import numpy as np
import numba

from . import scanpathMetrics
from .MultiMatch_adswa import createdirectedgraph, dijkstra

@numba.njit
def getScanpath(fixationList, startPositions, scanpathIdx=0):
	"""Return a scanpath in a list of scanpaths
	"""
	if scanpathIdx >= len(startPositions)-1:
		range_ = np.arange(startPositions[scanpathIdx], fixationList.shape[0])
	else:
		range_ = np.arange(startPositions[scanpathIdx], startPositions[scanpathIdx+1])

	return fixationList[range_, :].copy()

@numba.njit
def sphere2UnitVector(sequence):
	"""Convert from longitude/latitude to 3D unit vectors
	"""
	UVec = np.zeros( (sequence.shape[0], 3) )
	UVec[:, 0] = np.sin(sequence[:,1]) * np.cos(sequence[:,0])
	UVec[:, 1] = np.sin(sequence[:,1]) * np.sin(sequence[:,0])
	UVec[:, 2] = np.cos(sequence[:,1])
	return UVec

normscanpaths = np.array([0, np.pi/2])
@numba.jit
def transformScanpath(fixations, starts, iSC):
	# Get individual experiment trials
	scanpath1 = getScanpath(fixations, starts, iSC)

	# Remove consecutive fixations sampled at the same location on the sphere
	#	Possible (though rare) in a model, impossible in real gaze data
	sim = ~np.all(scanpath1[1:, 1:3] == scanpath1[:-1, 1:3], axis=1)
	scanpath1[:-1] = scanpath1[:-1][sim]

	# 0,1,2: 3D unit vector; 3: starting timestamp; 4,5: mercator, 6: amplitude
	VEC = np.zeros( (scanpath1.shape[0], 10) )

	# Store starting timestamp
	VEC[:, 3] = scanpath1[:, 3]

	# Convert latitudes/longitudes to unit vectors
	VEC[:, :3] = sphere2UnitVector(scanpath1[:, 1:3])

	# Convert long/lat data to mercator
	VEC[:, 4:6] = scanpath1[:, 1:3] - normscanpaths
	VEC[:, 5] = np.log(np.tan(np.pi/4 + (VEC[:, 5])/2))

	# Saccade amplitudes: Angular distance between 3D positions on unitary sphere
	VEC[1:, 6] = np.arccos( np.einsum("ji,ji->j", VEC[1:, :3], VEC[:-1, :3]) )

	Sacc = VEC[1:, 4:6] - VEC[:-1, 4:6]
	# looping saccades - assumption: the smaller saccade vector is the right one
	# 	looping R
	loopR = np.where(Sacc[:-1,0] > np.pi/2)[0]
	Sacc[loopR,:] = VEC[loopR+1, 4:6] - ([2*np.pi, 0] + VEC[loopR, 4:6])
	# 	looping L
	loopL = np.where(Sacc[:-1,0] < -np.pi/2)[0]
	Sacc[loopL,:] = ([2*np.pi, 0] + VEC[loopL+1, 4:6]) - VEC[loopL, 4:6]

	prevSacc = Sacc[:-1, :]
	nextSacc = Sacc[1:, :]
	# Saccade relative direction
	VEC[1:-1, 7] = -np.arctan2( np.cross( prevSacc, nextSacc), np.einsum("ji,ji->j", prevSacc, nextSacc) )
	# Saccade vector on the mercator projection
	VEC[1:, 8:10] = Sacc

	return VEC

@numba.jit(parallel=True)
def computeMatrixPairs(starts_GT, starts_mod, fixations_GT, fixations_mod):
	# Named "GT" and "Mod" because it was originally written for comparing ground truth data to data generated by saccadic models.
	
	scores = np.zeros( (starts_GT.shape[0], starts_mod.shape[0], measureLen) )

	for iSC1 in numba.prange(0, starts_GT.shape[0]):
		
		VEC1 = transformScanpath(fixations_GT, starts_GT, iSC1)

		for iSC2 in numba.prange(0, starts_mod.shape[0]):

			VEC2 = transformScanpath(fixations_mod, starts_mod, iSC2)

			print(" "*8, "\r", " "*8, "{} - {} | {:>4}/{}".format(iSC1+1, iSC2+1, iSC1*len(starts_mod) + iSC2 +1, len(starts_GT)*len(starts_mod)), end="")

			scores[iSC1, iSC2, :] = compareScanpath(VEC1, VEC2)[1]

	return scores

@numba.jit
def alignScanpaths(WMat: np.ndarray, scanpath_dim: list):	
	numVert,rows,cols,weight = createdirectedgraph(scanpath_dim, WMat)
	# find the shortest path (= lowest sum of weights) through the graph using scipy dijkstra
	path, dist = dijkstra(numVert,rows,cols,weight,0, scanpath_dim[0] * scanpath_dim[1] - 1, scanpath_dim)

	return path.astype(int)

@numba.jit
def compareScanpath(VEC1, VEC2, weight=None):
	"""Return comparison scores between two scanpaths.
	"""
	weight = np.array(weight)

	# Get weight matrix and individual score values per cell in the weight matrix
	WMat, Vals = scanpathMetrics.computeWeightMatrix(VEC1, VEC2, weight=weight)
	
	path = alignScanpaths(WMat, [VEC1.shape[0], VEC2.shape[0]])

	values = Vals[path[:, 0], path[:, 1], :]
	scores = np.zeros_like(weight, dtype=float)

	# Do not take into account missing values (NaN)
	#	Values that cannot be computed should not lower the result
	for metric in range(scores.shape[0]):
		nanmask = np.isnan(values[:, metric])
		if np.all(nanmask):
			scores[metric] = np.nan
		else:
			values[:, metric][nanmask] = 0
			scores[metric] = np.average(values[:, metric], weights=~nanmask, axis=0)
			values[:, metric][nanmask] = np.nan

	nanmask = np.isnan(scores)
	scores[nanmask] = 0
	score = np.average(scores, weights=~nanmask)
	scores[nanmask] = np.nan

	return 1-score, 1-scores

measureNames = ["Position", "Duration", "Length", "Shape", "Direction"]
measureLen = len(measureNames)
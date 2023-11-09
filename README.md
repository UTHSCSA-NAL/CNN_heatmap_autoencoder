# CNN_heatmap_autoencoder

This is the code for paper:
Heatmaps Autoencoders Robustly Capture Alzheimer’s Disease’s Brain Alterations

Di Wang

Deep neural networks have achieved unprecedented success in diagnosing patients with Alzheimer's Disease from their MRI scans. Unfortunately, the decisions taken by these complex nonlinear architectures are difficult to interpret. Heatmap methods were introduced to visualize the brain regions where deep networks focus their attention when classifying patients, but very few quantitative comparisons have been conducted so far to determine what approaches would be the most relevant to study the atrophy induced by Alzheimer's Disease.  In this work, we propose to use autoencoders to fuse the maps generated by different heatmap methods to produce more reliable brain maps. We establish that combining the heatmaps produced by Layer-wise Relevance Propagation, Integrated Gradients, and the Guided Grad-CAM method for a CNN trained using 502 T1 MRI scans provided by the Alzheimer's Disease Neuroimaging Initiative produces brain maps better capturing the Alzheimer's Disease effects reported in a large independent meta-analysis combining 77 voxel-based morphometry studies. These results suggest that our nonlinear maps fusion is a promising approach to take advantage of the great variety of heatmap methods recently published.

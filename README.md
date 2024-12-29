# Modification (Outlier) detection with Gaussian Mixture Models

This is a tool to detect outliers of a multidimensional Gaussian Mixture Model. 

It is targeted towards Nanopore sequencing. Specifically, it contains a wrapper around remora for resquiggling in addition to a custom storage format to store resquiggled signals that makes it relatively efficient. Then, given a control and some potentially modified sample, for each position a GMM is fit based on the control observations. Based on how likely the potentially modified observation is within this model, we can give it a probability of being part of the same distribution (in other words, of it being unmodified, i.e. a p value). 

This tool is primarily built for de novo modification detection using sample-compare statistics. This is needed in cases where it is infeasible to generate ground modification truth datasets. In particular, we employ it to detect chemically introduced modifications that are present at high rates. 

This tool is somewhat similar to other GMM tools such as Nanocompore or xPore, with the difference that we don't attempt to create components for potentially modified observations. We don't do this because this would potentially bias the detection towards/against different types of modifications. In addition, we have observed that even fully unmodified samples can have multiple components, as such assigning either of these to be 'modified' would result in a larger number of false-positive modification calls. 

This tool does not have base-resolution, as modifications on a single base is likely to affect the signal of surrounding bases as well, likely resulting in these also being detected as modified. However, it is a good starting point when attempting to detect modifiations de novo. 
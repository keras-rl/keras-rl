import numpy as np


class GraphBatchInputProcessor(object):
	def __init__(self, feature_names):
		self.feature_names = feature_names
		self.n_feature_names = len(self.feature_names)

	def process(self, batch_data):
		# batch_data has the following structure: (batch_size, temporal_size, len(feature_names))
		
		# Initialize data structures.
		processed_batch_data = {}
		for name in self.feature_names:
			processed_batch_data[name] = []

		# Process each sample, which is distributed over time.
		for time_distributed_sample in batch_data:
			time_distributed_data = [[] for _ in xrange(self.n_feature_names)]

			# For each sample (across the time axis), collect the data.
			for sample_idx, sample in enumerate(time_distributed_sample):
				assert len(sample) == self.n_feature_names
				for feature_idx, feature in enumerate(sample):
					time_distributed_data[feature_idx].append(feature)
			
			# Now, collect data into final data structure.
			for idx, name in enumerate(self.feature_names):
				f = time_distributed_data[idx]
				processed_batch_data[name].append(f)

		# Finally, convert everything into a numpy array.
		for name in self.feature_names:
			processed_batch_data[name] = np.array(processed_batch_data[name], dtype='float32')
		return processed_batch_data


class GraphBatchInputProcessor2(object):
	def __init__(self, feature_names):
		self.feature_names = feature_names
		self.n_feature_names = len(self.feature_names)

	def process(self, batch_data):
		# batch_data has the following structure: (batch_size, len(feature_names))
		
		# Initialize data structures.
		processed_batch_data = {}
		for name in self.feature_names:
			processed_batch_data[name] = []

		# For each sample (across the time axis), collect the data.
		for sample in batch_data:
			assert len(sample) == self.n_feature_names
			for feature_idx, feature in enumerate(sample):
				processed_batch_data[self.feature_names[feature_idx]].append(feature)
			
		# Finally, convert everything into a numpy array.
		for name in self.feature_names:
			processed_batch_data[name] = np.array(processed_batch_data[name], dtype='float32')
		return processed_batch_data

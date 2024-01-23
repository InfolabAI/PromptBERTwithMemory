from typing import Union
import faiss
import numpy as np
import torch
import os
import pickle
from typing import List
from typing import Union


"""FAISS Nearest Neighbourhood Class"""
class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 8, device: Union[int, torch.device]=0, prenorm:bool=True) -> None:
        """
        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None
        if isinstance(device, torch.device):
            device = int(torch.cuda.current_device())
        self.device = device
        self.prenorm = prenorm

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), self.device, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            gpu_config = faiss.GpuIndexFlatConfig()
            gpu_config.device = self.device
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, gpu_config
            )
        return faiss.IndexFlatL2(dimension)

    
    # features: NxD Array
    def fit(self, features: np.ndarray) -> None:
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)

        # normalize
        if self.prenorm:
            faiss.normalize_L2(features)

        self.search_index.add(features.cpu())

    def _train(self, _index, _features):
        pass
    
    
    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        
        if self.prenorm:
            faiss.normalize_L2(query_features)

        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        
        # normalize
        if self.prenorm:
            faiss.normalize_L2(index_features)

        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None

class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: Union[list, np.ndarray]):
        if type(features) == list:
            features = [self._reduce(feature) for feature in features]
            return torch.cat(features, dim=1)
        else:
            return self._reduce(features)

class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        """Nearest-Neighbourhood Anomaly Scorer Class"""
        # (B,N,D) -> (B*N,D) or (N,D) -> (N,D)
        D = features.shape[-1]
        return features.reshape(-1, D)

class NearestNeighbourScorer(object):
    def __init__(self, n_nearest_neighbours: int, local_nn_method=FaissNN(False, 8, prenorm=False), temp=0.0) -> None:
        """
        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.local_nn_method = local_nn_method
        self.temp = temp
        self.patch_nn = lambda query: self.local_nn_method.run(
            n_nearest_neighbours, query
        )

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.local_nn_method.fit(self.detection_features)

    def predict(
        self, query_features: np.ndarray
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
             detection_query_features: np.array
                 corresponding to the test features generated by
                 some backbone network.
        """
        local_query_features = self.feature_merger.merge(
            query_features,
        )
        
        query_distances, query_nns = self.patch_nn(local_query_features)

        if self.temp:
            query_distances = np.exp((1/self.temp)*query_distances)

        local_anomaly_scores = np.mean(query_distances, axis=-1)

        return local_anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.local_nn_method.save(self._index_file(save_folder, f"{prepend}"))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.local_nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.local_nn_method.load(self._index_file(load_folder, f"{prepend}"))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )
import torch
import tqdm
import numpy as np


class Detection(torch.nn.Module):
    def __init__(self, model, sampler, scorer, layers_to_extract_from):
        super().__init__()
        self.backbone = model
        self.layers_to_extract_from = layers_to_extract_from
        self.featuresampler = sampler
        self.anomaly_scorer = scorer

    def _embed(self, images, detach=False, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return features.detach().cpu().numpy()
            return features

        self.backbone.eval()
        with torch.no_grad():
            features_list = []
            num_layers = 0
            for image in images:
                features_list.append(self.backbone(**image)['hidden_states'])
                num_layers = len(features_list[-1])

            gathered_features = []
            for i in range(num_layers):
                # gathered_features.append(torch.cat([features[i] for features in features_list], dim=1).mean( 1, keepdim=True))  # NOTE token 수가 다양하면 처리가 안되므로, 모두 average pooling 으로 처리
                gathered_features.append(torch.cat([features[i] for features in features_list], dim=1)[
                                         :, 0, :].unsqueeze(1))  # NOTE token 수가 다양하면 처리가 안되므로, 0번, 즉, [CLS] token 만 이용

        # NOTE BERT 의 encoder 도, uniformaly 도 layer 번호 순으로 gathered_features 가 나오므로 동일하게 사용 가능
        gathered_features = [gathered_features[int(
            layer)] for layer in self.layers_to_extract_from]

        patch_shapes = [(int(x.shape[1]**0.5), int(x.shape[1]**0.5))
                        for x in gathered_features]

        gathered_features = [x.unsqueeze(1) for x in gathered_features]

        gathered_features = torch.cat(gathered_features, dim=1)

        # [1, 8, 1, 768] --> [1, 1, 768] NOTE 8개의 layer 에 대해 average pooling
        gathered_features = gathered_features.mean(1)

        if provide_patch_shapes:
            return _detach(gathered_features), patch_shapes
        return _detach(gathered_features)

    def fit(self, input_data):
        """Computes and sets the support features for Uniformaly."""
        features = []
        self.backbone.eval()
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for text in data_iterator:
                _features = self._embed(text)
                features.append(_features)
                torch.cuda.empty_cache()

        features = torch.cat(features)
        features = features.reshape(-1, features.shape[-1])

        # subsampling
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, input_data):
        """This function provides anomaly scores/maps for full dataloaders."""
        scores = []

        self.backbone.eval()
        # Measure FPS
        with tqdm.tqdm(input_data, desc="Inferring...", leave=True) as data_iterator:
            for i, text in enumerate(data_iterator):
                features, patch_shapes = self._embed(
                    text, provide_patch_shapes=True)

                pred = self.anomaly_scorer.predict(features.cpu().numpy())
                # NOTE 이후는 수백 개의 patch 를 다루어 image score 를 계산하고, mask 를 생성하는 부분이므로, text 당 patch 가 1개인 지금은 필요없음
                _scores = [pred[0]]
                # convert numpy.array to tensor
                _scores = torch.from_numpy(np.asarray(_scores))

                for score in _scores:
                    scores.append(score)

        return torch.cat(scores)

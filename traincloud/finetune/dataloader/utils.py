import torch

def _get_video(self, video_id):
    if video_id not in self.features:
        print(video_id)
        video = torch.zeros(1, self.features_dim)
    else:
        video = self.features[video_id].float()
    if len(video) > self.max_feats:
        sampled = []
        for j in range(self.max_feats):
            sampled.append(video[(j * len(video)) // self.max_feats])
        video = torch.stack(sampled)
        video_len = self.max_feats
    elif len(video) < self.max_feats:
        video_len = len(video)
        video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
    else:
        video_len = self.max_feats

    return video, video_len
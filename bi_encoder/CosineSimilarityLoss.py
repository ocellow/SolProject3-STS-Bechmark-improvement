import torch
import torch.nn as nn 


class CosineSimilarityLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()
        self.cos_score_transformation = nn.Identity()


    def forward(self, sentence_features, labels): # return from batch collate fct
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.view(-1))

import torch


def load_classifier_weights(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    return checkpoint["fc.weight"]


def compute_similarity(weights):
    weights = torch.nn.functional.normalize(
        weights,
        dim=1
    )

    return weights @ weights.T


def get_top_references(similarity, target_class, top_k=3):
    scores = similarity[target_class].clone()

    scores[target_class] = -float("inf")

    top_scores, top_indices = torch.topk(
        scores,
        k=top_k
    )

    return top_indices, top_scores

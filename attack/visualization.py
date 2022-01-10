import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def vis(node_embeddings, node_ids, true_matches):
    # https://stellargraph.readthedocs.io/en/stable/demos/embeddings/graphsage-unsupervised-sampler-embeddings.html
    X = node_embeddings
    if len(X[1]) > 2:
        transform = TSNE  # PCA

        trans = transform(n_components=2)
        emb_transformed = pd.DataFrame(trans.fit_transform(X), index=node_ids)
        emb_transformed = add_color_attr(emb_transformed, true_matches)
    else:
        return
    alpha = 0.7

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        emb_transformed[0],
        emb_transformed[1],
        c=emb_transformed["label"].astype("category"),
        cmap="jet",
        alpha=alpha,
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    plt.title(
        "{} visualization of GraphSAGE embeddings for dataset".format(transform.__name__)
    )
    plt.show()


def add_color_attr(emb_transformed, true_matches):
    emb_transformed['label'] = len(emb_transformed) * ['k'] # default color: black
    colors = ['tab:blue','tab:orange','tab:green','tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i in range(0, len(colors)):
        emb_transformed.loc[true_matches[i][0], 'label'] = colors[i]
        emb_transformed.loc[true_matches[i][1], 'label'] = colors[i]
    return emb_transformed
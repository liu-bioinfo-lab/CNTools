import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from spatial_lda.featurization import featurize_samples
from spatial_lda.featurization import neighborhood_to_cluster
from spatial_lda.featurization import make_merged_difference_matrices
from spatial_lda.model import order_topics_consistently
import spatial_lda.model


def Spatial_LDA(ds, n_cns, eps, b, train_size_fraction, n_processes):
    data = ds.data
    dfs_image = {}
    for sample in data:
        for image in data[sample]:
            df_image = pd.DataFrame(data[sample][image].locs, columns=['X', 'Y'])
            df_image['cluster'] = data[sample][image].cts
            df_image['isb'] = True
            dfs_image[sample + '-' + str(image)] = df_image
    feats = featurize_samples(dfs_image, neighborhood_to_cluster, eps, 'isb', 'X', 'Y', n_processes=n_processes, include_anchors=True)

    all_sample_idxs = feats.index.map(lambda x: x[0])
    _sets = train_test_split(feats, test_size=1-train_size_fraction, stratify=all_sample_idxs)
    feats_image_train, _ = _sets
    train_difference_matrices = make_merged_difference_matrices(feats_image_train, dfs_image, 'X', 'Y')

    model = spatial_lda.model.train(sample_features=feats_image_train, 
                                    difference_matrices=train_difference_matrices,
                                    difference_penalty=b,
                                    n_topics=n_cns,
                                    n_parallel_processes=n_processes,                                                                         
                                    verbosity=1,
                                    admm_rho=0.1,
                                    primal_dual_mu=1e+5)
    order_topics_consistently([model])

    dfs_image_cns = pd.DataFrame(model.transform(feats), index=feats.index, columns=model.topic_weights.columns)
    cns = ds.get_data_tpl_copy()
    for sample_image in dfs_image:
        sample, image = sample_image.split('-')
        sample = int(sample) if sample.isnumeric() else sample
        image = int(image) if image.isnumeric() else image
        for i, _ in dfs_image[sample_image].iterrows():
            cns[sample][image].append(np.argmax(dfs_image_cns.loc(axis=0)[(sample_image, i),]))
    feats = feats.reindex(sorted(feats.columns), axis=1).to_numpy()
    return cns, feats

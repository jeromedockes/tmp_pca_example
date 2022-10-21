from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets

# TODO: replace sklearn import with our implementation of PCA
from sklearn.decomposition import PCA

IMG_SHAPE = (64, 64)
CMAP = "gray"

all_faces, _ = datasets.fetch_olivetti_faces(
    return_X_y=True, shuffle=True, random_state=0
)
assert all_faces.shape[1] == np.prod(IMG_SHAPE)

pca = PCA(n_components=100).fit(all_faces)

fig_components, axes_components = plt.subplots(4, 3)
fig_components.suptitle("First 12 components")
for component, ax in zip(pca.components_, axes_components.ravel()):
    ax.imshow(component.reshape(IMG_SHAPE), cmap=CMAP)

fig_samples, axes_samples = plt.subplots(3, 2)
axes_samples[0, 0].set_title("original face")
axes_samples[0, 1].set_title("reconstructed face")
for face, (ax_original, ax_reconstruction) in zip(all_faces, axes_samples):
    ax_original.imshow(face.reshape(IMG_SHAPE), cmap=CMAP)
    ax_reconstruction.imshow(
        pca.inverse_transform(pca.transform([face])).reshape(IMG_SHAPE),
        cmap=CMAP,
    )

plt.show()

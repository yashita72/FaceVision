import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from recognizer import FaceRecognizer

# Load recognizer and database
recognizer = FaceRecognizer(
    "models/face_recognition_sface_2021dec.onnx"
)

X = []
labels = []

# Collect embeddings
for person, embeddings in recognizer.database.items():

    for emb in embeddings:

        X.append(emb.flatten())
        labels.append(person)

# Reduce 128D -> 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8,6))

for i in range(len(X_pca)):
    plt.scatter(X_pca[i,0], X_pca[i,1], s=80)
    plt.text(
        X_pca[i,0]+0.02,
        X_pca[i,1]+0.02,
        labels[i],
        fontsize=10
    )

plt.title("SFace Embeddings (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.show()
# FAISS

FAISS is a library for efficient similarity search over dense vectors.
IndexFlatIP performs exact inner-product search, which is equivalent to
cosine similarity when the vectors are L2 normalised.

For large corpora, approximate indexes such as IVF or HNSW trade a small
amount of recall for a large speed-up.

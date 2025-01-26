from langchain_community.embeddings import LlamaCppEmbeddings
llama = LlamaCppEmbeddings(model_path="models/nomic-embed-text-v1.5.Q4_0.gguf")

text = "This is a test document."

doc_result = llama.embed_documents([text])
print(doc_result)
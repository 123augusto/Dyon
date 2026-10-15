import chromadb
from sentence_transformers import SentenceTransformer
from config import CAMINHO_CHROMA, MODELO_EMBEDDING, HARDWARE_ATUAL

def carregar_modelo_embedding():
    """
    Carrega o BAAI/bge-m3 usando:
    - 'sdpa' (Squared Dot Product Attention do PyTorch 2.2+), que funciona na T4 sem dar o erro do flash_attention_2.
    - bfloat16 automático na L4/A100 ou float16 na T4/V100.
    """
    dispositivo = HARDWARE_ATUAL["dispositivo"]
    kwargs_modelo = {}
    if dispositivo == "cuda":
        kwargs_modelo = {
            "torch_dtype": HARDWARE_ATUAL["dtype"],
            "attn_implementation": HARDWARE_ATUAL["attn_implementation"]
        }

    return SentenceTransformer(
        MODELO_EMBEDDING,
        device=dispositivo,
        model_kwargs=kwargs_modelo
    )

def inicializar_chromadb():
    """Conecta ao ChromaDB persistente no Google Drive."""
    cliente = chromadb.PersistentClient(path=CAMINHO_CHROMA)
    colecao_site = cliente.get_or_create_collection(
        name="conteudo_site_rag",
        metadata={"hnsw:space": "cosine"}
    )
    colecao_cache = cliente.get_or_create_collection(
        name="cache_respostas_aprovadas",
        metadata={"hnsw:space": "cosine"}
    )
    return cliente, colecao_site, colecao_cache

def indexar_chunks_no_chroma(chunks_site: list[dict], url_origem: str, modelo_emb, colecao_site):
    """Vetoriza e salva todos os blocos do site ajustando o batch_size conforme a VRAM."""
    textos = [c["conteudo"] for c in chunks_site]
    ids = [c["id"] for c in chunks_site]
    metadados = [{"titulo": c["titulo"], "url": url_origem} for c in chunks_site]

    # Usa batch menor na T4 (15 GB) e maior na L4 (22.5 GB) / A100 (40 GB)
    lote = 4 if HARDWARE_ATUAL.get("vram_util_gb", 15) <= 15.0 else 16

    vetores = modelo_emb.encode(
        textos,
        batch_size=lote,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    colecao_site.upsert(
        ids=ids,
        documents=textos,
        embeddings=vetores,
        metadatas=metadados
    )
    print(f"Sucesso! {colecao_site.count()} blocos indexados com batch_size={lote}.")

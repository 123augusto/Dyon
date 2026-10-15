import os
import uuid
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import CAMINHO_LOG_REPROVADAS, HARDWARE_ATUAL
from etapa2_indexador_bge import carregar_modelo_embedding, inicializar_chromadb
from etapa3_motor_qa_rag import customizar_qa_para_cliente

modelo_embedding = carregar_modelo_embedding()
_, colecao_site, colecao_cache = inicializar_chromadb()
pendentes_feedback = {}

app = FastAPI(title="API Projeto Dionísio - RAG & Feedback Loop")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PerfilClienteInput(BaseModel):
    id: str
    nome: str
    segmento: str
    nivel_tecnico: str
    tom_de_voz: str
    objetivo: str

class RequisicaoPergunta(BaseModel):
    pergunta: str
    resposta_base: Optional[str] = ""
    perfil_cliente: PerfilClienteInput

class RequisicaoFeedback(BaseModel):
    interacao_id: str
    aprovado: bool
    comentario_cliente: Optional[str] = ""

@app.get("/status_gpu")
def endpoint_status_gpu():
    """Retorna informações da GPU ativa e consumo de unidades de computação por hora."""
    return {
        "gpu": HARDWARE_ATUAL["nome_gpu"],
        "unidades_por_hora": HARDWARE_ATUAL["unidades_hora"],
        "horas_estimadas_200_creditos": HARDWARE_ATUAL["horas_estimadas_200_creditos"],
        "respostas_aprovadas_no_cache": colecao_cache.count()
    }

@app.post("/perguntar")
def endpoint_perguntar(req: RequisicaoPergunta):
    cliente_id = req.perfil_cliente.id
    pergunta = req.pergunta.strip()

    # 1. Consulta o Cache Semântico de respostas já aprovadas com 👍 (>= 90% de similaridade)
    if colecao_cache.count() > 0:
        vetor_busca = modelo_embedding.encode([pergunta], normalize_embeddings=True).tolist()
        res_cache = colecao_cache.query(
            query_embeddings=vetor_busca,
            n_results=1,
            where={"cliente_id": cliente_id}
        )
        if res_cache["ids"] and len(res_cache["ids"][0]) > 0:
            distancia = res_cache["distances"][0][0]
            similaridade = 1.0 - distancia
            if similaridade >= 0.90:
                dados_salvos = json.loads(res_cache["metadatas"][0][0]["resposta_json"])
                return {
                    "interacao_id": None,
                    "veio_do_cache": True,
                    "similaridade": round(similaridade * 100, 2),
                    "exibir_botoes_feedback": False,
                    "dados": dados_salvos
                }

    # 2. Se não está no cache, gera com Qwen3 + RAG
    resposta_gerada = customizar_qa_para_cliente(
        pergunta_base=pergunta,
        resposta_base=req.resposta_base,
        perfil_cliente=req.perfil_cliente.model_dump(),
        modelo_emb=modelo_embedding,
        colecao_site=colecao_site
    )

    interacao_id = f"int_{uuid.uuid4().hex[:10]}"
    pendentes_feedback[interacao_id] = {
        "pergunta": pergunta,
        "cliente_id": cliente_id,
        "resposta": resposta_gerada,
        "criado_em": datetime.now().isoformat()
    }

    return {
        "interacao_id": interacao_id,
        "veio_do_cache": False,
        "exibir_botoes_feedback": True,
        "dados": resposta_gerada
    }

@app.post("/feedback")
def endpoint_feedback(req: RequisicaoFeedback):
    item = pendentes_feedback.get(req.interacao_id)
    if not item:
        raise HTTPException(status_code=404, detail="Interação não encontrada ou já avaliada.")

    if req.aprovado:
        vetor = modelo_embedding.encode([item["pergunta"]], normalize_embeddings=True).tolist()
        id_cache = f"cache_{item['cliente_id']}_{req.interacao_id}"

        colecao_cache.upsert(
            ids=[id_cache],
            documents=[item["pergunta"]],
            embeddings=vetor,
            metadatas=[{
                "cliente_id": item["cliente_id"],
                "resposta_json": json.dumps(item["resposta"], ensure_ascii=False),
                "data_aprovacao": datetime.now().isoformat()
            }]
        )
        del pendentes_feedback[req.interacao_id]
        return {"status": "sucesso", "acao": "Resposta aprovada pelo cliente e salva no Cache Semântico!"}
    else:
        historico = []
        if os.path.exists(CAMINHO_LOG_REPROVADAS):
            with open(CAMINHO_LOG_REPROVADAS, "r", encoding="utf-8") as f:
                historico = json.load(f)

        historico.append({
            **item,
            "comentario_cliente": req.comentario_cliente,
            "data_reprovacao": datetime.now().isoformat()
        })
        with open(CAMINHO_LOG_REPROVADAS, "w", encoding="utf-8") as f:
            json.dump(historico, f, ensure_ascii=False, indent=2)

        del pendentes_feedback[req.interacao_id]
        return {"status": "reprovado", "acao": "Resposta bloqueada do cache e enviada para revisão."}

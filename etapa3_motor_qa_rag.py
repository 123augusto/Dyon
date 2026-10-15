import json
import ollama
from pydantic import BaseModel, Field
from config import MODELO_LLM, NUM_CTX_SEGURO

class ParQA(BaseModel):
    pergunta: str = Field(description="Pergunta clara que um cliente faria sobre a plataforma ou organização da festa")
    resposta_base: str = Field(description="Resposta fiel baseada exclusivamente no conteúdo do site")
    categoria: str = Field(description="Categoria (ex: Orçamento, Fornecedores, Convidados, Funcionamento da Plataforma)")

class ListaQA(BaseModel):
    itens: list[ParQA]

class RespostaCustomizada(BaseModel):
    pergunta_adaptada: str = Field(description="A pergunta reescrita para o tipo de festa/evento do cliente")
    resposta_personalizada: str = Field(description="Resposta adaptada ao perfil da festa do cliente mantendo fidelidade às regras do site")
    dicas_praticas_evento: list[str] = Field(description="2 a 3 passos práticos para ajudar o cliente a organizar sua festa na plataforma")

def extrair_qa_do_bloco(bloco_texto: str) -> list[dict]:
    """Extrai todas as perguntas e respostas possíveis ajustando num_ctx para a VRAM real da GPU."""
    prompt_sistema = (
        "Você é um especialista na plataforma de festas Dionísio. "
        "Leia o trecho do site e extraia TODAS as perguntas e respostas relevantes que um usuário "
        "teria sobre o funcionamento da plataforma e sobre como organizar sua festa.\n"
        "Regras:\n"
        "1. Use APENAS informações presentes no trecho.\n"
        "2. Escreva em português claro.\n"
        "3. Retorne estritamente em JSON. /no_think"
    )

    resposta = ollama.chat(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Extraia todas as perguntas e respostas deste trecho:\n\n{bloco_texto} /no_think"}
        ],
        format=ListaQA.model_json_schema(),
        options={"temperature": 0.2, "num_ctx": NUM_CTX_SEGURO}
    )
    dados = json.loads(resposta["message"]["content"])
    return dados.get("itens", [])

def buscar_contexto_rag(consulta: str, modelo_emb, colecao_site, top_k: int = 2) -> str:
    """Recupera os trechos mais relevantes do site no ChromaDB."""
    vetor_busca = modelo_emb.encode([consulta], normalize_embeddings=True).tolist()
    resultados = colecao_site.query(query_embeddings=vetor_busca, n_results=top_k)
    documentos = resultados["documents"][0] if resultados["documents"] else []
    return "\n\n---\n\n".join(documentos)

def customizar_qa_para_cliente(pergunta_base: str, resposta_base: str, perfil_cliente: dict, modelo_emb, colecao_site) -> dict:
    """Adapta a resposta via RAG para o perfil específico da festa/cliente."""
    contexto_site = buscar_contexto_rag(pergunta_base, modelo_emb, colecao_site, top_k=2)

    prompt_sistema = (
        "Você é o assistente inteligente da plataforma de festas Dionísio. "
        "Sua tarefa é responder à dúvida do cliente combinando a Resposta Base, o Contexto do Site (RAG) "
        "e o Perfil do Cliente/Evento.\n"
        "Regras:\n"
        "1. Adapte exemplos e linguagem para o tipo de festa, estilo e objetivo do cliente.\n"
        "2. NUNCA invente funcionalidades ou preços que não estejam no Contexto do site.\n"
        "3. Responda estritamente em JSON. /no_think"
    )

    prompt_usuario = f"""
PERFIL DO CLIENTE / FESTA:
- Nome: {perfil_cliente['nome']}
- Tipo de Evento/Segmento: {perfil_cliente['segmento']}
- Experiência com Eventos: {perfil_cliente['nivel_tecnico']}
- Tom de Voz Desejado: {perfil_cliente['tom_de_voz']}
- Objetivo Principal: {perfil_cliente['objetivo']}

PERGUNTA DO CLIENTE: {pergunta_base}
RESPOSTA OFICIAL BASE: {resposta_base}

CONTEXTO DO SITE (RAG):
{contexto_site}
/no_think
"""
    resposta = ollama.chat(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
        format=RespostaCustomizada.model_json_schema(),
        options={"temperature": 0.3, "num_ctx": NUM_CTX_SEGURO}
    )
    return json.loads(resposta["message"]["content"])

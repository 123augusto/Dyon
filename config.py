import os
import torch

PASTA_PROJETO = os.getenv("PASTA_PROJETO", "/content/drive/MyDrive/projeto_dionisio")
os.makedirs(PASTA_PROJETO, exist_ok=True)

CAMINHO_CHROMA = os.path.join(PASTA_PROJETO, "chroma_db")
CAMINHO_QA_BASE = os.path.join(PASTA_PROJETO, "catalogo_qa_base.json")
CAMINHO_FAQ_CLIENTES = os.path.join(PASTA_PROJETO, "faq_customizado_por_cliente.json")
CAMINHO_LOG_REPROVADAS = os.path.join(PASTA_PROJETO, "log_respostas_reprovadas.json")

MODELO_EMBEDDING = "BAAI/bge-m3"
URL_ALVO_PADRAO = "https://pt.wikipedia.org/wiki/Festa"

# Dados reais extraídos do benchmark de GPUs do Google Colab (Chris McCormick)
PERFIS_GPU_COLAB = {
    "T4":   {"unidades_hora": 1.84, "vram_util_gb": 15.0, "arquitetura": "Turing",       "suporta_bf16": False, "num_ctx_seguro": 1024},
    "V100": {"unidades_hora": 4.91, "vram_util_gb": 16.0, "arquitetura": "Volta",        "suporta_bf16": False, "num_ctx_seguro": 2048},
    "L4":   {"unidades_hora": 4.82, "vram_util_gb": 22.5, "arquitetura": "Ada Lovelace", "suporta_bf16": True,  "num_ctx_seguro": 4096},
    "A100": {"unidades_hora": 11.77,"vram_util_gb": 40.0, "arquitetura": "Ampere",       "suporta_bf16": True,  "num_ctx_seguro": 4096},
}

def detectar_configuracao_hardware(creditos_disponiveis: float = 200.0) -> dict:
    """
    Detecta a GPU ativa no Colab e ajusta automaticamente:
    - Tipo de ponto flutuante: bfloat16 (L4/A100) vs float16 (T4/V100).
    - Implementação de atenção: 'sdpa' (compatível com T4 e tão rápido quanto FlashAttention v2).
    - Janela de contexto (num_ctx) e modelo para respeitar os 15 GB úteis da T4 (descontado 1 GB de ECC).
    """
    if not torch.cuda.is_available():
        return {
            "dispositivo": "cpu",
            "nome_gpu": "CPU",
            "dtype": torch.float32,
            "attn_implementation": "sdpa",
            "modelo_llm": "qwen3:8b",
            "num_ctx": 1024,
            "unidades_hora": 0.10,
            "horas_estimadas_200_creditos": round(creditos_disponiveis / 0.10, 1)
        }

    nome_placa = torch.cuda.get_device_name(0)
    perfil = PERFIS_GPU_COLAB["T4"] # Padrão conservador (T4)
    chave_detectada = "T4"

    for chave, dados in PERFIS_GPU_COLAB.items():
        if chave in nome_placa.upper():
            perfil = dados
            chave_detectada = chave
            break

    # Seleciona bfloat16 apenas em GPUs mais novas (L4 e A100); mantém float16 na T4 e V100
    dtype_ideal = torch.bfloat16 if perfil["suporta_bf16"] else torch.float16

    # Na T4 (apenas 15 GB úteis devido a 1 GB do ECC), podemos usar qwen3:8b no desenvolvimento
    # ou qwen3:14b com sequência controlada (1024) para não estourar a VRAM
    modelo_padrao = os.getenv("MODELO_LLM", "qwen3:14b")
    horas_restantes = round(creditos_disponiveis / perfil["unidades_hora"], 1)

    info = {
        "dispositivo": "cuda",
        "nome_gpu": f"{nome_placa} ({chave_detectada})",
        "vram_util_gb": perfil["vram_util_gb"],
        "dtype": dtype_ideal,
        "attn_implementation": "sdpa", # Evita erro do flash_attention_2 na T4
        "modelo_llm": modelo_padrao,
        "num_ctx": perfil["num_ctx_seguro"],
        "unidades_hora": perfil["unidades_hora"],
        "horas_estimadas_200_creditos": horas_restantes
    }

    print(f"=== DIAGNÓSTICO DE GPU DO COLAB ===")
    print(f"GPU Detectada: {info['nome_gpu']} | VRAM Útil Real: {info['vram_util_gb']} GB")
    print(f"Precisão Selecionada: {info['dtype']} | Atenção: {info['attn_implementation']} (PyTorch 2.2+)")
    print(f"Custo Estimado: {info['unidades_hora']} unidades/h -> Seus {creditos_disponiveis} créditos rendem ~{horas_restantes} horas.")
    print(f"Janela de Contexto Segura (num_ctx): {info['num_ctx']} tokens")
    print(f"===================================")
    return info

HARDWARE_ATUAL = detectar_configuracao_hardware()
MODELO_LLM = HARDWARE_ATUAL["modelo_llm"]
NUM_CTX_SEGURO = HARDWARE_ATUAL["num_ctx"]

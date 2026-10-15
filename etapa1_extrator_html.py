import asyncio
import re
import nest_asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

nest_asyncio.apply()

async def extrair_markdown_do_site(url: str) -> str:
    """Acessa o HTML do site da plataforma de festas, limpa ruídos e converte em Markdown."""
    browser_cfg = BrowserConfig(headless=True)
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["nav", "footer", "header", "script", "style", "aside", "noscript"],
        remove_overlay_elements=True,
    )
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        resultado = await crawler.arun(url=url, config=run_cfg)
        if not resultado.success:
            raise RuntimeError(f"Erro ao acessar {url}: {resultado.error_message}")
        return resultado.markdown

def dividir_markdown_em_chunks(markdown_texto: str, tamanho_max: int = 1200) -> list[dict]:
    """
    Divide o Markdown em blocos de até 1200 caracteres (~300-400 tokens),
    garantindo que fiquem dentro do limite ótimo de 1024 tokens de sequência na GPU T4.
    """
    secoes = re.split(r'(?m)^(?=#{1,3}\s)', markdown_texto)
    chunks = []

    for idx, secao in enumerate(secoes):
        secao = secao.strip()
        if len(secao) < 60:
            continue

        linhas = secao.split('\n')
        titulo = linhas[0].lstrip('#').strip() if linhas[0].startswith('#') else f"Seção {idx+1}"

        if len(secao) <= tamanho_max:
            chunks.append({"id": f"chunk_{len(chunks)}", "titulo": titulo, "conteudo": secao})
        else:
            paragrafos = secao.split('\n\n')
            bloco_atual = f"# {titulo}\n\n"
            for p in paragrafos:
                if len(bloco_atual) + len(p) <= tamanho_max:
                    bloco_atual += p + "\n\n"
                else:
                    chunks.append({"id": f"chunk_{len(chunks)}", "titulo": titulo, "conteudo": bloco_atual.strip()})
                    bloco_atual = f"# {titulo} (continuação)\n\n{p}\n\n"
            if len(bloco_atual.strip()) > 60:
                chunks.append({"id": f"chunk_{len(chunks)}", "titulo": titulo, "conteudo": bloco_atual.strip()})

    return chunks

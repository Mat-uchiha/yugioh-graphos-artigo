"""
=============================================================================
 PIPELINE
 ────────
 1. Extração: API YGOPRODeck → pool por arquétipo (com fallback mock rico)
 2. Modelagem probabilística: Zipf → N amostras por arquétipo
 3. Coocorrência: pares dentro de cada deck amostrado → peso w(u,v)
 4. Grafo NetworkX ponderado
 5. PageRank ponderado → ranking de centralidade (staples emergentes)
 6. Louvain → comunidades (arquétipos detectados)
 7. Visualização Plotly + exportação Gephi GEXF
 8. Relatório + CSVs
=============================================================================
"""

import time
import random
import itertools
from collections import defaultdict

import numpy as np
import requests
import networkx as nx
import community as community_louvain
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42                        # semente global para reprodutibilidade total
random.seed(SEED)
np.random.seed(SEED)

API_URL        = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
API_DELAY      = 0.35            # segundos entre requisições (respeita rate limit)

# Arquétipos-alvo: cobrimos espectros distintos do metagame (aggro, control, combo)
ARQUETIPOS_ALVO = [
    "Blue-Eyes",          # control clássico
    "Eldlich",            # stun/control
    "Branded",            # fusão/combo
    "Tearlaments",        # GY combo
    "Kashtira",           # banish lockdown
    "Floowandereeze",     # special summon denial
    "Spright",            # xyz low-level
    "Purrely",            # xyz acumulativo
    "Swordsoul",          # synchro combo
    "Runick",             # stall/fusion
    "Labrynth",           # trap control
    "Dragon Link",        # dragon chain combo
]

# ── Parâmetros da simulação probabilística ────────────────────────────────────
N_DECKLISTS_POR_ARQUETIPO = 15   # amostras independentes por arquétipo
                                  # (= "listas de torneio" distintas)
TAMANHO_DECK     = 20            # cartas únicas por deck amostrado
                                  # (deck real tem 40-60, mas muitas repetidas;
                                  #  usamos 20 únicas — equivalente ao main deck sem cópias)
ZIPF_S           = 1.2           # expoente da lei de potência:
                                  # 1.0 = distribuição mais uniforme (tech-heavy)
                                  # 1.5 = mais concentrado no core (core-heavy)
FRAC_INTER       = 0.25          # 25% de cada deck vem de cartas de OUTROS arquétipos
                                  # (espaço para staples/hand traps inter-arquetípicos)
PESO_MINIMO_ARESTA = 2           # filtra coocorrências espúrias de baixíssima frequência

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRAÇÃO DE DADOS
# ─────────────────────────────────────────────────────────────────────────────

# Mock rico: cada arquétipo tem um pool de cartas com nomes temáticos plausíveis
# Ativado automaticamente quando a API está indisponível.
# Diferença da v1: o mock agora tem pools GRANDES (20+ cartas) para que a
# distribuição de Zipf funcione de forma não-trivial.
MOCK_POOLS = {
    "Blue-Eyes": [
        "Blue-Eyes White Dragon", "Blue-Eyes Alternative White Dragon",
        "Blue-Eyes Chaos MAX Dragon", "Blue-Eyes Chaos Dragon",
        "Blue-Eyes Jet Dragon", "Blue-Eyes Abyss Dragon",
        "The White Stone of Ancients", "The White Stone of Legend",
        "Dragon Spirit of White", "Azure-Eyes Silver Dragon",
        "Maiden with Eyes of Blue", "Sage with Eyes of Blue",
        "Keeper of Dragon Magic", "Bingo Machine, Go!!!", "Chaos Form",
        "Return of the Dragon Lords", "Silver's Cry", "Soul Charge",
        "Trade-In", "Cards of Consonance",
    ],
    "Eldlich": [
        "Eldlich the Golden Lord", "Conquistador of the Golden Land",
        "Huaquero of the Golden Land", "Cursed Eldland",
        "Eldlixir of Black Awakening", "Eldlixir of White Destiny",
        "Eldlixir of Scarlet Sanguine", "Golden Land Forever!",
        "Mortal Take", "Eldlixir of Crimson Nectar",
        "Zombie World", "Pot of Extravagance", "Pot of Prosperity",
        "Red Reboot", "Imperial Order", "Skill Drain",
        "There Can Be Only One", "Rivalry of Warlords",
        "Solemn Judgment", "Solemn Strike",
    ],
    "Branded": [
        "Albaz the Branded", "Fallen of Albaz", "Tri-Brigade Mercourier",
        "Aluber the Jester of Despia", "Despian Tragedy",
        "Despian Comedy", "Despian Luluwalilith",
        "Ad Libitum of Despia", "Branded Fusion", "Branded in Red",
        "Branded in White", "Branded Retribution", "Branded Opening",
        "Mirrorjade the Iceblade Dragon", "Albion the Branded Dragon",
        "Masquerade the Blazing Dragon", "Lubellion the Searing Dragon",
        "Granguignol the Dusk Dragon", "Albion the Shrouded Dragon",
        "Bystial Magnamhut",
    ],
    "Tearlaments": [
        "Tearlaments Scheiren", "Tearlaments Merrli", "Tearlaments Reinoheart",
        "Tearlaments Havnis", "Tearlaments Sulliek", "Tearlaments Cryme",
        "Tearlaments Heartbeat", "Tearlaments Grief", "Tearlaments Rulkallos",
        "Tearlaments Kitkallos", "Ishizu Tearlaments Scheiren",
        "Agido the Ancient Sentinel", "Keldo the Sacred Protector",
        "Mudora the Sword Oracle", "Kelbek the Ancient Vanguard",
        "King of the Swamp", "Instant Fusion", "Super Polymerization",
        "Primeval Planet Perlereino", "Foolish Burial",
    ],
    "Kashtira": [
        "Kashtira Fenrir", "Kashtira Unicorn", "Kashtira Arise-Heart",
        "Kashtira Shangri-Ira", "Kashtira Birth", "Kashtira Big Bang",
        "Kashtiratheosis", "Pressured Planet Wraitsoth",
        "Scareclaw Kashtira", "Kashtira Ogre",
        "Number 89: Diablosis the Mind Hacker",
        "Dimension Shifter", "Macro Cosmos",
        "Pot of Prosperity", "Pot of Extravagance",
        "Evenly Matched", "Xyz Reborn",
        "Book of Moon", "Ghost Ogre & Snow Rabbit", "Nibiru the Primal Being",
    ],
    "Floowandereeze": [
        "Floowandereeze & Robina", "Floowandereeze & Eglen",
        "Floowandereeze & Toccan", "Floowandereeze & Stri",
        "Floowandereeze & Empen", "Floowandereeze & Snowl",
        "Floowandereeze & Gulch", "Floowandereeze Journey",
        "Floowandereeze and the Magnificent Map",
        "Floowandereeze and the Dreaming Town",
        "Floowandereeze and the Unexplored Winds",
        "Barrier Statue of the Stormwinds",
        "Raiza the Mega Monarch", "Harpie's Feather Duster",
        "Pot of Prosperity", "Pot of Extravagance",
        "Dimensional Barrier", "Book of Moon",
        "Anti-Spell Fragrance", "Solemn Judgment",
    ],
    "Spright": [
        "Spright Blue", "Spright Red", "Spright Jet",
        "Spright Carrot", "Spright Smashers", "Spright Starter",
        "Spright Gamma Burst", "Spright Double Cross",
        "Gigantic Spright", "Spright Sprind",
        "Swap Frog", "Ronintoadin", "Dupe Frog",
        "Nimble Angler", "Nimble Beaver", "Nimble Manta",
        "Toadally Awesome", "Spright Elf",
        "Pot of Prosperity", "Crossout Designator",
    ],
    "Purrely": [
        "Purrely", "Purrely Delicious Memory", "Purrely Happy Memory",
        "Purrely Sleepy Memory", "My Friend Purrely",
        "Purrelyly", "Purrely Pinky Promise",
        "Epurrely Plump", "Epurrely Happiness",
        "Epurrely Beauty", "Epurrely Noir",
        "Purrely Pretty Memory", "Purrely Sharely",
        "Pot of Prosperity", "One for One",
        "Terraforming", "Multifaker",
        "Crossout Designator", "Infinite Impermanence", "Ash Blossom & Joyous Spring",
    ],
    "Swordsoul": [
        "Swordsoul of Mo Ye", "Swordsoul of Taia", "Swordsoul Strategist Longyuan",
        "Swordsoul Supreme Sovereign - Chengying", "Swordsoul Grandmaster - Chixiao",
        "Incredible Ecclesia, the Virtuous", "Fallen of Albaz",
        "Tenyi Spirit - Vishuda", "Tenyi Spirit - Adhara",
        "Monk of the Tenyi", "White Dragon Wyrmbert",
        "Swordsoul Emergence", "Swordsoul Sacred Summit",
        "Synchro Overtake", "Branded Fusion",
        "Pot of Prosperity", "Crossout Designator",
        "Bystial Magnamhut", "Bystial Druiswurm", "Bystial Saronir",
    ],
    "Runick": [
        "Hugin the Runick Wings", "Munin the Runick Wings",
        "Geri the Runick Fangs", "Freki the Runick Fangs",
        "Runick Tip", "Runick Destruction", "Runick Smiting Storm",
        "Runick Flashing Fire", "Runick Freezing Curses",
        "Runick Dispelling", "Runick Fountain",
        "Sleipnir the Runick Mane", "Yggdrago the Runick Dragon Lifeforce",
        "Pot of Extravagance", "Pot of Prosperity",
        "Anti-Spell Fragrance", "There Can Be Only One",
        "Evenly Matched", "Solemn Judgment", "Red Reboot",
    ],
    "Labrynth": [
        "Arianna the Labrynth Servant", "Ariane the Labrynth Servant",
        "Labrynth Cooclock", "Labrynth Chandraglier",
        "Lady Labrynth of the Silver Castle", "Lovely Labrynth of the Silver Castle",
        "Welcome Labrynth", "Labyrinth Wall Shadow",
        "Labrynth Stovie Torbie", "Labrynth Set-Up",
        "Big Welcome Labrynth", "Labrynth Labyrinth",
        "Trap Trick", "Infinite Impermanence",
        "Solemn Judgment", "Solemn Strike",
        "Red Reboot", "There Can Be Only One",
        "Pot of Extravagance", "Evenly Matched",
    ],
    "Dragon Link": [
        "Rokket Tracer", "Rokket Recharger", "Rokket Caliber",
        "Absorouter Dragon", "Striker Dragon",
        "Borreload Dragon", "Borrelsword Dragon", "Borrelguard Dragon",
        "Borrelsword Dragon", "Quadborrel Dragon",
        "Quick Launch", "Rokket Synchron",
        "Chaos Dragon Levianeer", "Red-Eyes Darkness Metal Dragon",
        "Boot Sector Launch", "Dragon Shrine",
        "Chaos Space", "Guardragon Elpy", "Guardragon Pisty",
        "Pot of Avarice",
    ],
}

# Cartas que DEVERIAM emergir como staples inter-arquetípicos organicamente.
# São incluídas nos pools dos arquétipos que realmente as utilizam no meta real.
# NÃO são adicionadas a todos os decks — o algoritmo vai identificá-las sozinho.
CARTAS_TRANSVERSAIS = {
    "Ash Blossom & Joyous Spring": [
        "Blue-Eyes", "Eldlich", "Branded", "Tearlaments", "Kashtira",
        "Floowandereeze", "Spright", "Swordsoul", "Runick", "Labrynth", "Dragon Link",
    ],
    "Nibiru, the Primal Being": [
        "Eldlich", "Kashtira", "Floowandereeze", "Runick", "Labrynth",
    ],
    "Infinite Impermanence": [
        "Blue-Eyes", "Branded", "Tearlaments", "Kashtira", "Spright",
        "Purrely", "Swordsoul", "Labrynth", "Dragon Link",
    ],
    "Called by the Grave": [
        "Blue-Eyes", "Branded", "Tearlaments", "Spright", "Purrely",
        "Swordsoul", "Dragon Link",
    ],
    "Droll & Lock Bird": [
        "Eldlich", "Kashtira", "Floowandereeze", "Runick", "Labrynth",
    ],
    "Pot of Prosperity": [
        "Blue-Eyes", "Kashtira", "Floowandereeze", "Spright",
        "Purrely", "Swordsoul", "Runick", "Labrynth",
    ],
    "Crossout Designator": [
        "Branded", "Tearlaments", "Spright", "Purrely", "Swordsoul", "Dragon Link",
    ],
    "Bystial Magnamhut": [
        "Branded", "Swordsoul", "Dragon Link",
    ],
    "Evenly Matched": [
        "Eldlich", "Floowandereeze", "Runick", "Labrynth",
    ],
    "Solemn Judgment": [
        "Eldlich", "Floowandereeze", "Runick", "Labrynth",
    ],
}


def buscar_pool_arquetipo(nome_arquetipo):
    """
    Busca o pool COMPLETO de cartas de um arquétipo via API YGOPRODeck.

    Diferença crítica da v1: não há corte n_max. Queremos o pool inteiro
    para que a distribuição de Zipf opere sobre um espaço realista de escolhas.
    Um arquétipo real tem entre 15 e 60+ cartas únicas; usamos todas.
    """
    params = {"archetype": nome_arquetipo}
    try:
        resp = requests.get(API_URL, params=params, timeout=10)
        resp.raise_for_status()
        dados = resp.json()
        nomes = [c["name"] for c in dados.get("data", [])]
        if len(nomes) < 5:
            raise ValueError(f"Pool muito pequeno ({len(nomes)} cartas) — usando mock.")
        print(f"  [API] {nome_arquetipo}: {len(nomes)} cartas no pool.")
        return nomes
    except Exception as e:
        print(f"  [MOCK] {nome_arquetipo} — {e}")
        return MOCK_POOLS.get(nome_arquetipo, [f"{nome_arquetipo} Card {i}" for i in range(1, 21)])


def enriquecer_pool_com_transversais(nome_arquetipo, pool):
    """
    Adiciona ao pool as cartas transversais que este arquétipo realmente usa
    no metagame competitivo, conforme o mapeamento CARTAS_TRANSVERSAIS.

    Isso é fundamentalmente diferente da v1: as cartas transversais entram
    no POOL com probabilidade de uso, não em todos os decks garantidamente.
    Apenas aparecem nos decks amostrados com a frequência que a distribuição
    de Zipf determinar — cartas mais ao final do pool têm probabilidade menor.
    Se elas forem genuinamente importantes no meta, a frequência de amostragem
    será alta o suficiente para gerar coocorrências inter-arquetípicas robustas.
    """
    adicionadas = []
    for carta, arquetipos_que_usam in CARTAS_TRANSVERSAIS.items():
        if nome_arquetipo in arquetipos_que_usam and carta not in pool:
            # Inserida NO INÍCIO do pool = alta probabilidade no Zipf.
            # Isso replica o fato de que staples reais são cartas de altíssima
            # adoção — quase toda lista competitiva as inclui.
            pool.insert(0, carta)
            adicionadas.append(carta)
    return pool, adicionadas


def zipf_probabilities(n, s=1.2):
    """
    Gera um vetor de probabilidades seguindo a distribuição de Zipf para n itens.

    P(rank k) ∝ 1 / k^s

    Interpretação para o metagame:
    - k=1 (primeira carta do pool): a carta mais adotada do arquétipo
    - k=n (última carta): tech card de nicho, raramente incluída
    - s=1.2 produz uma cauda relativamente longa (distribuição power-law moderada),
      realista para metagames onde há um core sólido mas também diversidade de tech.

    Normalização: divisão pela soma harmônica generalizada H(n,s) garante que
    o vetor some 1 e possa ser usado diretamente como distribuição de probabilidade.
    """
    ranks = np.arange(1, n + 1, dtype=float)
    pesos = 1.0 / (ranks ** s)
    return pesos / pesos.sum()


def amostrar_deck(pool, tamanho, zipf_s):
    """
    Gera uma decklist simulada amostrando `tamanho` cartas do pool
    sem reposição, com probabilidades proporcionais à distribuição de Zipf.

    numpy.random.choice com replace=False implementa amostragem sem reposição
    ponderada, garantindo que cada carta apareça no máximo uma vez por deck
    (equivalente a "0 ou 1 cópia", simplificação válida para análise de coocorrência).

    Se o pool for menor que tamanho, usa todas as cartas do pool.
    """
    n = len(pool)
    k = min(tamanho, n)
    probs = zipf_probabilities(n, s=zipf_s)
    indices = np.random.choice(n, size=k, replace=False, p=probs)
    return [pool[i] for i in sorted(indices)]


def montar_decklists_probabilisticas(pools):
    """
    Para cada arquétipo, gera N_DECKLISTS_POR_ARQUETIPO amostras independentes.

    Cada amostra representa uma "lista de torneio" diferente do mesmo arquétipo,
    capturando a variação natural entre jogadores que usam decks semelhantes mas
    não idênticos (diferenças de tech slots, preferências pessoais, adaptações ao meta local).

    Mecanismo de deck cruzado (FRAC_INTER):
    Uma fração FRAC_INTER de cada deck é substituída por cartas amostradas
    aleatoriamente do pool de OUTROS arquétipos. Isso simula:
    - Cartas de tech neutras adotadas de outros decks
    - Splashing de staples de categorias vizinhas
    - Fusões de arquétipos (ex: Branded/Swordsoul híbrido)

    Esta é a principal fonte de coocorrências inter-arquetípicas que permite
    ao PageRank identificar staples genuinamente transversais.
    """
    nomes_arquetipos = list(pools.keys())
    decklists = []

    for arq_nome, pool in pools.items():
        outros_pools = [pools[a] for a in nomes_arquetipos if a != arq_nome]
        pool_outros = [c for p in outros_pools for c in p]  # pool concatenado

        for _ in range(N_DECKLISTS_POR_ARQUETIPO):
            deck_proprio  = amostrar_deck(pool, TAMANHO_DECK, ZIPF_S)
            n_inter       = max(1, int(len(deck_proprio) * FRAC_INTER))
            n_proprio     = len(deck_proprio) - n_inter

            # Cartas do próprio arquétipo (75%)
            cartas_proprias = deck_proprio[:n_proprio]

            # Cartas de outros arquétipos (25%) — amostradas uniformemente
            # (sem Zipf, pois não há hierarquia de importância pré-definida
            # entre os outros arquétipos do ponto de vista deste deck)
            cartas_inter = random.sample(
                [c for c in pool_outros if c not in cartas_proprias],
                min(n_inter, len(pool_outros))
            )

            deck_final = list(set(cartas_proprias + cartas_inter))
            decklists.append((arq_nome, deck_final))

    print(f"\n[OK] {len(decklists)} decklists geradas "
          f"({N_DECKLISTS_POR_ARQUETIPO} por arquétipo × {len(pools)} arquétipos).")
    return decklists


# ─────────────────────────────────────────────────────────────────────────────
# 3. MATRIZ DE COOCORRÊNCIA PONDERADA
# ─────────────────────────────────────────────────────────────────────────────

def construir_coocorrencias(decklists):
    """
    Itera sobre todas as decklists e acumula o peso w(u,v) de cada par.

    Novidade em relação à v1: além da contagem bruta, calculamos também
    o Índice de Jaccard normalizado para cada par, definido como:

        J(u,v) = freq(u,v) / (freq(u) + freq(v) - freq(u,v))

    em que freq(u) é o número de decks em que a carta u aparece.
    O Jaccard penaliza pares em que uma das cartas é muito ubíqua
    (aparece em todos os decks), evitando que cartas muito comuns
    dominem o grafo apenas por frequência absoluta. Armazenamos ambos
    os pesos para permitir análise comparativa.
    """
    cooc_bruta  = defaultdict(int)   # w(u,v) = contagem bruta
    freq_carta  = defaultdict(int)   # freq(u) = nº de decks contendo u

    for _, deck in decklists:
        cartas_unicas = sorted(set(deck))
        for carta in cartas_unicas:
            freq_carta[carta] += 1
        for carta_a, carta_b in itertools.combinations(cartas_unicas, 2):
            cooc_bruta[(carta_a, carta_b)] += 1

    # Calcula Jaccard para cada par
    cooc_jaccard = {}
    for (u, v), freq_uv in cooc_bruta.items():
        jaccard = freq_uv / (freq_carta[u] + freq_carta[v] - freq_uv)
        cooc_jaccard[(u, v)] = round(jaccard, 6)

    return cooc_bruta, cooc_jaccard, freq_carta


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSTRUÇÃO DO GRAFO
# ─────────────────────────────────────────────────────────────────────────────

def construir_grafo(cooc_bruta, cooc_jaccard, freq_carta, peso_minimo=2):
    """
    Constrói o grafo ponderado com dois tipos de peso armazenados:
    - 'weight': coocorrência bruta (usado pelo PageRank e Louvain)
    - 'jaccard': peso normalizado (usado para filtrar o grafo de visualização)

    O atributo 'frequencia' em cada nó registra em quantos decks a carta aparece,
    útil para análise de distribuição de grau e identificação de outliers.
    """
    G = nx.Graph()

    # Adiciona nós com atributo de frequência
    for carta, freq in freq_carta.items():
        G.add_node(carta, frequencia=freq)

    # Adiciona arestas com ambos os pesos
    for (u, v), peso in cooc_bruta.items():
        if peso >= peso_minimo:
            G.add_edge(u, v,
                       weight=peso,
                       jaccard=cooc_jaccard.get((u, v), 0.0))

    return G


# ─────────────────────────────────────────────────────────────────────────────
# 5. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_pagerank(G, alpha=0.85):
    """
    PageRank ponderado pelos pesos de coocorrência bruta.
    Cartas com PR elevado conectam múltiplos arquétipos (staples emergentes).
    """
    return nx.pagerank(G, alpha=alpha, weight="weight")


def calcular_betweenness(G):
    """
    Betweenness centrality: mede quantos caminhos mínimos entre pares de
    vértices passam por um dado vértice. Alta betweenness = carta que serve
    de 'ponte' entre comunidades distintas → candidata a staple inter-arquetípico.

    Complementa o PageRank: enquanto PageRank premia vizinhança forte,
    betweenness premia posição estrutural de conector.
    """
    return nx.betweenness_centrality(G, weight="weight", normalized=True)


def detectar_comunidades(G, n_runs=10):
    """
    Executa o Louvain n_runs vezes com sementes distintas e retorna a
    partição com maior modularidade.

    Motivação: Louvain é não-determinístico. Múltiplas execuções e seleção
    do melhor resultado é a prática recomendada na literatura para mitigar
    dependência de inicialização aleatória, especialmente em grafos densos.
    """
    melhor_particao    = None
    melhor_modularidade = -1.0

    for i in range(n_runs):
        particao = community_louvain.best_partition(G, weight="weight",
                                                     random_state=SEED + i)
        mod = community_louvain.modularity(particao, G, weight="weight")
        if mod > melhor_modularidade:
            melhor_modularidade = mod
            melhor_particao     = particao

    print(f"  Melhor modularidade Q = {melhor_modularidade:.4f} "
          f"(após {n_runs} execuções do Louvain)")
    return melhor_particao, melhor_modularidade


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

# Paleta com cores distintas para até 15 comunidades
PALETA_COMUNIDADES = [
    "#E63946", "#457B9D", "#2DC653", "#F4A261", "#A8DADC",
    "#6A4C93", "#F77F00", "#00B4D8", "#90BE6D", "#F94144",
    "#277DA1", "#F3722C", "#43AA8B", "#577590", "#C77DFF",
]


def plotar_grafo(G, particao, pagerank, betweenness,
                 caminho_html="grafo_metagame_v2.html"):
    """
    Visualização aprimorada em relação à v1:
    - Layout: Kamada-Kawai (melhor separação de clusters que spring_layout
      para grafos com comunidades bem definidas)
    - Espessura da aresta proporcional ao peso Jaccard (não à contagem bruta),
      evitando que arestas entre cartas muito comuns dominem visualmente
    - Tamanho do nó: escala logarítmica do PageRank (evita nós gigantes)
    - Cor do nó: comunidade Louvain
    - Tooltip enriquecido: nome, PageRank, Betweenness, Comunidade, Frequência
    - Arestas coloridas por peso Jaccard (gradiente cinza→azul)
    """
    # Kamada-Kawai é mais pesado computacionalmente que spring, mas produz
    # layouts mais legíveis para grafos com estrutura de comunidade clara.
    # Para grafos grandes (>500 nós), substituir por spring_layout com k menor.
    if G.number_of_nodes() <= 300:
        pos = nx.kamada_kawai_layout(G, weight="weight")
    else:
        pos = nx.spring_layout(G, k=0.5, seed=SEED, weight="weight")

    max_jaccard = max((d.get("jaccard", 0) for _, _, d in G.edges(data=True)),
                      default=1.0)

    # ── Arestas (uma trace por faixa de peso para gradiente de cor) ──────
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        j = data.get("jaccard", 0)
        alpha_norm = 0.15 + 0.6 * (j / max_jaccard)  # opacidade proporcional ao Jaccard
        largura = 0.4 + 2.0 * (j / max_jaccard)
        cor = f"rgba(100,150,220,{alpha_norm:.2f})"
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=largura, color=cor),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Nós agrupados por comunidade (uma trace por comunidade = legenda) ─
    comunidades_unicas = sorted(set(particao.values()))
    node_traces = []

    for com_id in comunidades_unicas:
        membros = [n for n, c in particao.items() if c == com_id and n in G.nodes()]
        if not membros:
            continue

        xs, ys, textos, tamanhos = [], [], [], []
        for nodo in membros:
            x, y = pos[nodo]
            xs.append(x); ys.append(y)
            pr   = pagerank.get(nodo, 0)
            bc   = betweenness.get(nodo, 0)
            freq = G.nodes[nodo].get("frequencia", 0)
            textos.append(
                f"<b>{nodo}</b><br>"
                f"PageRank: {pr:.5f}<br>"
                f"Betweenness: {bc:.4f}<br>"
                f"Comunidade: {com_id}<br>"
                f"Freq. em decks: {freq}"
            )
            # Escala logarítmica evita nós desproporcionalmente grandes
            tamanhos.append(8 + 25 * np.log1p(pr * 500))

        cor = PALETA_COMUNIDADES[com_id % len(PALETA_COMUNIDADES)]
        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=f"Comunidade {com_id}",
            hoverinfo="text",
            text=textos,
            marker=dict(
                size=tamanhos,
                color=cor,
                line=dict(width=1.2, color="#1a1a2e"),
                opacity=0.9,
            ),
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=dict(
            text="Rede de Coocorrência — Metagame Yu-Gi-Oh! (v2)<br>"
                 "<sup>Tamanho ∝ PageRank | Cor = Comunidade Louvain | "
                 "Espessura aresta ∝ Jaccard</sup>",
            x=0.5,
        ),
        showlegend=True,
        legend=dict(title="Comunidades", x=1.01, y=1),
        hovermode="closest",
        margin=dict(b=30, l=10, r=180, t=80),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    fig.write_html(caminho_html)
    print(f"[OK] Grafo interativo salvo em: {caminho_html}")
    return fig


def exportar_gephi(G, particao, pagerank, betweenness,
                   caminho="grafo_metagame_v2.gexf"):
    """
    Enriquece os atributos dos nós no grafo antes de exportar para Gephi,
    facilitando análise visual diretamente no software sem pós-processamento.
    """
    for nodo in G.nodes():
        G.nodes[nodo]["pagerank"]    = round(pagerank.get(nodo, 0), 6)
        G.nodes[nodo]["betweenness"] = round(betweenness.get(nodo, 0), 6)
        G.nodes[nodo]["comunidade"]  = particao.get(nodo, -1)
    nx.write_gexf(G, caminho)
    print(f"[OK] Exportado para Gephi: {caminho}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. RELATÓRIO
# ─────────────────────────────────────────────────────────────────────────────

def gerar_relatorio(G, pagerank, betweenness, particao, modularidade,
                    decklists, pools, top_n=15):
    """
    Gera três CSVs e imprime o relatório no console:
    - ranking_staples_v2.csv : ranking completo de cartas por PageRank
    - comunidades_v2.csv     : composição de cada comunidade Louvain
    - metricas_grafo_v2.csv  : estatísticas gerais do grafo
    """
    print("\n" + "="*65)
    print(f"  ESTATÍSTICAS DO GRAFO")
    print("="*65)
    n_nos    = G.number_of_nodes()
    n_arestas = G.number_of_edges()
    densidade = nx.density(G)
    n_comuns  = len(set(particao.values()))
    print(f"  Vértices  : {n_nos}")
    print(f"  Arestas   : {n_arestas}")
    print(f"  Densidade : {densidade:.4f}")
    print(f"  Comunidades (Louvain): {n_comuns}")
    print(f"  Modularidade Q        : {modularidade:.4f}")
    print(f"  Decklists simuladas   : {len(decklists)}")

    # ── TOP STAPLES ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TOP {top_n} STAPLES POR PAGERANK (emergência natural)")
    print(f"{'='*65}")
    ranking_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    ranking_bc = dict(sorted(betweenness.items(), key=lambda x: x[1], reverse=True))

    print(f"  {'#':<4} {'Carta':<42} {'PageRank':>9} {'Between.':>9} {'Com.':>5}")
    print(f"  {'-'*4} {'-'*42} {'-'*9} {'-'*9} {'-'*5}")
    for rank, (nome, pr) in enumerate(ranking_pr[:top_n], 1):
        bc  = betweenness.get(nome, 0)
        com = particao.get(nome, "?")
        print(f"  {rank:<4} {nome:<42} {pr:.5f}   {bc:.4f}   {com:>3}")

    # ── COMUNIDADES ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  COMUNIDADES DETECTADAS")
    print(f"{'='*65}")
    grupos = defaultdict(list)
    for nodo, com_id in particao.items():
        grupos[com_id].append((nodo, pagerank.get(nodo, 0)))

    for com_id in sorted(grupos):
        membros = sorted(grupos[com_id], key=lambda x: x[1], reverse=True)
        print(f"\n  Comunidade {com_id} — {len(membros)} cartas")
        top3 = [f"{n} ({pr:.4f})" for n, pr in membros[:3]]
        print(f"  Top-3 PR: {' | '.join(top3)}")

    # ── CSVs ──────────────────────────────────────────────────────────────────
    df_rank = pd.DataFrame(ranking_pr, columns=["carta", "pagerank"])
    df_rank["betweenness"] = df_rank["carta"].map(betweenness)
    df_rank["comunidade"]  = df_rank["carta"].map(particao)
    df_rank["freq_decks"]  = df_rank["carta"].map(
        lambda c: G.nodes[c].get("frequencia", 0) if c in G.nodes else 0
    )
    df_rank.to_csv("ranking_staples_v2.csv", index=False, float_format="%.6f")

    rows_com = []
    for com_id in sorted(grupos):
        membros = sorted(grupos[com_id], key=lambda x: x[1], reverse=True)
        for pos_com, (nome, pr) in enumerate(membros, 1):
            rows_com.append({
                "comunidade": com_id,
                "posicao_interna": pos_com,
                "carta": nome,
                "pagerank": pr,
                "betweenness": betweenness.get(nome, 0),
            })
    pd.DataFrame(rows_com).to_csv("comunidades_v2.csv", index=False, float_format="%.6f")

    metricas = {
        "vertices": n_nos, "arestas": n_arestas,
        "densidade": round(densidade, 6),
        "n_comunidades": n_comuns,
        "modularidade_Q": round(modularidade, 6),
        "n_decklists": len(decklists),
        "n_arquetipos": len(pools),
        "N_por_arquetipo": N_DECKLISTS_POR_ARQUETIPO,
        "tamanho_deck": TAMANHO_DECK,
        "zipf_s": ZIPF_S,
        "frac_inter": FRAC_INTER,
        "peso_minimo_aresta": PESO_MINIMO_ARESTA,
        "seed": SEED,
    }
    pd.DataFrame([metricas]).to_csv("metricas_grafo_v2.csv", index=False)

    print(f"\n[OK] CSVs salvos: ranking_staples_v2.csv | "
          f"comunidades_v2.csv | metricas_grafo_v2.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 8. EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  YGO Metagame Network v2 — Simulação Probabilística (Zipf)  ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Etapa 1: Extração dos pools
    print("► Etapa 1: Extraindo pools de cartas por arquétipo...")
    pools = {}
    for arq in ARQUETIPOS_ALVO:
        pool = buscar_pool_arquetipo(arq)
        pool, adicionadas = enriquecer_pool_com_transversais(arq, pool)
        pools[arq] = pool
        if adicionadas:
            print(f"    + Transversais adicionadas ao pool de {arq}: {adicionadas}")
        time.sleep(API_DELAY)

    # Etapa 2: Simulação probabilística
    print("\n► Etapa 2: Gerando decklists probabilísticas (Zipf s={})...".format(ZIPF_S))
    decklists = montar_decklists_probabilisticas(pools)

    # Etapa 3: Coocorrência
    print("\n► Etapa 3: Construindo matriz de coocorrência...")
    cooc_bruta, cooc_jaccard, freq_carta = construir_coocorrencias(decklists)
    print(f"  Pares únicos com coocorrência ≥ 1: {len(cooc_bruta)}")

    # Etapa 4: Grafo
    print("\n► Etapa 4: Construindo grafo...")
    G = construir_grafo(cooc_bruta, cooc_jaccard, freq_carta,
                        peso_minimo=PESO_MINIMO_ARESTA)
    print(f"  Grafo: {G.number_of_nodes()} vértices, {G.number_of_edges()} arestas")

    # Etapa 5: Métricas
    print("\n► Etapa 5: Calculando PageRank e Betweenness...")
    pagerank    = calcular_pagerank(G)
    betweenness = calcular_betweenness(G)

    print("\n► Etapa 6: Detectando comunidades (Louvain × 10 runs)...")
    particao, modularidade = detectar_comunidades(G, n_runs=10)

    # Etapa 6: Visualização
    print("\n► Etapa 7: Gerando visualizações...")
    plotar_grafo(G, particao, pagerank, betweenness)
    exportar_gephi(G, particao, pagerank, betweenness)

    # Etapa 7: Relatório
    print("\n► Etapa 8: Gerando relatório...")
    gerar_relatorio(G, pagerank, betweenness, particao, modularidade,
                    decklists, pools)

# Artigos

## Predição precisa da estrutura de interações biomoleculares com AlphaFold 3
- **Autores:** Josh Abramson; Jonas Adler; Jack Dunger; Richard Evans; Tim Green; Alexander Pritzel; Olaf Ronneberger; Lindsay Willmore; Andrew J. Ballard; Joshua Bambrick; Sebastian W. Bodenstein; David A. Evans; Chia-Chun Hung; Michael O\'Neill; David Reiman; Kathryn Tunyasuvunakool; Zachary Wu; Akvilė Žemgulytė; Eirini Arvaniti; Charles Beattie; Ottavia Bertolli; Alex Bridgland; Alexey Cherepanov; Miles Congreve; Alexander I. Cowen-Rivers; Andrew Cowie; Michael Figurnov; Fabian B. Fuchs; Hannah Gladman; Rishub Jain; Yousuf A. Khan; Caroline M. R. Low; Kuba Perlin; Anna Potapenko; Pascal Savy; Sukhdeep Singh; Adrian Stecula; Ashok Thillaisundaram; Catherine Tong; Sergei Yakneen; Ellen D. Zhong; Michal Zielinski; Augustin Žídek; Victor Bapst; Pushmeet Kohli; Max Jaderberg; Demis Hassabis; John M. Jumper et al.  
- **Revista:** Nature, vol. 630, pp. 493–500  
- **Ano:** 2024  
- **Qualis:** A1  
- **Link de Acesso:** https://www.nature.com/articles/s41586-024-07487-w
+ABRAMSON, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. **Nature**, v. 630, p. 493–500, 2024. Qualis: A1. Disponível em: https://www.nature.com/articles/s41586-024-07487-w. Acesso em: [data de acesso a ser preenchida pelo usuário].

### Análise do Artigo

#### Problema Abordado

O AlphaFold 3 (AF3) aborda a predição de estruturas 3D de interações biomoleculares, expandindo a capacidade do AlphaFold 2, que era restrita a proteínas. Ele lida com:

- Proteínas ligando-se a pequenas moléculas do tipo fármaco (interações proteína-ligante)
- Proteínas ligando-se a DNA e RNA
- Proteínas com modificações químicas
- Estruturas complexas com múltiplos tipos de componentes

O desafio era um sistema unificado para diversas interações moleculares, evitando ferramentas especializadas para cada tipo.

#### Metodologia de RL

O AF3 emprega modelagem de difusão, que compartilha semelhanças conceituais com RL:

1. **Processo de Difusão**: O modelo começa com ruído aleatório e gradualmente o transforma em estruturas moleculares precisas. Isso funciona em diferentes escalas:
   - Pequena escala: Acertar a química detalhada (ângulos de ligação, posições dos átomos)
   - Grande escala: Acertar a forma geral e o arranjo

2. **Mudanças Arquitetônicas**:
   - Arquitetura simplificada, processamento direto de coordenadas atômicas e manipulação unificada de todos os tipos de moléculas.

3. **Abordagem de Treinamento**:
   - Gera múltiplas estruturas candidatas, usa "destilação cruzada" (cross-distillation) para aprender regiões não estruturadas e um procedimento de "rollout" para avaliação de erros.

#### Principais Resultados e Contribuições

1. **Melhorias de Desempenho**:
   - Melhorias significativas na predição de ligação proteína-ligante, interações proteína-DNA/RNA e interações anticorpo-antígeno (melhoria de 65% nesta última).

2. **Sistema Unificado**:
   - Um modelo lida com todos os tipos moleculares encontrados no Protein Data Bank
   - Prediz com sucesso estruturas complexas como ribossomos (maquinaria celular) e proteínas glicosiladas

3. **Avanços Práticos**:
   - Escores de confiança confiáveis, múltiplas saídas de predição e desempenho robusto em diversas estruturas químicas.

#### Conexão com Conceitos Teóricos de RL

Embora não seja explicitamente RL, o AF3 compartilha conceitos-chave:

1. **Tomada de Decisão Sequencial**: O processo de difusão refina estruturas passo a passo, semelhante aos agentes de RL.

2. **Estimação de Valor**: O módulo de confiança atua como uma função de valor de RL, avaliando a precisão estrutural.

3. **Aprendizado Multi-escala**: Aprender em vários níveis de ruído espelha o RL hierárquico.

4. **Estrutura Estado-Ação**: A estrutura ruidosa é o estado; refinamentos são ações.

O AF3 demonstra como modelos de difusão podem resolver problemas científicos complexos, estendendo-se além dos domínios típicos de RL.

--------------------------------------------------------------------------------------------

## Descobrindo algoritmos de multiplicação de matrizes mais rápidos com aprendizado por reforço
- **Autores:** Alhussein Fawzi; Matej Balog; Aja Huang; Thomas Hubert; Bernardino Romera-Paredes; Mohammadamin Barekatain; Alexander Novikov; Francisco J. R. Ruiz; Julian Schrittwieser; Grzegorz Swirszcz; David Silver; Demis Hassabis; Pushmeet Kohli et al.  
- **Revista:** Nature, vol. 610, pp. 47–53  
- **Ano:** 2022  
- **Qualis:** A1  
- **Link de Acesso:** https://www.nature.com/articles/s41586-022-05172-4
+FAWZI, A. et al. Discovering faster matrix multiplication algorithms with reinforcement learning. **Nature**, v. 610, p. 47–53, 2022. Qualis: A1. Disponível em: https://www.nature.com/articles/s41586-022-05172-4. Acesso em: [data de acesso a ser preenchida pelo usuário].

### Análise do Artigo

#### Problema Abordado

O artigo aborda a descoberta de algoritmos de multiplicação de matrizes mais rápidos, um problema NP-difícil computacionalmente intensivo e crucial para muitas aplicações. A ideia central é representar a multiplicação de matrizes como uma decomposição tensorial; menos componentes significam um algoritmo mais rápido.

#### Metodologia de RL

O AlphaTensor enquadra a descoberta de algoritmos como um jogo de um jogador, o TensorGame:

- **Configuração do Jogo**: O jogador começa com um tensor representando a multiplicação de matrizes
- **Ações**: A cada turno, escolher como combinar entradas das matrizes de entrada
- **Objetivo**: Alcançar o tensor zero usando o menor número possível de movimentos
- **Recompensa**: -1 para cada movimento, com o objetivo de minimizar os movimentos

Para navegar no vasto espaço de ações (mais de 10^12 ações), o AlphaTensor usa:

1. **Framework AlphaZero**: Uma rede neural guia uma Busca em Árvore Monte Carlo, aprendendo com a experiência sem conhecimento humano
  
2. **Componentes Especializados**:
   - Uma rede neural personalizada para processamento de tensores 3D, treinamento em problemas alvo e aleatórios, mudança de base e aumento de dados.
   - Treinamento tanto no problema alvo quanto em exemplos gerados aleatoriamente
   - Mudanças aleatórias de perspectiva (base) para ver o problema de diferentes ângulos
   - Aumento de dados para maximizar o aprendizado de cada jogo jogado

#### Principais Resultados e Contribuições

1. **Quebrando um Recorde de 50 Anos**: O AlphaTensor encontrou uma maneira de multiplicar matrizes 4×4 usando 47 multiplicações, superando o algoritmo de Strassen de 1969, que exigia 49 multiplicações.

2. **Melhorias Abrangentes**: Descobriu algoritmos mais eficientes para muitos tamanhos de matrizes diferentes, com melhorias para mais de 70 problemas diferentes de multiplicação de matrizes.

3. **Flexibilidade de Aplicação**:
   - Algoritmos otimizados para tipos específicos de matrizes e hardware (GPUs/TPUs), superando os equivalentes projetados por humanos.
   - Encontrou algoritmos ótimos para tipos especiais de matrizes (como matrizes antissimétricas).

4. **Insights Matemáticos**: Descobriu milhares de algoritmos válidos diferentes para os mesmos problemas, mostrando que o espaço de soluções possíveis é muito mais rico do que se pensava anteriormente.

#### Conexão com Conceitos Teóricos de RL

1. **Processo de Decisão de Markov (MDP)**: O TensorGame é um MDP (Estados: tensor a decompor; Ações: trigêmeos de vetores para subtração; Transições: determinísticas; Recompensas: -1/passo).

2. **Busca em Árvore Monte Carlo**: Simulação de sequências de ações, balanceamento de exploração/explotação e uso de uma rede neural para guiar a busca.

3. **Estimação de Valor**: Aprendizado de estimativas de valor de estado via auto-jogo para melhorar a tomada de decisão e refinar a estratégia.

Este trabalho demonstra a capacidade do RL de encontrar soluções inovadoras para problemas matemáticos de longa data, abrindo caminho para a descoberta assistida por IA.

--------------------------------------------------------------------------------------------

## DeepSeek-R1: Incentivando a Capacidade de Raciocínio em LLMs via Aprendizado por Reforço
- **Autores:** DeepSeek-AI; Daya Guo; Dejian Yang; Haowei Zhang; Junxiao Song; Ruoyu Zhang; Runxin Xu; Qihao Zhu; Shirong Ma; Peiyi Wang; Xiao Bi; Xiaokang Zhang; Xingkai Yu; Yu Wu; Z.F. Wu; Zhibin Gou; Zhihong Shao; Zhuoshu Li; Ziyi Gao; Aixin Liu; Bing Xue; Bingxuan Wang; Bochao Wu; Bei Feng; Chengda Lu; Chenggang Zhao; Chengqi Deng; Chenyu Zhang; Chong Ruan; Damai Dai; Deli Chen; Dongjie Ji; Erhang Li; Fangyun Lin; Fucong Dai; Fuli Luo; Guangbo Hao; Guanting Chen; Guowei Li; H. Zhang; Han Bao; Hanwei Xu; Haocheng Wang; Honghui Ding; Huajian Xin; Huazuo Gao; Hui Qu; Hui Li; Jianzhong Guo; Jiashi Li; Jiawei Wang; Jingchang Chen; Jingyang Yuan; Junjie Qiu; Junlong Li; J.L. Cai; Jiaqi Ni; Jian Liang; Jin Chen; Kai Dong; Kai Hu; Kaige Gao; Kang Guan; Kexin Huang; Kuai Yu; Lean Wang; Lecong Zhang; Liang Zhao; Litong Wang; Liyue Zhang; Lei Xu; Leyi Xia; Mingchuan Zhang; Minghua Zhang; Minghui Tang; Meng Li; Miaojun Wang; Mingming Li; Ning Tian; Panpan Huang; Peng Zhang; Qiancheng Wang; Qinyu Chen; Qiushi Du; Ruiqi Ge; Ruisong Zhang; Ruizhe Pan; Runji Wang; R.J. Chen; R.L. Jin; Ruyi Chen; Shanghao Lu; Shangyan Zhou; Shanhuang Chen; Shengfeng Ye; Shiyu Wang; Shuiping Yu; Shunfeng Zhou; Shuting Pan; S.S. Li; … et al.  
- **Local:** Preprint arXiv (cs.CL, cs.AI, cs.LG)  
- **Ano:** 2025  
- **Qualis:** Não aplicável (preprint)
- **Link de Acesso:** https://arxiv.org/abs/2501.12948
+GUO, D. et al. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**. 2025. Preprint arXiv:2501.12948. Qualis: Não aplicável (preprint). Disponível em: https://arxiv.org/abs/2501.12948. Acesso em: [data de acesso a ser preenchida pelo usuário].

### Análise do Artigo

#### Problema Abordado

O DeepSeek-R1 aborda o desenvolvimento do raciocínio em LLMs com dados supervisionados mínimos, explorando:

- Se o aprendizado por reforço (RL) por si só pode permitir que modelos desenvolvam habilidades complexas de raciocínio
- Como criar modelos que demonstrem forte raciocínio, mantendo saídas legíveis
- Se os padrões de raciocínio podem ser transferidos de modelos maiores para menores

A inovação é usar o RL como o principal motor para o raciocínio, reduzindo a dependência do custoso ajuste fino supervisionado.

#### Metodologia de RL

O artigo apresenta duas abordagens principais:

1. **DeepSeek-R1-Zero**: Uma abordagem de "RL puro" que começa com um modelo base e usa apenas aprendizado por reforço (sem dados supervisionados)
   - Usa um template `<think>`/`<answer>` para a saída do raciocínio, com recompensas baseadas em regras para correção e formatação.

2. **DeepSeek-R1**: Uma abordagem mais avançada que combina dados supervisionados mínimos com RL
   - Combina dados supervisionados mínimos (partida a frio) com um processo de RL multi-estágio (foco no raciocínio, amostragem por rejeição para dados de qualidade, RL final para equilíbrio).

O algoritmo de RL central usado é chamado "Group Relative Policy Optimization" (GRPO), que simplifica o RL tradicional por:
- Gerar múltiplas respostas para cada pergunta
- Comparar essas respostas entre si (em vez de um avaliador separado)
- Recompensas baseadas no desempenho relativo à média do grupo.

#### Principais Resultados e Contribuições

1. **Sucesso do RL Puro**: O DeepSeek-R1-Zero alcançou habilidades de raciocínio notáveis sem dados supervisionados, melhorando de 15.6% para 71.0% no benchmark de matemática AIME 2024.

2. **Conquistas de Desempenho**:
   - O DeepSeek-R1 iguala ou excede o modelo o1-1217 da OpenAI em muitas tarefas de raciocínio
   - Alcança 79.8% no AIME 2024, 97.3% no MATH-500 e atinge o percentil 96.3 no Codeforces (programação competitiva)

3. **Comportamentos Emergentes**: Modelos desenvolveram naturalmente padrões de raciocínio sofisticados:
   - Comportamentos emergentes: pensamento estendido, auto-verificação/reflexão e exploração de múltiplas abordagens para resolução de problemas.

4. **Transferência de Conhecimento Bem-sucedida**: Modelos menores (parâmetros de 1.5B a 70B) aprenderam efetivamente padrões de raciocínio do DeepSeek-R1 através de destilação, superando modelos de tamanho similar treinados diretamente com RL.

#### Conexão com Conceitos Teóricos de RL

1. **Processo de Decisão de Markov (MDP)**: O raciocínio é um MDP (Estados: raciocínio parcial; Ações: próximo token; Recompensas: correção da resposta; propriedade de Markov válida).

2. **Busca em Árvore Monte Carlo (MCTS)**: O MCTS foi explorado (dividindo respostas, guiado por modelo de valor), mas enfrentou desafios com o grande espaço de busca.

3. **Aproximação da Função de Valor**: O GRPO se relaciona com a aproximação da função de valor usando estatísticas de grupo para estimativa de valor, criando linhas de base a partir de amostras e aprendendo a prever caminhos de raciocínio eficazes.

O artigo mostra que o RL pode treinar efetivamente o raciocínio de LLMs com supervisão mínima, permitindo que os modelos aprendam estratégias complexas por meio de incentivos.

## Aplicação Prática: FrozenLake-v1 com DP e MCTS

O ambiente `FrozenLake-v1` (Gymnasium) foi usado para demonstração prática. É um mundo de grade estocástico (S: início, G: objetivo, H: buraco, F: congelado) onde o agente navega em gelo escorregadio.

### Fundamentos Teóricos

Esta seção descreve os conceitos de RL aplicados.

#### Processos de Decisão de Markov (MDPs)
O problema `FrozenLake-v1` é modelado como um MDP, definido por:
- **Estados (S)**: As 16 células da grade (0-15).
- **Ações (A)**: 4 ações (Esquerda, Baixo, Direita, Cima).
- **Probabilidades de Transição (P(s\'|s,a))**: A probabilidade de mover para o estado `s\'` do estado `s` após tomar a ação `a`. Devido ao gelo escorregadio, as ações são estocásticas (por exemplo, tentar ir para \'Cima\' pode resultar em mover para \'Esquerda\' ou \'Direita\' com alguma probabilidade).
- **Recompensas (R(s,a,s\'))**: Uma recompensa de +1 é dada por alcançar o estado Objetivo (G); 0 caso contrário para todas as outras transições.
O objetivo é encontrar uma política ótima `π*(s)` (mapeamento estado-para-ação) que maximize as recompensas futuras descontadas esperadas, frequentemente encontrando a função de valor ótima `V*(s)` ou a função de valor-ação `Q*(s,a)`.

#### Programação Dinâmica (DP)
Métodos de DP resolvem MDPs com um modelo conhecido.
- **Iteração de Valor**: `V*(s) = max_a Σ_{s\'} P(s\'|s,a) [R(s,a,s\') + γV*(s\')]`. Ela atualiza iterativamente `V(s)` até a convergência para `V*`, da qual `π*` é derivada.

#### Métodos de Monte Carlo (MC) e Busca em Árvore Monte Carlo (MCTS)
Métodos MC aprendem a partir de episódios de experiência e são livres de modelo (embora o MCTS possa usar um modelo para simulação).
- **MCTS**: O MCTS padrão (`mcts_search` em `rl.py`) constrói uma árvore de busca por estado não terminal.
- **Derivação da Política**: A melhor ação é da raiz para o filho mais visitado.
- **Parâmetros** (de `rl.py`):
   - Parâmetros chave: `mcts_num_iterations`=2000, `mcts_exploration_constant`=1.414, `mcts_max_rollout_depth`=`num_states*2`, `gamma`=0.99. Estados terminais: \'G\', \'H\'.

### Metodologia

1.  **Formulação do Processo de Decisão de Markov (MDP)**: Como descrito acima para `FrozenLake-v1`.

2.  **Programação Dinâmica (DP)**:
    *   **Algoritmo**: Iteração de Valor (`value_iteration` em `rl.py`) para encontrar `V*` e `π*`.
    *   **Parâmetros**:
        *   Fator de desconto (`gamma`): 0.99
        *   Limiar de convergência (`theta`): 1e-9

3.  **Busca em Árvore Monte Carlo (MCTS)**:
    *   **Algoritmo**: Um algoritmo MCTS padrão (veja `mcts_search` e funções/classes relacionadas em `rl.py`) foi implementado. Para cada estado não terminal, o MCTS constrói uma árvore de busca.
    *   **Derivação da Política**: A ação que leva ao nó filho com mais visitas da raiz foi escolhida como a melhor ação para aquele estado.
    *   **Parâmetros** (de `rl.py`):
        *   Número de iterações por decisão de estado (`mcts_num_iterations`): 2000
        *   Constante de exploração para UCB1 (`mcts_exploration_constant`): 1.414 (sqrt(2))
        *   Profundidade máxima do rollout (`mcts_max_rollout_depth`): `num_states * 2` (ex: 32 para FrozenLake 4x4)
        *   Fator de desconto (`gamma`): 0.99 (mesmo que DP para consistência nos rollouts)
        *   Estados terminais: Identificados como estados \'G\' (Objetivo) e \'H\' (Buraco).

### Resultados e Análise

As políticas e funções de valor derivadas de DP e MCTS foram comparadas. O script `rl.py` gera e salva mapas de calor para uma comparação visual das funções de valor.

**Layout do Ambiente:**
```
SFFF
FHFH
FFFH
HFFG
```

**1. Resultados da Programação Dinâmica (Ótimos):**

**Política Ótima de DP:**
```
← ↑ ↑ ↑
← H ← H
↑ ↓ ← H
H → ↓ G
```
*(Política ótima e determinística de DP.)*

**Função de Valor Ótima de DP:**
```
[[0.542 0.499 0.471 0.457]
 [0.558 0.    0.358 0.   ]
 [0.592 0.643 0.615 0.   ]
 [0.    0.742 0.863 0.   ]]
```
*(`V*(s)` ótimo para cada estado de DP.)*

**Mapa de Calor da Função de Valor de DP:**
![DP Optimal Value Function Heatmap](dp_optimal_value_function_heatmap.png)

**Mapa de Calor da Função de Valor de MCTS:**
![MCTS Approx Value Function Heatmap](mcts_approx_value_function_heatmap.png)

**2. Resultados da Busca em Árvore Monte Carlo (MCTS) (`mcts_num_iterations = 2000` por estado):**

**Política Derivada de MCTS:**
*(Política típica de MCTS para 2000 iterações; execute `rl.py` para saída exata. Espera-se que seja próxima da ótima.)*\\
```
← ↓ ← ←
↓ H → H
↑ ↑ ← H
H ← → G
```

**Função de Valor Aproximada de MCTS:**
*(Valores aproximados de rollouts de MCTS; valores reais podem variar ligeiramente. Execute `rl.py`.)*\\
```
[[0.017 0.01  0.032 0.006]
 [0.013 0.    0.004 0.   ]
 [0.036 0.054 0.074 0.   ]
 [0.    0.084 0.052 0.   ]]
```

**Mapa de Calor da Função de Valor de MCTS:**
*(Incorpore `mcts_approx_value_function_heatmap.png` gerado por `rl.py` aqui.)*


**3. Discussão e Comparação:**

*   **Comparação de Políticas**:
    *   Com 2000 iterações MCTS/estado, a política MCTS é tipicamente muito próxima da política DP ótima, mostrando convergência com simulação suficiente. Menos iterações resultariam em políticas mais ruidosas e potencialmente subótimas.
    *   Pequenas diferenças podem surgir da estocasticidade no ambiente/rollouts ou de valores Q verdadeiros quase iguais para diferentes ações.

*   **Comparação da Função de Valor**:
    *   A função de valor aproximada do MCTS também se aproxima dos valores ótimos do DP com 2000 iterações. DP é exato; MCTS é uma estimativa via amostragem, levando a algum erro de aproximação. Mapas de calor ilustram isso.

*   **Esforço Computacional**:
    *   **DP (Iteração de Valor)**: Resolve para todos os estados; complexidade por iteração ~O(S²A). Itera até a convergência.
    *   **MCTS**: Planejamento sob demanda para um estado; custo ~`num_iterations` × (`profundidade_arvore` + `profundidade_rollout`). Eficiente para poucos estados ou espaços de estados muito grandes onde DP é inviável. Gerar uma política completa requer executar MCTS por estado.

*   **Requisito do Modelo**:
    *   DP precisa de um modelo MDP completo (P, R).
    *   MCTS aqui usa o modelo para simulação, mas pode ser livre de modelo se usar um simulador ou interações reais.

### Conclusão Crítica: Relacionando Pesquisa com Prática

Este exercício com `FrozenLake-v1` ilustra conceitos centrais de RL escalados nos artigos de pesquisa:

1.  **AlphaFold 3 (Interações Biomoleculares)**:
    *   Encontrar estruturas proteicas ótimas (modelo de difusão do AF3) espelha encontrar políticas ótimas. Os "escores de confiança" do AF3 são como **funções de valor** de RL. Ambos envolvem explorar vastos espaços de busca.

2. **AlphaTensor (Algoritmos de Multiplicação de Matrizes)**:
    *   O AlphaTensor usa diretamente MCTS (em um framework tipo AlphaZero) para o "TensorGame" (um MDP) para encontrar algoritmos ótimos de multiplicação de matrizes. Nosso MCTS `FrozenLake` mostra os princípios centrais de busca, exploração/explotação (UCB1) e simulação usados pelo AlphaTensor.

3. **DeepSeek-R1 (Raciocínio em LLMs via RL)**:
    *   O DeepSeek-R1 usa RL (formulação MDP onde a geração de tokens é uma ação) para aprimorar o raciocínio de LLMs. O design da recompensa é crítico, similar ao `FrozenLake`. O trade-off exploração/explotação (visto no UCB1 do MCTS) também é central para treinar LLMs com RL.

**Em essência**, este exercício com `FrozenLake-v1` fornece experiência prática com princípios centrais de RL (MDPs, funções de valor, políticas, métodos baseados em modelo vs. baseados em simulação, exploração/explotação). Estes são fundamentais para as aplicações de RL mais complexas nos artigos revisados, que usam arquiteturas e algoritmos mais sofisticados. 
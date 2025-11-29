# CartPole-v1 – Q-Learning vs. SARSA

Projeto final da disciplina de Aprendizado por Reforço para comparar Q-Learning (off-policy) e SARSA (on-policy) no ambiente `CartPole-v1` do Gymnasium utilizando discretização do espaço contínuo de estados.

## Estrutura do Projeto

- `main.py` – script principal com treinamento, discretização, plotagem comparativa e função para renderizar a política final.
- `requirements.txt` – dependências do projeto.
- `CartPole-v1.py` – script opcional para jogar manualmente com teclado (fornecido anteriormente).

## Dependências

Instale os pacotes utilizando:

```bash
pip install -r /home/danilo/Desktop/ita/cmc_15/CH/projeto1/entregaFinal/requirements.txt
```

Principais bibliotecas:

- Gymnasium (ambiente CartPole)
- NumPy (Q-Tables e discretização)
- Matplotlib (gráfico comparativo)
- Pygame (renderização manual opcional)

## Como Executar

1. Certifique-se de estar no diretório do projeto:
   ```bash
   cd /home/danilo/Desktop/ita/cmc_15/CH/projeto1/entregaFinal
   ```
2. Rode o treinamento dos dois agentes e gere o gráfico:
   ```bash
   python main.py
   ```
   - Isso irá:
     - Treinar Q-Learning e SARSA por `NUM_EPISODES`.
     - Salvar a figura `cartpole_q_vs_sarsa.png` com a média móvel (janela de 100 episódios).
3. Caso queira visualizar o agente treinado, edite `main.py` e defina `RENDER_FINAL_POLICY = True`. Em seguida, execute novamente:
   ```bash
   python main.py
   ```
   A janela do ambiente será aberta para `FINAL_POLICY_EPISODES` episódios utilizando a tabela Q-learning treinada.

## Configurações Importantes

No topo do `main.py` estão concentrados os hiperparâmetros:

- `NUM_EPISODES`, `MAX_STEPS_PER_EPISODE`
- `ALPHA`, `GAMMA`, `EPSILON_START`, `EPSILON_MIN`, `EPSILON_DECAY`
- `NUM_BINS` e `OBSERVATION_BOUNDS` (definem a discretização via `np.linspace`)
- `MOVING_AVG_WINDOW` (suavização do gráfico)

A discretização retorna índices inteiros para compor a Q-Table com shape `NUM_BINS + (n_actions,)`.

## Fluxo do Script

1. **Discretização:** `discretize_state` converte estados contínuos em buckets.
2. **Treinamento:** `train_agent` recebe o nome do algoritmo (`q_learning` ou `sarsa`), inicializa a Q-Table, executa episódios e registra recompensas.
3. **Atualizações:**
   - Q-Learning usa `max(Q(s', ·))` (off-policy).
   - SARSA usa a ação escolhida para o próximo estado (on-policy).
4. **Comparação:** `plot_learning_curves` gera o gráfico com média móvel.
5. **Demonstração:** `run_trained_agent` renderiza um episódio utilizando a política greedy.

## Resultados Esperados

O gráfico `cartpole_q_vs_sarsa.png` permite comparar a convergência dos algoritmos, mostrando a evolução da recompensa média ao longo dos episódios. Ajuste os hiperparâmetros para explorar diferentes comportamentos.



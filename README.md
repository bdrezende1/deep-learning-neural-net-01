# deep-learning-neural-net-01
Aborda o problema de classificação de dígitos manuscritos do MNIST usando a arquitetura de rede neural InceptionV3. Conclui-se que a combinação do InceptionV3 e transferência de aprendizado permite a construção de um classificador eficaz para o MNIST.
---

## **Descrição Geral**
O problema consiste em classificar corretamente dígitos manuscritos (0-9) a partir de imagens em escala de cinza, com dimensões de 28x28 pixels. O conjunto de dados MNIST é amplamente usado como benchmark em aprendizado de máquina.

O código utiliza a biblioteca **PyTorch** para construção e treinamento do modelo.

---

## **Seções do Código**

### **1. Bibliotecas Utilizadas**
As principais bibliotecas empregadas no código são:
- **`numpy`**: Para operações matemáticas e manipulação de dados.
- **`torch`** e **`torchvision`**: Para construção, treinamento e manipulação de redes neurais.
- **`matplotlib.pyplot`**: Para visualização de dados.
- **`time`**: Para medir a duração do treinamento.

---

### **2. Pré-processamento dos Dados**
- **Transformação**: As imagens são convertidas para tensores com `transforms.ToTensor()`.
- **Carregamento**: Utiliza-se `torchvision.datasets.MNIST` para carregar os datasets de treino e validação.
- **Buffers de dados**: Os datasets são divididos em lotes de tamanho 64 e embaralhados para o treinamento utilizando `DataLoader`.

---

### **3. Visualização Inicial**
Uma amostra da imagem é exibida usando `plt.imshow` para confirmar que os dados estão corretamente carregados.

---

### **4. Estrutura da Rede Neural**
A rede implementada possui três camadas principais:
1. **Camada de Entrada**: Conecta 784 neurônios (28x28 pixels) a 128 neurônios.
2. **Camada Intermediária**: Conecta 128 neurônios a 64.
3. **Camada de Saída**: Conecta 64 neurônios a 10, representando os dígitos de 0 a 9.

#### **Detalhes Adicionais**
- **Função de ativação**: A ReLU é usada nas camadas de entrada e intermediária.
- **Função de saída**: LogSoftmax, para calcular a probabilidade logarítmica dos dígitos.

---

### **5. Treinamento do Modelo**
A função `treino` executa as seguintes etapas:
- Define o otimizador **SGD (Stochastic Gradient Descent)** com:
  - Taxa de aprendizado: `0.01`
  - Momentum: `0.5`
- Utiliza a função de perda **NLLLoss (Negative Log Likelihood)**.
- Realiza o treinamento por **10 épocas**, calculando a perda em cada iteração e ajustando os pesos e vieses da rede.

---

### **6. Validação do Modelo**
A função `validacao`:
- Desativa o autograd (para maior desempenho na validação).
- Faz previsões sobre os dados de validação.
- Compara as previsões com os rótulos reais e calcula a precisão.

---

### **7. Configuração e Execução**
- O modelo é movido para a GPU, caso disponível, utilizando `torch.device`.
- A classe `Modelo` é instanciada e passada para as funções de treinamento e validação.

---

## **Resumo Final**
O script implementa uma rede neural densa simples para resolver o problema de classificação de dígitos manuscritos no MNIST. Embora o texto mencione o modelo **InceptionV3**, ele não é implementado diretamente. 

Se precisar de mais informações sobre como executar, modificar ou expandir este código, estou à disposição para ajudar! 😊

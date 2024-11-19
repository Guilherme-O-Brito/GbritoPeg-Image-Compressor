# GbritoPeg-Image-Compressor
# Implementação de Compressão JPEG

Este repositório contém minha implementação do algoritmo de compressão JPEG, desenvolvida como parte das aulas de **Computação Gráfica e Multimídia** no **Instituto Nacional de Telecomunicações - Inatel** com o professor **Marcelo Vinícius Cysneiros Aragão**. O objetivo principal deste projeto foi aprofundar o entendimento sobre o funcionamento interno da compressão JPEG e facilitar a visualização prática do processo.

## Sobre o Projeto

Nesta implementação, criei uma versão levemente adaptada do algoritmo JPEG. Algumas diferenças em relação à especificação original incluem:

- O canal alpha (transparência) não é descartado, sendo processado junto com os demais canais da imagem.
- O código gera uma imagem comprimida, mas não passa pela etapa de codificação final (como Huffman ou codificação aritmética, etapa ainda em desenvolvimento). 
  - **Atenção:** O tamanho do arquivo gerado **não deve ser usado para calcular a taxa de compressão**, pois ele inclui apenas as etapas de perdas da compressão JPEG.

## Etapas Implementadas

O projeto contempla as seguintes etapas do processo de compressão JPEG:

1. **Conversão para o espaço de cor YCrCb**:
   - Realiza a separação dos canais de luminância e crominância.
2. **Subamostragem dos canais de cor**:
   - Reduz a resolução dos canais de crominância para simular a sensibilidade do olho humano a cores.
3. **Transformada Discreta do Cosseno (DCT)**:
   - Aplica a DCT em blocos de 8x8 pixels para compactar as informações de frequência.
4. **Quantização**:
   - Reduz a precisão dos coeficientes da DCT usando tabelas de quantização padrão.
5. **Ordenação Zig-Zag**:
   - Percorre os blocos quantizados em uma ordem específica para agrupar os coeficientes mais significativos.

Embora o processo esteja incompleto (faltam etapas como a codificação final), já é possível observar os efeitos da compressão na qualidade da imagem, comparando a imagem original em formato BMP com a imagem resultante.

## Estrutura do Código

O arquivo principal é `codec.py`,  arquivo `runMe.ipynb` é um notebook preparado com as chamadas das funções do arquivo `codec.py`, organizadas na ordem necessária para realizar a compressão e descompressão da imagem. Além disso, ele exibe as imagens original e comprimida lado a lado e mostra os canais de crominância Cr e Cb da imagem final após a descompressão, que concentra a maior parte das implementações do processo. Como tentei implementar manualmente todas as etapas possíveis, o código é extenso e detalhado.

## Exemplos

### Imagem Original
![Imagem Original](imgs/Arara-Azul.bmp)

### Imagem Após Compressão (Sem Codificação Final)
![Imagem Comprimida](tests/arara-azul-compressed.png)

### Comparação Entre Original e Comprimida
![Comparação](tests/comparação.png)

## Este projeto foi inspirado pelas aulas da disciplina e por materiais de referência disponíveis na internet.

## Funcionalidades Futuras

Este projeto ainda está em desenvolvimento, e planejo adicionar várias melhorias e funcionalidades nas próximas etapas. Abaixo estão algumas delas:

### 1. Codificação do Arquivo Final
Atualmente, o projeto não inclui a etapa de codificação final (como a codificação Huffman ou aritmética), responsável por comprimir ainda mais os dados gerados. Estou trabalhando nessa etapa para que o processo complete todas as fases especificadas no padrão JPEG.

### 2. Interface Gráfica
Pretendo criar uma interface gráfica intuitiva para facilitar a utilização do compressor, permitindo que usuários selecionem imagens, ajustem parâmetros e visualizem os resultados de maneira mais prática.

### 3. Paralelização do Código
Para melhorar a eficiência e reduzir o tempo de execução, especialmente ao processar imagens grandes, planejo implementar paralelização no código. Isso permitirá que diferentes etapas ou blocos sejam processados simultaneamente, aproveitando o desempenho de sistemas com múltiplos núcleos.

## Contribuições

Se você tiver sugestões ou melhorias, fique à vontade para abrir uma issue ou enviar um pull request. 

## Licença

Este projeto está licenciado sob a [GPL-3.0 License](LICENSE).

---

Obrigado por visitar este repositório!

# GbritoPeg-Image-Compressor
# Implementação de Compressão JPEG

Este repositório contém minha implementação do algoritmo de compressão JPEG, desenvolvida como parte das aulas de **Computação Gráfica e Multimídia** no **Instituto Nacional de Telecomunicações - Inatel** com o professor **Marcelo Vinícius Cysneiros Aragão**. O objetivo principal deste projeto foi aprofundar o entendimento sobre o funcionamento interno da compressão JPEG e facilitar a visualização prática do processo.

---

## Sobre o Projeto

Nesta implementação, criei uma versão levemente adaptada do algoritmo JPEG. Alguns destaques incluem:

- **O canal alpha (transparência) é mantido no processo de compressão.** Ele passa pelas etapas de **DCT**, **quantização**, e **codificação**, utilizando a tabela de quantização da luminância.
- O arquivo resultante (`compressed.gpeg`) contém o canal alpha junto com os demais canais, garantindo a preservação de transparência durante a compressão e descompressão.
- **O processo agora está 100% funcional,** incluindo a codificação e decodificação completas. A **taxa de compressão printada no arquivo `runMe.ipynb` já pode ser considerada confiável.**

---

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
5. **Codificação e manutenção do canal alpha**:
   - O canal alpha é processado exatamente como o canal de luminância, utilizando a tabela de quantização de luminância.
6. **Ordenação Zig-Zag**:
   - Percorre os blocos quantizados em uma ordem específica para agrupar os coeficientes mais significativos.
7. **Codificação Final (RLE e Huffman)**:
   - Após a ordenação Zig-Zag, os blocos de 8x8 pixels passam por um processo de codificação otimizado para compressão:
     - **RLE (Run-Length Encoding) não convencional**: 
       - Salva a quantidade de zeros consecutivos antes de um número não nulo, o tamanho em bits do número não nulo e, em seguida, o próprio número não nulo.
       - Esse método explora a alta ocorrência de zeros após a quantização, característica dos coeficientes JPEG.
     - **Codificação Huffman**:
       - Gera uma tabela de símbolos baseada nos pares `(quantidade de zeros, tamanho em bits do número não nulo)` derivados do RLE.
       - Os números não nulos são codificados diretamente em binário, enquanto os símbolos Huffman comprimem os pares.
     - **Metadados salvos no início do arquivo**:
       - Inclui os shapes originais da imagem e o dicionário de Huffman para decodificação. Esses dados são gravados em binário sem compressão.
       - Antes dos blocos de imagem codificados, as tabelas de quantização (após quantização, Zig-Zag, RLE e Huffman) são inseridas de forma semelhante aos blocos.
     - **O arquivo resultante é gravado em bytes para reduzir o consumo de memória**

O arquivo `runMe.ipynb` demonstra o funcionamento completo do codec, incluindo a compressão e descompressão, e exibe as imagens original e comprimida lado a lado, além dos canais de crominância Cr e Cb da imagem descomprimida.

---

## Exemplos

### Imagem Original
![Imagem Original](imgs/Arara-Azul.bmp)

### Imagem Após Compressão
![Imagem Comprimida](tests/arara-azul-compressed.png)

### Comparação Entre Original e Comprimida
![Comparação](tests/comparação.png)

---

## Funcionalidades Finais

Com as melhorias realizadas, o codec agora oferece:

1. **Manutenção do canal alpha**:
   - Preserva a transparência original da imagem.
2. **Taxa de compressão confiável**:
   - A taxa printada no arquivo `runMe.ipynb` pode ser usada como referência para avaliar o desempenho do codec.

---

## Funcionalidades Futuras

Embora o codec esteja funcional, algumas funcionalidades podem ser implementadas no futuro:

1. **Interface Gráfica**:
   - Facilitará a interação do usuário, permitindo carregamento de imagens e visualização em tempo real.
2. **Paralelização**:
   - Melhorará o desempenho em sistemas com múltiplos núcleos.

---

---

## 📚 Documentação Detalhada

Para uma explicação mais profunda sobre o funcionamento das funções, etapas de compressão e decodificação, além de exemplos práticos de uso do codec, confira a documentação completa disponível no arquivo **`doc.ipynb`**.

👉 [Acesse a Documentação Detalhada](doc.ipynb)

Nesta documentação, você encontrará:
- Explicação passo a passo das funções principais.
- Tutoriais simplificados e avançados para usar o codec.
- Testes realizados durante o desenvolvimento.
- Detalhes técnicos sobre cada etapa do processo.

A documentação foi organizada de forma sequencial para facilitar o entendimento do fluxo de compressão e descompressão.

---


## Contribuições

Sugestões e melhorias são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

---

## Licença

Este projeto está licenciado sob a [GPL-3.0 License](LICENSE).

---

Obrigado por visitar este repositório!

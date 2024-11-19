####### este arquivo contem apenas algumas fnuções criadas durante o desenvolvimento do codigo que funcionam
####### porem não estão sendo utilizadas no codigo final


# Faz a redução de escala da imagem
import numpy as np

def downScaling(image: np.ndarray, scaleFactor):
    # Fator de escala de redução é calculado e convertido para inteiro
    scaleFactor = int(scaleFactor**(-1))
    # Cria um array para salvar o resultado
    result = np.zeros(image.shape, dtype=np.float32)
    # Calcula tamanho final da imagem apos a redução
    size_x = int(image.shape[0] / scaleFactor)
    size_y = int(image.shape[1] / scaleFactor)

    # Calcula os pixels da imagem reduzida
    for i in range(int(size_x)):
        for j in range(int(size_y)):
            temp = np.array([0], dtype=np.float32)
            for x in range(scaleFactor):
                for y in range(scaleFactor):
                    temp += image[i*scaleFactor + x, j*scaleFactor + y]
            result[i,j] = temp / (scaleFactor*scaleFactor)
    
    # recortando o array de resultado para que fique com o tamanho final da imagem (elimando area sem pixels devido ao tamanho original da imagem)
    result = result[0:size_x,0:size_y]

    return result

# upscaling atraves do metodo nearest neighbor (QUALIDADE FINAL MUITO INFERIOR)
def nnb(image, scaleFactor):
    ls, cs, p = image.shape[0] * scaleFactor, image.shape[1] * scaleFactor, image.shape[2]
    result = np.zeros((ls,cs,p), dtype=np.uint8)

    for i in range(ls):
        for j in range(cs):
            ny = int(np.floor(i * (image.shape[0] / ls)))
            nx = int(np.floor(j * (image.shape[1] / cs)))

            result[i,j] = image[ny,nx]
    
    return result


# Faz o aumento e escala da imagem usando interpolação bilinear (QUALIDADE MUITO SUPERIOR POREM MAIS LENTO)
def upScaling(image: np.ndarray, scaleFactor):
    # Convertendo o scaleFactor apra inteiro
    scaleFactor = int(scaleFactor)
    
    # Calculando qual o tamanho da imagem final apos o upScaling
    size_x = int(image.shape[0] * scaleFactor)
    size_y = int(image.shape[1] * scaleFactor)
    
    # Salvando o shape final da imagem que sera gerada
    shape = (size_x,size_y)
    

    # calculando fatores
    sl = image.shape[0] / size_x 
    sc = image.shape[1] / size_y

    # Array responsavel por armazenar a imagem resultante
    result = np.zeros(shape, dtype=np.float32)

    # Realiza o upscaling usando interpolação bilinear
    for l in range(size_x):
        for c in range(size_y):
            lf = l * (sl) 
            cf = c * (sc)
            l0 = int(np.floor(lf)) 
            if l0+1 >= image.shape[0] : l0 = image.shape[0] - 2
            c0 = int(np.floor(cf))
            if c0+1 >= image.shape[1] : c0 = image.shape[1] - 2
            deltaL = lf - l0
            deltaC = cf - c0
            result[l,c] = ( image[l0,c0] * (1 - deltaL) * (1 - deltaC)
                            + image[l0 + 1, c0] * deltaL * (1 - deltaC)
                            + image[l0,c0 + 1] * (1 - deltaL) * deltaC
                            + image[l0 + 1,c0 + 1] * deltaL * deltaC
            )
    
    return result









def decodeWithTable(encoded_data, huffman_codes):
    # Inverter a tabela de códigos para facilitar a busca
    code_to_symbol = {v: k for k, v in huffman_codes.items()}

    decoded_blocks = []
    for block in encoded_data:
        decoded_block = []
        current_code = ''
        for bit in block:
            current_code += bit
            if current_code in code_to_symbol:
                symbol = eval(code_to_symbol[current_code])  # Converte string para tupla
                decoded_block.append(symbol)
                current_code = ''  # Reinicia para o próximo código
        decoded_blocks.append(decoded_block)
    return decoded_blocks

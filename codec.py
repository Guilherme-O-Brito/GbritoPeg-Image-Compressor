import numpy as np
import cv2

# função que converte uma imagem para o espaço de cor YCrCb retornando sempre a imgagem obtida em um array de numpy
def toYCrCb(image) -> np.ndarray:
    # convertendo a imagem para um np array
    arr = np.array(image, dtype=np.float32)

    r = arr[:,:,0]
    g = arr[:,:,1]
    b = arr[:,:,2]

    # criando um np array de mesmas dimensoes da imagem preenchido com zeros
    yCrCb = np.zeros(arr.shape, dtype=np.float32)
    
    yCrCb[:,:,0] = (0.299 * r) + (0.587 * g) + (0.114 * b)
    yCrCb[:,:,1] = (r - yCrCb[:,:,0]) / 1.402
    yCrCb[:,:,2] = (b - yCrCb[:,:,0]) / 1.772
    
    # verifica existencia do canal alpha
    if arr.shape[2] == 4:
        yCrCb[:,:,3] = arr[:,:,3]
    
    return yCrCb

# função responsavel por converter o espaço de cor de YCrCb para RGB sem perder o canal alpha
def toRGB(image:np.ndarray) -> np.ndarray:

    # cria o array de resultados com mesmo shape da imagem em YCrCb
    result = np.zeros(image.shape, dtype=np.float32)

    # realiza as operações de converção definindo os canais R G B da nova imagem e o canal alpha caso exista
    result[:,:,0] = image[:,:,0] + (1.402 * image[:,:,1])
    result[:,:,1] = image[:,:,0] - (0.299 * 1.402 / 0.587) * image[:,:,1] - (0.114 * 1.772 / 0.587) * image[:,:,2]
    result[:,:,2] = image[:,:,0] + (1.772 * image[:,:,2])

    # verifica existencia do canal alpha
    if image.shape[2] == 4:
        result[:,:,3] = image[:,:,3]

    # Corrige erros na escala de cor causados pela reconstrução dos canais de chorminancia no sub sampling
    result[result > 255] = 255
    result[result < 0] = 0

    return result.astype(np.uint8)

# Aplica subamostragem de cores a, a -> fatores de escala para subsampling
def subSampling(a:int, b:int, image):
    # convertendo a imagem para um np array
    arr = np.array(image, dtype=np.float32)

    # salva o shape inicial da imagem
    imgShape = arr.shape

    # trata o caso de um dos fatores receber 0
    if a == 0: a = 1
    if b == 0: b = 1

    # separando a imagem em 3 arrays, um de luminancia e outros dois de chrominancia
    y = arr[:,:,0]
    cr = arr[:,:,1]
    cb = arr[:,:,2]

    # aplica sub sampling nos canais cr e cb
    crSub = cr[::a,::b] 
    cbSub = cb[::a,::b] 

    # verifica se a imagem possui canal alpha e realiza a operação de sub sampling   
    if imgShape[2] == 4: 
        alpha = arr[:,:,3]
        alphaSub = alpha
        
    else:
        # em caso da imagem não possuir canal alpha cria um canal alpha de mesmo tamanho dos canais crSub,cbSub preenchido com 255 para não modificar a imagem
        alphaSub = np.full(shape=y.shape, fill_value=255, dtype=np.float32)

    return y, crSub, cbSub, alphaSub

# função responsavel por ser o operação inversa a operação de sub sampling
def upSampling(y:np.ndarray, crSub:np.ndarray, cbSub:np.ndarray, alphaSub:np.ndarray, a:int, b:int) -> np.ndarray: 
    
    result = np.zeros((y.shape[0],y.shape[1],4), dtype=np.float32)
    
    result[:,:,0] = y
    
    for i in range(0, crSub.shape[0]): 
        for j in range(0, crSub.shape[1]):
            result[i*a,j*b,1] = crSub[i,j]
            result[i*a,j*b,2] = cbSub[i,j]
            #result[i*a,j*b,3] = alphaSub[i,j]
            if(j*b)+1 < result.shape[1]:
                for k in range(0,b):
                    result[i*a,(j*b)+k,1] = crSub[i,j]
                    result[i*a,(j*b)+k,2] = cbSub[i,j]
                    #result[i*a,(j*b)+k,3] = alphaSub[i,j]
        if (i*a)+1 < result.shape[0]:
            for l in range(0,a):
                result[(i*a)+l,:,1] = result[i*a,:,1]
                result[(i*a)+l,:,2] = result[i*a,:,2]
                #result[(i*a)+l,:,3] = result[i*a,:,3] 
    result[:,:,3] = alphaSub
    return result

# realiza as transformadas nos canais da imagem e ja aplica a quantização
def compress(y:np.ndarray, cr:np.ndarray, cb:np.ndarray, alpha:np.ndarray, qty:np.ndarray, qtc:np.ndarray):

    # como os blocos de quantização tem tamanho fixo define-se uma constante com o tamanho do lado
    BLOCKSIZE = 8 
    # valor de suavização multiplicado a tabela dos blocos de transformada para ajudar a evitar que valores medios sejam perdidos no arredondamento (no maximo 1 ou no minimo 0.9)
    SUAVIZACAO = 1
    # normaliza os canais subtraindo 128 de todos eles
    y = y - 128
    cr = cr - 128
    cb = cb - 128

    # verifica se o tamanho de y pode ser dividido igualmente em blocos de 8 por 8 pixel, caso não seja ajusta o shape de y adicionando linhas e colunas de 0
    yWidth, yHeight = int(np.ceil(y.shape[1] / BLOCKSIZE) * BLOCKSIZE), int(np.ceil(y.shape[0] / BLOCKSIZE) * BLOCKSIZE)
    if y.shape[1] % BLOCKSIZE == 0 and y.shape[0] % BLOCKSIZE == 0:
        yPadding = y.copy()
    else:
        yPadding = np.zeros((yHeight,yWidth))
        yPadding[0:y.shape[0],0:y.shape[1]] += y

    crWidth, crHeight = int(np.ceil(cr.shape[1] / BLOCKSIZE) * BLOCKSIZE), int(np.ceil(cr.shape[0] / BLOCKSIZE) * BLOCKSIZE)
    # como os canais Cr e Cb tem sempre o mesmo tamanho os dois são ajustados dentro do mesmo if
    if cr.shape[1] % BLOCKSIZE == 0 and cr.shape[0] % BLOCKSIZE == 0:
        crPadding = cr.copy()
        cbPadding = cb.copy()
    else:
        crPadding = np.zeros((crHeight,crWidth))
        cbPadding = np.zeros((crHeight,crWidth))
        crPadding[0:cr.shape[0],0:cr.shape[1]] += cr
        cbPadding[0:cb.shape[0],0:cb.shape[1]] += cb

    # definindo quantidade de blocos na horizontal e vertical para luminancia e chrominancia
    jBlocksY = int(yPadding.shape[1] / BLOCKSIZE) # quantidade de blocos na horizontal para Y
    iBlocksY = int(yPadding.shape[0] / BLOCKSIZE) # quantidade de blocos na vertical para Y
    # novamente como os canais cr e cb tem sempre as mesmas dimensões so é necessario definir quantidade de blocos para um deles
    jBlocksC = int(crPadding.shape[1] / BLOCKSIZE) # quantidade de blocos na horizontal para Chrominancias
    iBlocksC = int(crPadding.shape[0] / BLOCKSIZE) # quantidade de blocos na vertical para Chrominancias

    yDct, crDct, cbDct = np.zeros((yHeight, yWidth)), np.zeros((crHeight, crWidth)), np.zeros((crHeight, crWidth))

    yQauntized, crQuantized, cbQuantized = np.zeros((yHeight, yWidth)), np.zeros((crHeight, crWidth)), np.zeros((crHeight, crWidth))

    yZigzag, crZigzag, cbZigzag = np.zeros((yHeight * yWidth)), np.zeros((crHeight * crWidth)), np.zeros((crHeight * crWidth))
    
    #yRle, crRle, cbRle = np.array([]), np.array([]), np.array([])

    # calculando as transformadas e as arrays ja quantizadas de Y
    for i in range(iBlocksY):
        for j in range(jBlocksY):
            index = (i * jBlocksY + j) * BLOCKSIZE**2
            # aplicando a transformada no bloco
            yDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.dct(yPadding[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE]) * SUAVIZACAO
            # aplicando a quantização no bloco
            yQauntized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = np.round((yDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] ) / qty)
            # fazendo a varredura em zigzag
            yZigzag[index:index + BLOCKSIZE**2] = zigzagVector(yQauntized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])

    # calculando as transformadas e as arrays ja quantizadas de Chrominancia
    for i in range(iBlocksC):
        for j in range(jBlocksC):
            index = (i * jBlocksC + j) * BLOCKSIZE**2
            # aplicando a transformada no bloco
            crDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.dct(crPadding[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE]) * SUAVIZACAO
            cbDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.dct(cbPadding[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE]) * SUAVIZACAO
            # aplicando a quantização no bloco
            crQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = np.round((crDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] ) / qtc)
            cbQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = np.round((cbDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] ) / qtc)
            # fazendo a varredura em zigzag
            crZigzag[index:index + BLOCKSIZE**2] = zigzagVector(crQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])
            cbZigzag[index:index + BLOCKSIZE**2] = zigzagVector(cbQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])

    # codificando em RLE
    yRle = rleEncode(yZigzag)
    crRle = rleEncode(crZigzag)
    cbRle = rleEncode(cbZigzag)
    # salva os shapes originais dos canais para serem restaurados posteriormente na decodificação da imagem
    originalShapes = [y.shape,cr.shape,cb.shape,alpha.shape]
    paddedShapes = [(yHeight,yWidth),(crHeight,crWidth),(crHeight,crWidth)]

    shapes = {
        'original':originalShapes,
        'padded':paddedShapes
    }
    
    return yQauntized, crQuantized, cbQuantized, yRle, crRle, cbRle, alpha, shapes

# realiza o processo inverso da função anterior, desquantiza e ja aplica a transformada inversa, retornando ja os 4 canais da imagem prontos para continuar a descompressão
def deCompress(yCompressed:np.ndarray, crCompressed:np.ndarray, cbCompressed:np.ndarray, alphaCompressed:np.ndarray, shapes: np.ndarray, qty:np.ndarray, qtc:np.ndarray):

    BLOCKSIZE = 8

    originalShapes = shapes['original']
    paddedShapes = shapes['padded']

    # definindo quantidade de blocos na horizontal e vertical para luminancia e chrominancia
    jBlocksY = int(paddedShapes[0][1] / BLOCKSIZE) # quantidade de blocos na horizontal para Y
    iBlocksY = int(paddedShapes[0][0] / BLOCKSIZE) # quantidade de blocos na vertical para Y
    # novamente como os canais cr e cb tem sempre as mesmas dimensões so é necessario definir quantidade de blocos para um deles
    jBlocksC = int(paddedShapes[1][1] / BLOCKSIZE) # quantidade de blocos na horizontal para Chrominancias
    iBlocksC = int(paddedShapes[1][0] / BLOCKSIZE) # quantidade de blocos na vertical para Chrominancias

    yIDCT, crIDCT, cbIDCT = np.zeros(paddedShapes[0]), np.zeros(paddedShapes[1]), np.zeros(paddedShapes[2])

    yDeQauntized, crDeQuantized, cbDeQuantized = np.zeros(paddedShapes[0]), np.zeros(paddedShapes[1]), np.zeros(paddedShapes[2])

    yReconstructed, crReconstructed, cbReconstructed = np.zeros(paddedShapes[0]), np.zeros(paddedShapes[1]), np.zeros(paddedShapes[2])

    yRLEDecoded, crRLEDecoded, cbRLEDecoded = np.array([]), np.array([]), np.array([])

    # decodificando o RLE
    yCompressed = rleDecode(yCompressed)
    crCompressed = rleDecode(crCompressed)
    cbCompressed = rleDecode(cbCompressed)

    # calculando as transformadas e as arrays ja quantizadas de Y
    for i in range(iBlocksY):
        for j in range(jBlocksY):
            index = (i * jBlocksY + j) * BLOCKSIZE**2
            # reconstruindo o bloco em zigzag
            yReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = zigzagReconstruct(yCompressed[index:index + BLOCKSIZE**2])
            # desquantizando o bloco
            yDeQauntized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = (yReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] ) * qty
            # aplicando a transformada inversa
            yIDCT[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.idct(yDeQauntized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])

    # calculando as transformadas e as arrays ja quantizadas de Chrominancia
    for i in range(iBlocksC):
        for j in range(jBlocksC):
            index = (i * jBlocksC + j) * BLOCKSIZE**2
            # reconstruindo o bloco em zigzag
            crReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = zigzagReconstruct(crCompressed[index:index + BLOCKSIZE**2])
            cbReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = zigzagReconstruct(cbCompressed[index:index + BLOCKSIZE**2])
            # desquantizando o bloco
            crDeQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = (crReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])  * qtc    
            cbDeQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = (cbReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])  * qtc
            # aplicando a transformada inversa
            crIDCT[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.idct(crDeQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])
            cbIDCT[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.idct(cbDeQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])

    # recortando os blocos para eliminar o padding inserido na compressão
    shape = originalShapes[0]
    if yIDCT.shape > shape:
        yIDCT = yIDCT[0:shape[0],0:shape[1]]

    # recortando os blocos para eliminar o padding inserido na compressão
    shape = originalShapes[1]
    if crIDCT.shape > shape:
        crIDCT = crIDCT[0:shape[0],0:shape[1]]
        cbIDCT = cbIDCT[0:shape[0],0:shape[1]]

    yIDCT = yIDCT + 128
    crIDCT = crIDCT + 128
    cbIDCT = cbIDCT + 128

    return yIDCT, crIDCT, cbIDCT, alphaCompressed

# função responsavel por gerar o vetor zigzag de um canal da imagem, o parametro desta função é um array bidmensional 
# contendo apenas um canal da imagem como um todo
def zigzagVector(channel:np.ndarray):
    # cria o vetor zigzag com a quantidade correta de linhas
    zigzag = np.zeros((channel.shape[0] * channel.shape[1]))
    # define altura e largura maxima
    h, w = channel.shape
    # iniciando duas variaveis responsaveis por iterar no canal recebido
    y, x  = 0, 0
    # indicações de direção iniciais para o algoritmo iterar
    right, left, up, down = True, False, False, False
    # itera sobre o canal original em zigzag salvando os valores lidos
    for i in range(zigzag.shape[0]):
        # caso ele va apenas para a direita
        if right == True and left == False and up == False and down == False:
            #print('Direção: para direita')
            # atribui o valor encontrado no passo da diagonal no array zigzag
            zigzag[i] = channel[y,x]
            #print(f'ZigZag: {zigzag[i]}')
            # efetua o passo para direita
            x += 1
            # verifica se o array original ja chegou na ultima linha
            if y+1 >= h:
                # define que o proximo passo sera uma diagonal para cima e para direita
                right, left, up, down = True, False, True, False
            else:
                # define que o proximo passo sera uma diagonal para baixo e para esquerda
                right, left, up, down = False, True, False, True
            continue
        # caso ele va para uma diagonal para baixo e esquerda
        if right == False and left == True and up == False and down == True:
            #print('Direção: para baixo e para esquerda')
            # atribui o valor encontrado no passo da diagonal no array zigzag
            zigzag[i] = channel[y,x]
            #print(f'ZigZag: {zigzag[i]}')
            # efetua o passo na diagonal para baixo e esquerda
            x -= 1
            y += 1
            # verifica se o array original ja chegou na ultima linha
            if y+1 >= h:
                # define que o proximo passo sera para direita
                right, left, up, down = True, False, False, False
            # verifica se o array original ja voltou para a primeira coluna
            elif x-1 < 0:
                # define que o proximo passo sera para baixo
                right, left, up, down = False, False, False, True
            else:
                # define que o proximo passo sera uma diagonal para baixo e para esquerda
                right, left, up, down = False, True, False, True
            continue
        # caso ele va para baixo
        if right == False and left == False and up == False and down == True:
            #print('Direção: para baixo')
            # atribui o valor encontrado no passo da diagonal no array zigzag
            zigzag[i] = channel[y,x]
            #print(f'ZigZag: {zigzag[i]}')
            # efetua o passo para baixo
            y += 1
            # verifica se o array original ja chegou na ultima coluna
            if x+1 >= w:
                # define que o proximo passo sera uma diagonal para baixo e para esquerda
                right, left, up, down = False, True, False, True
            else:
                # define que o proximo passo sera uma diagonal para cima e para direita
                right, left, up, down = True, False, True, False
            continue
        # caso ele va para uma diagonal para cima e direita
        if right == True and left == False and up == True and down == False:
            #print('Direção: para cima e para direita')
            # atribui o valor encontrado no passo da diagonal no array zigzag
            zigzag[i] = channel[y,x]
            #print(f'ZigZag: {zigzag[i]}')
            # efetua o passo na diagonal para cima e para direita
            x += 1
            y -= 1
            # verifica se o array original ja chegou na ultima coluna
            if x+1 >= w: 
                # define que o proximo passo sera para baixo
                right, left, up, down = False, False, False, True
            # verifica se o array original ja voltou para a primeira linha
            elif y-1 < 0:
                # define que o proximo passo sera para direita
                right, left, up, down = True, False, False, False
            else:
                # define que o proximo passo sera uma diagonal para cima e para direita
                right, left, up, down = True, False, True, False
            continue

    return zigzag

# função responsavel por reconstruir um canal da imagem a partir de um vetor zigzag, o parametro desta função é um array unidimensional 
# contendo apenas o vetor zigzag do canal a ser reconstruido
def zigzagReconstruct(zigzag:np.ndarray) -> np.ndarray:
    # criando array para reconstruir um canal com base no vetor zigzag recebido
    channel = np.zeros((8,8))
    # define altura e largura maxima
    h, w = channel.shape
    # iniciando duas variaveis responsaveis por iterar no canal recebido
    y, x  = 0, 0
    # indicações de direção iniciais para o algoritmo iterar
    right, left, up, down = True, False, False, False
    # itera sobre o vetor zigzag reconstruindo o canal original
    for i in range(zigzag.shape[0]):
        # caso ele va apenas para a direita
        if right == True and left == False and up == False and down == False:
            #print('Direção: para direita')
            # atribui o valor encontrado no passo da diagonal no array zigzag
            channel[y,x] = zigzag[i] 
            #print(f'ZigZag: {zigzag[i]}')
            # efetua o passo para direita
            x += 1
            # verifica se o array original ja chegou na ultima linha
            if y+1 >= h:
                # define que o proximo passo sera uma diagonal para cima e para direita
                right, left, up, down = True, False, True, False
            else:
                # define que o proximo passo sera uma diagonal para baixo e para esquerda
                right, left, up, down = False, True, False, True
            continue
        # caso ele va para uma diagonal para baixo e esquerda
        if right == False and left == True and up == False and down == True:
            #print('Direção: para baixo e para esquerda')
            # atribui o valor encontrado no passo da diagonal no array zigzag
            channel[y,x] = zigzag[i]
            #print(f'ZigZag: {zigzag[i]}')
            # efetua o passo na diagonal para baixo e esquerda
            x -= 1
            y += 1
            # verifica se o array original ja chegou na ultima linha
            if y+1 >= h:
                # define que o proximo passo sera para direita
                right, left, up, down = True, False, False, False
            # verifica se o array original ja voltou para a primeira coluna
            elif x-1 < 0:
                # define que o proximo passo sera para baixo
                right, left, up, down = False, False, False, True
            else:
                # define que o proximo passo sera uma diagonal para baixo e para esquerda
                right, left, up, down = False, True, False, True
            continue
        # caso ele va para baixo
        if right == False and left == False and up == False and down == True:
            #print('Direção: para baixo')
            # atribui o valor encontrado no passo da diagonal no array zigzag
            channel[y,x] = zigzag[i]
            #print(f'ZigZag: {zigzag[i]}')
            # efetua o passo para baixo
            y += 1
            # verifica se o array original ja chegou na ultima coluna
            if x+1 >= w:
                # define que o proximo passo sera uma diagonal para baixo e para esquerda
                right, left, up, down = False, True, False, True
            else:
                # define que o proximo passo sera uma diagonal para cima e para direita
                right, left, up, down = True, False, True, False
            continue
        # caso ele va para uma diagonal para cima e direita
        if right == True and left == False and up == True and down == False:
            #print('Direção: para cima e para direita')
            # atribui o valor encontrado no passo da diagonal no array zigzag
            channel[y,x] = zigzag[i]
            #print(f'ZigZag: {zigzag[i]}')
            # efetua o passo na diagonal para cima e para direita
            x += 1
            y -= 1
            # verifica se o array original ja chegou na ultima coluna
            if x+1 >= w: 
                # define que o proximo passo sera para baixo
                right, left, up, down = False, False, False, True
            # verifica se o array original ja voltou para a primeira linha
            elif y-1 < 0:
                # define que o proximo passo sera para direita
                right, left, up, down = True, False, False, False
            else:
                # define que o proximo passo sera uma diagonal para cima e para direita
                right, left, up, down = True, False, True, False
            continue

    return channel


def rleEncode(vector:np.ndarray) -> np.ndarray:
    # criando array para armazenar a codificação
    encoded = []
    # salvando tamanho do vetor original
    codeLength = len(vector)

    # cria um iterator iniciado em 0
    i = 0
    # percorre o vetor encontrando repetições e codificando
    while i < codeLength:
        # contador de ocorrencias do valor em i
        count = 1
        # continua percorrendo a frente do caracter atual buscando repetições
        while i < codeLength - 1 and vector[i] == vector[i + 1]:
            count += 1
            i += 1
        i += 1
        # salvando a sequencia encontrada
        encoded.append(vector[i - 1])
        encoded.append(count)

    return np.array(encoded, dtype=vector.dtype)

def rleDecode(encoded:np.ndarray) -> np.ndarray:
    # Calculando tamanho total do array 
    total_size = int(sum(encoded[1::2]))
    # criando array para armazenar a decodificação
    decoded = np.empty(total_size, dtype=encoded.dtype)

    index = 0
    for i in range(0,len(encoded),2):
        item = encoded[i]
        count = int(encoded[i + 1])
        decoded[index:index + count] = item
        index += count

    return decoded

# esta função deve receber um vetor unidimensional de inteiros gerado pelo algoritmo zigzag aplicado a um bloco de pixels
def jpegRLEEncode(vector: np.ndarray) -> list:

    # lista responsavel por armazenar a codificação RLE
    encoded = []
    # contador de zeros consecutivos
    zero_count = 0

    for value in vector:
        if value == 0:
            zero_count += 1
        else:
            # Quando encontra um valor não nulo, salva o par (run-length, tamanho do valor)
            # calcula o tamanho em bits necessario para armazenar o valor não nulo encontrado
            size = int(np.floor(np.log2(abs(value))) + 1) if value != 0 else 0
            encoded.append((zero_count, size))
            encoded.append(value)
            # Reseta a contagem de zeros
            zero_count = 0
    
    # Adiciona o símbolo EOB (End of Block) se o resto é zero
    if zero_count > 0:
        encoded.append((0, 0))  # EOB para indicar o fim do bloco

    return encoded

# a lista recebida deve ter o formato [(qtd_zeros,bit_qtd),int]
def jpegRLEDecode(encoded: list) -> np.ndarray:

    # lista responsavel por armazenar o codigo RLE decodificado
    decoded = []

    for i in range(0, len(encoded), 2):
        zeros, size = encoded[i]
        if size == 0 and zeros == 0:
            # EOB encontrado, preenche o restante com zeros
            while len(decoded) < 64:
                decoded.append(0)
            break
        else:
            # Preenche os zeros até o próximo valor
            decoded.extend([0] * zeros)
            # Adiciona o valor real
            value = encoded[i + 1]
            decoded.append(value)

    return np.array(decoded, dtype=int)



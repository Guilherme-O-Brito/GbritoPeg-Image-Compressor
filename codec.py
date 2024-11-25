import numpy as np
from PIL import Image
import cv2
import heapq
from collections import Counter

QTY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  # Tabela de qunatização da luminancia
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

QTC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],  # Tabela de quantização das chorminancias
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])

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
            if(j*b)+1 < result.shape[1]:
                for k in range(0,b):
                    result[i*a,(j*b)+k,1] = crSub[i,j]
                    result[i*a,(j*b)+k,2] = cbSub[i,j]
        if (i*a)+1 < result.shape[0]:
            for l in range(0,a):
                result[(i*a)+l,:,1] = result[i*a,:,1]
                result[(i*a)+l,:,2] = result[i*a,:,2]

    result[:,:,3] = alphaSub
    return result

# realiza as transformadas nos canais da imagem e ja aplica a quantização
def compress(y:np.ndarray, cr:np.ndarray, cb:np.ndarray, alpha:np.ndarray, qty:np.ndarray, qtc:np.ndarray, ssv:int, ssh:int):

    # como os blocos de quantização tem tamanho fixo define-se uma constante com o tamanho do lado
    BLOCKSIZE = 8 
    # valor de suavização multiplicado a tabela dos blocos de transformada para ajudar a evitar que valores medios sejam perdidos no arredondamento (no maximo 1 ou no minimo 0.9)
    SUAVIZACAO = 1
    # normaliza os canais subtraindo 128 de todos eles
    y = y - 128
    alpha = alpha - 128
    cr = cr - 128
    cb = cb - 128

    # lista contendo todos os blocos codificados em RLE
    all_blocks = []

    # codificando as tabelizas de quantização em RLE para que possam ser posteriormente codificadas em huffman
    qtyRle = jpegRLEEncode(zigzagVector(qty))
    qtcRle = jpegRLEEncode(zigzagVector(qtc))
    all_blocks.append(qtyRle)
    all_blocks.append(qtcRle)

    # verifica se o tamanho de y pode ser dividido igualmente em blocos de 8 por 8 pixel, caso não seja ajusta o shape de y adicionando linhas e colunas de 0
    yWidth, yHeight = int(np.ceil(y.shape[1] / BLOCKSIZE) * BLOCKSIZE), int(np.ceil(y.shape[0] / BLOCKSIZE) * BLOCKSIZE)
    if y.shape[1] % BLOCKSIZE == 0 and y.shape[0] % BLOCKSIZE == 0:
        yPadding = y.copy()
        alphaPadding = alpha.copy()
    else:
        yPadding = np.zeros((yHeight,yWidth))
        alphaPadding = np.zeros((yHeight,yWidth))
        yPadding[0:y.shape[0],0:y.shape[1]] += y
        alphaPadding[0:alpha.shape[0],0:alpha.shape[1]] += alpha

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

    yDct, crDct, cbDct, alphaDct = np.zeros((yHeight, yWidth)), np.zeros((crHeight, crWidth)), np.zeros((crHeight, crWidth)), np.zeros((yHeight, yWidth))

    yQauntized, crQuantized, cbQuantized, alphaQuantized = np.zeros((yHeight, yWidth)), np.zeros((crHeight, crWidth)), np.zeros((crHeight, crWidth)), np.zeros((yHeight, yWidth))

    yZigzag, crZigzag, cbZigzag, alphaZigzag = np.zeros((yHeight * yWidth)), np.zeros((crHeight * crWidth)), np.zeros((crHeight * crWidth)), np.zeros((yHeight * yWidth))
    
    yRle = []
    crRle = []
    cbRle = []
    alphaRle = []

    # calculando as transformadas e as arrays ja quantizadas de Y e alpha
    for i in range(iBlocksY):
        for j in range(jBlocksY):
            index = (i * jBlocksY + j) * BLOCKSIZE**2
            # aplicando a transformada no bloco
            yDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.dct(yPadding[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE]) * SUAVIZACAO
            alphaDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.dct(alphaPadding[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE]) * SUAVIZACAO
            # aplicando a quantização no bloco
            yQauntized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = np.round((yDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] ) / qty)
            alphaQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = np.round((alphaDct[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] ) / qty)
            # fazendo a varredura em zigzag
            yZigzag[index:index + BLOCKSIZE**2] = zigzagVector(yQauntized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])
            alphaZigzag[index:index + BLOCKSIZE**2] = zigzagVector(alphaQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])
            # codificando bloco em RLE Jpeg
            rleCode = jpegRLEEncode(yZigzag[index:index + BLOCKSIZE**2])
            yRle.append(rleCode)
            all_blocks.append(rleCode)
            rleCode = jpegRLEEncode(alphaZigzag[index:index + BLOCKSIZE**2])
            alphaRle.append(rleCode)
            all_blocks.append(rleCode)

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
            # codificando blocos em RLE Jpeg
            crRleCode = jpegRLEEncode(crZigzag[index:index + BLOCKSIZE**2])
            cbRleCode = jpegRLEEncode(cbZigzag[index:index + BLOCKSIZE**2])
            crRle.append(crRleCode)
            cbRle.append(cbRleCode)
            all_blocks.append(crRleCode)
            all_blocks.append(cbRleCode)

    # salva os shapes originais dos canais para serem restaurados posteriormente na decodificação da imagem
    originalShapes = [y.shape,cr.shape,cb.shape,alpha.shape]
    paddedShapes = [(yHeight,yWidth),(crHeight,crWidth),(crHeight,crWidth)]

    shapes = {
        'original':originalShapes,
        'padded':paddedShapes
    }
    
    # gerando arvore e tabela de huffman considerando todos os blocos codificados em RLE para garantir mair eficiencia na codificação de huffman
    root, huffman_codes = generateGlobalHuffmanTable(all_blocks)
    # codificando toda a imagem, passando primeiro os parametros ssv e ssh codificados em binario, depois os shapes codificados, apos eles a tabela de huffman,
    # as tabelas de quantização e finalmente os blocos dos canais y, Cr, Cb e alpha codificados em huffman
    encoded = format(ssv,'08b') + format(ssh,'08b') + encodeShapes(shapes) + encodeHuffmanTable(huffman_codes) + huffmanEncode(all_blocks, huffman_codes)

    # calculando tamanho necessario em bits para armzaenar a imagem orinal
    img_length = originalShapes[0][0] * originalShapes[0][1] * 4 * 8
    # tamanho em bits do codigo gerado pela compressão
    compressed_length = len(encoded)
    # calculando taxa de compressão
    compression_ratio = (1 - (compressed_length / img_length)) * 100
    print(f'Tamanho em bits da imagem original: {img_length} bits')
    print(f'Tamanho do codigo da imagem apos compressão: {compressed_length} bits')
    print(f'Taxa de compressão estimada: {format(compression_ratio, '.2f')}%')

    return encoded

# realiza o processo inverso da função anterior, desquantiza e ja aplica a transformada inversa, retornando ja os 4 canais da imagem prontos para continuar a descompressão
def deCompress(code:str):

    BLOCKSIZE = 8
    # decodificando parametros usados na sub amostragem
    ssv = int(code[:8], 2)
    ssh = int(code[8:16], 2)

    # removendo parte do codigo referente as constantes de sub amostragem
    code = code[16:]

    # decodificando os shapes da imagem
    shapes, encoded_blocks = decodeShapes(code)
    # decodificando tabela de huffman
    huffman_codes, encoded_blocks = decodeHuffmanTable(encoded_blocks)
    # decoficicando todos os blocos e as tabelas de quantização para o codigo RLE
    rle_blocks, qty, qtc = huffmanDecode(encoded_blocks, huffman_codes)
    # decodificando o RLE das tabelas de quantização neste ponto elas ja estão prontas para serem usadas na descompressão
    qty = zigzagReconstruct(jpegRLEDecode(qty))
    qtc = zigzagReconstruct(jpegRLEDecode(qtc))

    originalShapes = shapes['original']
    paddedShapes = shapes['padded']

    # definindo quantidade de blocos na horizontal e vertical para luminancia e chrominancia
    jBlocksY = int(paddedShapes[0][1] / BLOCKSIZE) # quantidade de blocos na horizontal para Y
    iBlocksY = int(paddedShapes[0][0] / BLOCKSIZE) # quantidade de blocos na vertical para Y
    # novamente como os canais cr e cb tem sempre as mesmas dimensões so é necessario definir quantidade de blocos para um deles
    jBlocksC = int(paddedShapes[1][1] / BLOCKSIZE) # quantidade de blocos na horizontal para Chrominancias
    iBlocksC = int(paddedShapes[1][0] / BLOCKSIZE) # quantidade de blocos na vertical para Chrominancias

    yIDCT, crIDCT, cbIDCT, alphaIDCT = np.zeros(paddedShapes[0]), np.zeros(paddedShapes[1]), np.zeros(paddedShapes[2]), np.zeros(paddedShapes[0])

    yDeQauntized, crDeQuantized, cbDeQuantized, alphaDeQuantized = np.zeros(paddedShapes[0]), np.zeros(paddedShapes[1]), np.zeros(paddedShapes[2]), np.zeros(paddedShapes[0])

    yReconstructed, crReconstructed, cbReconstructed, alphaReconstructed = np.zeros(paddedShapes[0]), np.zeros(paddedShapes[1]), np.zeros(paddedShapes[2]), np.zeros(paddedShapes[0])

    yZigzag, crZigzag, cbZigzag, alphaZigzag = np.zeros((paddedShapes[0][1] * paddedShapes[0][0])), np.zeros((paddedShapes[1][1] * paddedShapes[1][0])), np.zeros((paddedShapes[1][1] * paddedShapes[1][0])), np.zeros((paddedShapes[0][1] * paddedShapes[0][0]))

    # contador de blocos na lista de blocos codificados em RLE
    block = 0
    # calculando as transformadas e as arrays ja quantizadas de Y e alpha
    for i in range(iBlocksY):
        for j in range(jBlocksY):
            index = (i * jBlocksY + j) * BLOCKSIZE**2
            # decodificando RLE Jpeg
            yZigzag[index:index + BLOCKSIZE**2] = jpegRLEDecode(rle_blocks[block])
            block += 1
            alphaZigzag[index:index + BLOCKSIZE**2] = jpegRLEDecode(rle_blocks[block])
            block += 1
            # reconstruindo o bloco em zigzag
            yReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = zigzagReconstruct(yZigzag[index:index + BLOCKSIZE**2])
            alphaReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = zigzagReconstruct(alphaZigzag[index:index + BLOCKSIZE**2])
            # desquantizando o bloco
            yDeQauntized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = (yReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] ) * qty
            alphaDeQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = (alphaReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] ) * qty
            # aplicando a transformada inversa
            yIDCT[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.idct(yDeQauntized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])
            alphaIDCT[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = cv2.idct(alphaDeQuantized[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE])

    # calculando as transformadas e as arrays ja quantizadas de Chrominancia
    for i in range(iBlocksC):
        for j in range(jBlocksC):
            index = (i * jBlocksC + j) * BLOCKSIZE**2
            crZigzag[index:index + BLOCKSIZE**2] = jpegRLEDecode(rle_blocks[block])
            block += 1
            cbZigzag[index:index + BLOCKSIZE**2] = jpegRLEDecode(rle_blocks[block])
            block += 1
            # reconstruindo o bloco em zigzag
            crReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = zigzagReconstruct(crZigzag[index:index + BLOCKSIZE**2])
            cbReconstructed[i * BLOCKSIZE: (i * BLOCKSIZE) + BLOCKSIZE, j * BLOCKSIZE: (j * BLOCKSIZE) + BLOCKSIZE] = zigzagReconstruct(cbZigzag[index:index + BLOCKSIZE**2])
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
        alphaIDCT = alphaIDCT[0:shape[0],0:shape[1]]

    # recortando os blocos para eliminar o padding inserido na compressão
    shape = originalShapes[1]
    if crIDCT.shape > shape:
        crIDCT = crIDCT[0:shape[0],0:shape[1]]
        cbIDCT = cbIDCT[0:shape[0],0:shape[1]]

    yIDCT = yIDCT + 128
    crIDCT = crIDCT + 128
    cbIDCT = cbIDCT + 128
    alphaIDCT = alphaIDCT + 128

    return yIDCT, crIDCT, cbIDCT, alphaIDCT, ssv, ssh

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
            # atribui o valor encontrado no passo da diagonal no array zigzag
            zigzag[i] = channel[y,x]
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
            # atribui o valor encontrado no passo da diagonal no array zigzag
            zigzag[i] = channel[y,x]
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
            # atribui o valor encontrado no passo da diagonal no array zigzag
            zigzag[i] = channel[y,x]
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
            # atribui o valor encontrado no passo da diagonal no array zigzag
            zigzag[i] = channel[y,x]
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
            # atribui o valor encontrado no passo da diagonal no array zigzag
            channel[y,x] = zigzag[i] 
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
            # atribui o valor encontrado no passo da diagonal no array zigzag
            channel[y,x] = zigzag[i]
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
            # atribui o valor encontrado no passo da diagonal no array zigzag
            channel[y,x] = zigzag[i]
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
            # atribui o valor encontrado no passo da diagonal no array zigzag
            channel[y,x] = zigzag[i]
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
            #size = int(np.floor(np.log2(abs(value))) + 1)
            size = int(value).bit_length() + 1

            encoded.append((zero_count, size))
            encoded.append(int(value))
            # Reseta a contagem de zeros
            zero_count = 0
    
    # Adiciona o símbolo EOB (End of Block) se o resto é zero
    #if zero_count > 0:
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

# classes e funções relacionadas com a codificação de huffman
class Node:
    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.frequency < other.frequency

# constroi a arvore de huffman, deve receber em chars uma lista com todos os simbolos gerados por RLE e uma lista com as respectivas frequencias de cada simbolo
def buildHuffmanTree(chars, freq):
    # criando uma lista de prioridades
    priority = [Node(char, f) for char, f in zip(chars, freq)]
    heapq.heapify(priority)

    # construindo arvore de huffman
    while len(priority) > 1:
        left_child = heapq.heappop(priority)
        right_child = heapq.heappop(priority)
        merged_node = Node(frequency=left_child.frequency + right_child.frequency)
        merged_node.left = left_child
        merged_node.right = right_child
        heapq.heappush(priority, merged_node)
    
    # retorna a raiz da arvore 
    return priority[0]

# gera os codigos de huffman para a arvore inserida e retorna o dicionario para symbolos e codigos
def generateHuffmanCodes(node:Node, code='', huffman_codes={}):
    if node is not None:
        if node.symbol is not None:
            huffman_codes[node.symbol] = code
        generateHuffmanCodes(node.left, code + '0', huffman_codes)
        generateHuffmanCodes(node.right, code + '1', huffman_codes)
    
    return huffman_codes

# recebe uma lista contendo cada bloco codificado em RLE blocos esses tambem representados por uma lista
def generateGlobalHuffmanTable(rle_blocks: list):
    # coletando todos os simbolos de todos os blocos
    all_symbols = []
    for block in rle_blocks:
        for symbol in block[::2]:
            if isinstance(symbol, tuple):
                all_symbols.append(str(symbol))

    # contabilizando frequencias e simbolos
    frequencies = Counter(all_symbols)

    # constroi a arvore de huffman e gera o dicionario com os codigos de huffman
    root = buildHuffmanTree(list(frequencies.keys()), list(frequencies.values()))
    huffman_codes = generateHuffmanCodes(root)

    return root, huffman_codes

# codifica os blocos usando a tabela de huffman e retorna uma string com caracteres apenas binario
def huffmanEncode(rle_blocks: list, huffman_codes: dict) -> str:

    code = ''

    for block in rle_blocks:
        for i in range(0, len(block), 2):
            symbol = block[i]
            code += f'{huffman_codes[str(symbol)]}'
            if symbol != (0, 0):
                value = block[i + 1]
                if value != 0:
                    if value > 0:
                        code += format(value, f'0{symbol[1]}b')
                    else: 
                        two_complement = (1 << symbol[1]) + value
                        code += format(two_complement, f'0{symbol[1]}b') 


    return code

# decodifica os blocos codificados em huffman deve receber uma string com caracteres apenas binarios e retorna uma lista com todos os blocos decodificados
# e as tabelas de quantização QTY e QTC
def huffmanDecode(code:str, huffman_codes: dict) -> tuple:
    # invertendo as chaves com os valores do dicionario de huffman
    huffman_inverted = {code: symbol for symbol, code in huffman_codes.items()}
    # indice para percorrer o codigo
    i = 0
    # buffer para leitura de bits no codigo
    buffer = ''
    # listas para a reconstrução da tabela de 
    qty, qtc = [], []
    # listas para reconstruir o codigo RLE original
    decoded = []
    block = []
    # conta quantos blocos ja foram decodificados
    decoded_blocks = 0

    while i < len(code):
        buffer += code[i]
        if buffer in huffman_inverted:
            # reconstruindo uma das tuplas do codigo original
            symbol = eval(huffman_inverted[buffer])
            # salvando a tupla no bloco reconstruido
            block.append(symbol)
            # identifica se chegou ao fim do bloco
            if symbol == (0,0):
                if decoded_blocks == 0:
                    qty.append(block)
                elif decoded_blocks == 1:
                    qtc.append(block)
                else:
                    decoded.append(block)
                buffer = ''
                block = []
                i += 1 + symbol[1]
                decoded_blocks += 1
                continue

            # recortando os bits do inteiro não nulo com base no tamanho indicado na tupla acima
            num = code[i+1:(i + 1) + symbol[1]]
            # convertendo de binario para inteiro (considerando complemento de 2 para numeros negativos)
            # caso o numero seja negativo
            if num[0] == '1':
                # quantidade de bits 
                bits = len(num)
                # convertendo para inteiro
                num = int(num, 2)
                # ajustando complemento de 2
                num = num - (1 << bits)
            else:
                # se for positivo apenas converte para inteiro 
                num = int(num, 2)

            # salva o valor não nulo no bloco reconstruido
            block.append(num)
            # limpando buffer de bits
            buffer = ''
            i += 1 + symbol[1]
        else:
            i += 1

    return decoded, qty[0], qtc[0]

# funções responsaveis por codificar a tabela de huffman de modo que possa ser decodificada antes de decodificar os blocos no codigo final
def encodeHuffmanTable(huffman_table: dict) -> str:
    encoded = ''
    for symbol, code in huffman_table.items():
        # Converte o símbolo (tupla) em uma string e depois para bytes
        symbol_str = str(symbol)  # Converte a tupla em string
        # Armazena o tamanho do símbolo como binário 
        symbol_length = len(symbol_str)
        symbol_length_binary = format(symbol_length, '016b')
        # Converte cada caractere da string do símbolo para binário (8 bits por caractere)
        symbol_binary = ''.join(format(ord(char), '08b') for char in symbol_str)
        # Armazena o comprimento do código de Huffman em 8 bits
        code_length = len(code)
        code_length_binary = format(code_length, '08b')
        # Concatena símbolo, comprimento do código e o código
        encoded += symbol_length_binary + symbol_binary + code_length_binary + code
    # Adiciona o marcador de fim da tabela
    end_marker = "1111111111111111"  # 16 bits de 1
    encoded += end_marker
    return encoded

def decodeHuffmanTable(encoded: str) -> tuple:
    # marcador de final de tabela 16bits de 1
    end_marker = "1111111111111111"
    huffman_table = {}
    offset = 0
    while offset < len(encoded):
        # Verifica se encontrou o marcador de fim
        if encoded[offset:offset + 16] == end_marker:
            offset += 16
            break  # Fim da decodificação dos dados
        # Lê o comprimento do símbolo (16 bits)
        symbol_length_binary = encoded[offset:offset + 16]
        symbol_length = int(symbol_length_binary, 2)
        offset += 16
        # Lê o símbolo em formato binário
        symbol_binary = encoded[offset:offset + (symbol_length * 8)]
        symbol_str = ''.join(chr(int(symbol_binary[i:i + 8], 2)) for i in range(0, len(symbol_binary), 8))
        offset += symbol_length * 8
        # Lê o comprimento do código (8 bits)
        code_length_binary = encoded[offset:offset + 8]
        code_length = int(code_length_binary, 2)
        offset += 8
        # Lê o código de Huffman
        code = encoded[offset:offset + code_length]
        offset += code_length
        huffman_table[symbol_str] = code
    return huffman_table, encoded[offset:]

# funções responsaveis por codificar o dicionario que contem os shapes originais da imagem e o shape depois do padding
# esta codificação é ligeiramente parecida com a codificação usada para codificar a tabela de huffman
def encodeShapes(shapes: dict) -> str:
    encoded = ''
    for key, value in shapes.items():
        # Serializa a chave do dicionário
        key_length = len(key)
        key_length_binary = format(key_length, '016b')  # 16 bits para o comprimento da chave
        key_binary = ''.join(format(ord(char), '08b') for char in key)  # Cada caractere em 8 bits
        encoded += key_length_binary + key_binary
        
        # Serializa a lista de tuplas
        list_length = len(value)
        list_length_binary = format(list_length, '08b')  # 8 bits para o comprimento da lista
        encoded += list_length_binary
        for tup in value:
            # Cada elemento da tupla (dois inteiros) em 32 bits
            for num in tup:
                encoded += format(num, '032b')  # Inteiros em 32 bits

    # Adiciona o marcador de fim da seção
    end_marker = "1111111111111111"  # 16 bits de 1
    encoded += end_marker
    return encoded

def decodeShapes(encoded: str) -> tuple:
    # marcador de final de tabela 16bits de 1
    end_marker = "1111111111111111"
    decoded = {}
    offset = 0
    while offset < len(encoded):
        # Verifica se encontrou o marcador de fim
        if encoded[offset:offset + 16] == end_marker:
            offset += 16
            break  # Fim da decodificação dos dados
        # Lê o comprimento da chave
        key_length_binary = encoded[offset:offset + 16]
        key_length = int(key_length_binary, 2)
        offset += 16
        # Lê a chave
        key_binary = encoded[offset:offset + (key_length * 8)]
        # reconstruindo a chave
        key = ''.join(chr(int(key_binary[i:i + 8], 2)) for i in range(0, len(key_binary), 8))
        offset += key_length * 8
        
        # Lê o comprimento da lista
        list_length_binary = encoded[offset:offset + 8]
        list_length = int(list_length_binary, 2)
        offset += 8
        # Lê os elementos da lista (tuplas)
        value = []
        for _ in range(list_length):
            tup = []
            for _ in range(2):  # Cada tupla tem 2 inteiros
                num_binary = encoded[offset:offset + 32]
                num = int(num_binary, 2)
                tup.append(num)
                offset += 32
            value.append(tuple(tup))
        decoded[key] = value
    return decoded, encoded[offset:]

# função responsavel por escrever o arquivo com o codigo da imagem comprimida
def writeFile(code:str, filename:str = 'compressed'):
    # ajustando extensão de arquivo
    filename = filename + '.gpeg'
    
    # salva o comprimento original da string de bits
    original_length = len(code)

    # verificando se a string é um multiplo de 8 ja que o arquivo sera gravado em bytes não em bits
    if original_length % 8 != 0:
        # adiciona um padding de 0s a direita para poder salvar em bytes
        padded_code = code.ljust((original_length + 7) // 8 * 8, '0')
    else:
        padded_code = code

    # converte a string em bytes
    code_bytes = int(padded_code, 2).to_bytes(len(padded_code) // 8, byteorder='big')

    # escreve o arquivo
    with open(filename, 'wb') as file:
        # salvando o comprimento original no inicio com 4 bytes (permite um arquivo de ate 4gb)
        file.write(original_length.to_bytes(4, byteorder='big'))
        # salvando restante dos dados
        file.write(code_bytes)

def readFile(filepath:str) -> str:
    
    # abre o arquivo
    with open(filepath, 'rb') as file:
        # le os 4 bytes que representam o tamanho original da string
        orignal_length = int.from_bytes(file.read(4), byteorder='big')
        # le o restante do arquivo
        read_bytes = file.read()

    # convertendo os bytes para bits novamente
    padded_code = ''.join(f'{byte:08b}' for byte in read_bytes)

    # retorna apenas a parte sem o padding se ele tiver sido necessario na escrita
    return padded_code[:orignal_length]
    
def encode(image, qty:np.ndarray = QTY, qtc:np.ndarray = QTC, ssv:int = 2, ssh:int = 2, factor:float = 1, outputname:str = 'compressed'):

    # verifica se a imagem fornecida foi um filepath ou uma imagem em array de numpy
    if isinstance(image, str):
        # abrindo imagem 
        img = Image.open(image)
    else:
        img = image

    print('Inciando compressão!')

    # convertendo para o espaço de cor YCrCb
    colorSpace = toYCrCb(img)

    # constantes ssv, ssh de sub amostragem vertical e horizontal, não representão literalmente o 4:a:b
    # quanto maior o valor mais informação descartada e pior o resultado final
    # os valores equivalentes para 4:2:2 são ssv = 2, ssh = 1 e para 4:2:0 são ssv = 2 e ssh = 2
    y, crSub, cbSub, alphaSub = subSampling(ssv,ssh,colorSpace)

    # fator de qualidade aplicado nas tabelas de quantização, quanto maior mais qualidade e quanto menor mais compressão
    # recomendo usar valores de 1 ate no maximo 100 (em 100 praticamente ja não a perdas)
    qty = np.round(qty / factor)
    qty[qty == 0] = 1
    qtc = np.round(qtc / factor)
    qtc[qtc == 0] = 1

    # comprimindo a imagem realizando diretamente a DCT, quantização e codificação em ZIG ZAG, retorna a string codificada
    encoded = compress(y, crSub, cbSub, alphaSub, qty, qtc, ssv, ssh)
    # Escreve o arquivo comprimido
    writeFile(encoded, outputname)

    print('Compressão finalizada!')

    return encoded

def decode(filepath:str = 'compressed.gpeg', saveFile:bool = False):

    print('Iniciando descompressão!')
    try:
        # le o arquivo
        encoded = readFile(filepath)
    except:
        encoded = filepath

    # decodifica o arquivo e ja reconstroi as alterções gerados por quantização e DCT
    y, cr, cb, alpha, ssv, ssh = deCompress(encoded)

    # reconstruindo os canais que foram aplicados sub amostragem
    decodedYCrCb = upSampling(y, cr, cb, alpha, ssv, ssh)

    # voltando a imagem para o espaço de cor RGB com canal alpha
    decoded = toRGB(decodedYCrCb)

    print('Descompressão finalizada!')

    if saveFile:
        Image.fromarray(decoded).save('compressed.png')

    return decoded

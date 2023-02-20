import cv2
import time

# Realizar Download dos modelos em https://github.com/isl-org/MiDaS/releases e salvar no diretório "models"
path_model = "models/"

# Leitura do modelo (pode-se utilizar o Large ou o Small). O modelo Large possui melhor acurácia, 
# mas a inferência é muito mais lenta que o Small.]

# model_name = "model-f6b98070.onnx"; # MiDaS v2.1 Large
model_name = "model-small.onnx"; # MiDaS v2.1 Small


# Carregando modelo no módulo DNN do OpenCV
model = cv2.dnn.readNet(path_model + model_name)


if (model.empty()):
    print("Não foi possível carregar o modelo! - cheque o caminho")

# Caso opte por utilizar a GPU, configurar backend e saída com as duas linhas abaixo
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
# Abrindo a Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
   
    # Leitura da imagem e da variável "success", que é um boleano (True se a imagem existir ou False).
    success, img = cap.read()
    # Atributos altura, largura e canais da imagem
    imgHeight, imgWidth, channels = img.shape

    # Inicializando variável tempo para cálculo de FPS
    start = time.time()

    # Convertendo BGR para RGB (por default, o OpenCV faz a leitura em BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if model_name == "model-f6b98070.onnx":
    # Criando blob para as imagens (transformações para ajustar os dados ao modelo)
    # MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        blob = cv2.dnn.blobFromImage(img, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)
    else:
    # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

    # Passando o dado transformado para o modelo
    model.setInput(blob)

    # Obtendo a resposta a partir do dado de entrada
    output = model.forward()
    
    output = output[0,:,:]
    # Ajustando imagem de saída para o mesmo tamanho da imagem de entrada
    output = cv2.resize(output, (imgWidth, imgHeight))

    # Normalizando os pixels da saída entre 0 e 255, um canal de cores e com o tipo equivalente ao numpy.uint8

    output = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # Coloração sintética do tipo Magma
    output = cv2.applyColorMap(output, cv2.COLORMAP_MAGMA)

    # Finalizando o tempo de processamento da imagem
    end = time.time()
    # Realizando cálculo de FPS para a predição do frame
    fps = 1 / (end-start)

    # Exibindo FPS na imagem
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Convertendo de RGB para BGR (padrão do OpenCV) para que a imagem seja apresentada em RGB pela webcam
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('image', img)
    cv2.imshow('Depth Map', output)

  
    # Comando para parar processamento ao clicar a tecla "q" em cima do vídeo gerado pela webcam
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
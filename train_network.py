# Este código é responsável por fazer o treinamento do classificador,
# o banco de imagem irá separar aleatoriamente as imagens para treinamento
# e teste, ao final desta etapa um modelo com os pesos de treinamento será
# gerando e poderá ser utilizado no teste da rede.

import matplotlib

matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet.LeNet import LeNet
from imutils import paths
import numpy as np
import random
import cv2
import os


# Configuração da épocas de treinamento, taxa de aprendizado incial, e tamanho do batch
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# inicializa a matriz de dados (imagens) e a que conterá os rótulos (classes)
data = []
labels = []

# Embaralha os caminhos para a seleção das imagens de treinamento e teste
imagePaths = sorted(list(paths.list_images("bmw10_custom/bmw10_ims/")))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    # abre a imagem, redimensiona e armazena na matriz data
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    image = img_to_array(image)
    data.append(image)

    # extrai as classes dos rótulos
    label = imagePath.split(os.path.sep)[-2]

    if label == "3":
        label = 0
    elif label == "4":
        label = 1
    elif label == "5":
        label = 2
    else:
        label = 3
    labels.append(label)

# Normaliza os dados da imagem no intervalo de 0 a 1
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Particiona os dados em 70% para treino e 30% para teste
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.30, random_state=42)

# converte as lables de inteiro para vetores
trainY = to_categorical(trainY, num_classes=4)
testY = to_categorical(testY, num_classes=4)

# Realiza o aumento da base de dados para treino, realizando várias operações de alteração das imagens
# como rotações, inversões e etc.
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Inicialização da arquiterura Letnet
print("compilando o  modelo ...")
model = LeNet.build(width=64, heigth=64, depth=3, classes=4)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# A função de perda utilizada é a categorical_crossentropy, pois estamos realizando
# uma operação com mais de duas classes, no caso de duas, utiliza-se a binary_crossentropy.
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Nesta etapa ocorre o treinamento da rede utilizando os dados gerados pelo aumento
# no imageDataGenerator, os dados de treinamento e teste e o número de épocas utilizadas
# para o treino.
print("Treinamento da Rede ....")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS, verbose=1)

# Salva o modelo treinado no disco para utilização no teste,
print("salvando o modelo treinado em disco ...")
model.save("modelCar.model")

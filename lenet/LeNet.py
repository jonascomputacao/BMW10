# Implementação da arquiterura letnet para redes neurais convolucionais (CNN)
#

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

# Esta classe representa a estrutura de funcionamento de uma letnet
class LeNet:
    # Atributos referente a informações da imagem e quatidade de classes a serem reconhecidas
    #
    # width e heigth: Largura e altura da imagem
    # depth: Número de canais que a imagem utiliza
    # classes: Quantidade de classes que serão reconhecidas
    @staticmethod
    def build(width, heigth, depth, classes):
        # Cria um modelo de camadas sequenciais
        model = Sequential()
        inputShape = (heigth, width, depth)

        # Verifica o formato da imagem, se o formato é channel first (usado por Theano)
        #  ou channel last (usado por TensorFlow)
        if K.image_data_format() == "channel_first":
            inputShape = (depth, heigth, width)  # update input shape

        # Primeira camada: conv -> ReLU -> pool
        #
        # Camada de convolução irá aprender 20 filtros convolucionais 5x5.
        # Em seguida, aplicação da função de ativação relu seguida por
        # uma operação de maxpooling aplicado utilizando passos de 2x2 pixels
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Segunda camada: conv -> relu -> pool
        #
        # Aprendizagem de 50 filtros convolucionais, o aumento deste valor é comum
        # quando mais aprofunda-se na arquitetura da rede
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Nivelamento dos dados em um conjunto de camadas totalmente conectadas
        # FC -> relu
        model.add(Flatten())
        model.add(Dense(500)) #500 valores
        model.add(Activation("relu"))

        # Nivelamento de dados igual ao número de classes a serem classificadas.
        # por fim o classificador softmax transforma estes valores em probabilidade
        # para cada uma das classes
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # Retorna a arquiterura da rede
        return model

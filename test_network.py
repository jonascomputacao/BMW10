# Este código é reponsável por fazer o teste de classificação, ou seja,
# receber uma imagem qualquer e classificá-la utilizando as informações
# geradas no treinamento
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2


# Carrega a imagem a ser classificada
name = "alpine_class3"
image = cv2.imread("test/"+name+".png")
orig = image.copy()

# Pré-processa a imagem para classificá-la
image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0 # normaliza os dados de 0 a 1
image = img_to_array(image)
image = np.expand_dims(image, axis=0) # adiciona uma dimensão extra para não dar problema em predict

# Carrega o modelo de classificador treinado
print("Carregando o modelo ...")
model = load_model("modelCar.model")

# Realiza a predição da imagem e retorna as probabilidade para cada uma das classes
(bmw0, bmw1, bmw2, ind) = model.predict(image)[0]

# A definição da classe é tomada pela classe que obtiver a maior probabilidade
if bmw0 > bmw1 and bmw0 > bmw2 and bmw0 > ind:
    label = "bmw 3"
    proba = bmw0
elif bmw1 > bmw0 and bmw1 > bmw2 and bmw1 > ind:
    label = "bmw 4"
    proba = bmw1
elif bmw2 > bmw0 and bmw2 > bmw1 and bmw2 > ind:
    label = "bmw 5"
    proba = bmw2
else:
    label = "indefindo"
    proba = ind

label = "{}: {:.2f}%".format(label, proba * 100)

# Desenha a label e a probabilidade desta na imagem
output = imutils.resize(orig, width=400) # redimensiona a imagem de saída
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

# Apresenta a imagem com o resultado
cv2.imshow("Resultado: imagem classificada", output)
cv2.imwrite("result/"+name+"_result.png",output)
cv2.waitKey(0)
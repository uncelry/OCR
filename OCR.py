import random
import cv2
import scipy.misc
from scipy import ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os.path


def prepare_image(image_file, out_size=28):
    """
    Функция разбикви изображения текста на буквы
    :param image_file: Путь к изображению с текстом
    :param out_size: Размер стороны изображений букв
    :return: Список с изображениями букв заданного размера
    """

    # Открываем изображение и делаем его черно-белым
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 35)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Получаем контуры на изображении
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    # Создаем список под финальные изображения букв
    letters = []

    # Суммируем кол-во всех расстояний между буквами
    gap_sum = 0

    # Запоминаем позицию и ширину всех отступов
    gap_info = list()

    # Масштабируем все контуры букв до изображений 28x28 пикселей и сохраняем в результирующий список
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)

        # Находим все контуры, которые являются контурами букв
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = img_erode[y:y + h, x:x + w]

            # Запоминаем информацию о контуре
            gap_info.append({'x': x, 'w': w})

            # Изменяем размер букв до квадрата 28x28
            size_max = max(w, h)

            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)

            if w > h:
                # Увеличиваем картинку вертикально
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Увеличиваем картинку горизонтально
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Меняем размер буквы до 58x58, применяем размытие по Гауссу с сигмой = 3
            tmp = cv2.resize(cv2.GaussianBlur(letter_square, (3, 3), cv2.BORDER_DEFAULT), (58, 58))

            # Масштабируем букву до 22x22 пикселей
            tmp = cv2.resize(tmp, (out_size - 6, out_size - 6))

            # Создаем полностью белую картинку
            bg = np.full((28, 28), 255)

            # Центрируем картинку
            bg[3:3 + 22, 3:3 + 22] = tmp

            # Добавляем в финальный массив
            letters.append((x, w, bg))

    # Сортируем список по возрастанию координаты X
    letters.sort(key=lambda x: x[0], reverse=False)
    gap_info.sort(key=lambda x: x['x'], reverse=False)

    gap_len = list()

    # Находим сумму расстояний между контурами
    for i in range(1, len(gap_info)):
        gap_len.append(gap_info[i]['x'] - (gap_info[i - 1]['x'] + gap_info[i - 1]['w']))

    gap_sum = sum(gap_len)

    # Находим среднее расстояний между контурами
    avr_gap = gap_sum / len(gap_info)

    # Позиция пробелов
    spaces_pos = list()

    # Ищем все отступы, которые больше среднего
    more_avr_amount = 0
    more_avr_sum = 0

    # Проходимся по всем контурам
    for i in range(len(gap_len)):

        # Если расстояние больше среднего, суммируем
        if gap_len[i] > avr_gap:
            more_avr_amount += 1
            more_avr_sum += gap_len[i]

    # Находим среднее среди самых больших отступов
    more_avr = more_avr_sum / more_avr_amount

    # Проходимся по всем контурам
    for i in range(len(gap_len)):

        spaces_pos.append(0)

        # Если находим расстояние больше среднего, то добавляем пробел в этом месте
        if gap_len[i] > more_avr:
            spaces_pos[i] = 1

    # cv2.imshow("Input", img)
    # cv2.imshow("Enlarged", img_erode)
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)

    # Создаем список под результат
    res_list = []

    # Добавляем в список изображения отдельных символов
    for img in letters:
        res_list.append(img[2])

    # Возвращаем результирующий список и позиции пробелов
    return res_list, spaces_pos


# Функция инициализации параметров
def init_params():

    # Инициализируем значения по умолчанию
    W1 = np.random.rand(52, 784) - .5
    b1 = np.random.rand(52, 1) - .5
    W2 = np.random.rand(26, 52) - .5
    b2 = np.random.rand(26, 1) - .5

    return W1, b1, W2, b2


# Нелинейная функция активации, ограничивающая Z нулём снизу
def ReLU(Z):

    # Если элемент Z[i] < 0, возвращаем 0. Иначе - неизмененный элемент
    return np.maximum(Z, 0)


# Функция нормализации
def softmax(Z):

    # Делим экспоненту по Z на сумму всех экспонент по Z
    A = np.exp(Z) / sum(np.exp(Z))

    return A


# Прямое распространение
def forward_prop(W1, b1, W2, b2, X):

    # Вычисляем и возвращаем значения параметров
    Z1 = W1.dot(X) + b1

    # На скрытом слое используем функцию активации (A - активированный слой)
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2

    # На выходном слое нормализуем значения
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


# Метод One Hot Encoding (каждой букве в дата-сете выставляем 1, которая будет показывать к какому классу она относится)
def one_hot(Y):

    # Создаем матрицу нулей
    one_hot_Y = np.zeros((Y.size, Y.max()))

    # Создаем её копию со всеми значениями меньше на 1 (т.к. теги букв начинаются с 1, а не с 0)
    Y_copy = np.array([i-1 for i in Y])

    # Заполняем матрицу 1 по алгоритму One Hot Encoding
    one_hot_Y[np.arange(Y.size), Y_copy] = 1

    # Транспонируем
    one_hot_Y = one_hot_Y.T

    return one_hot_Y


# Производная функции активации
def deriv_ReLU(Z):

    # Вернет 1, если Z > 0, и 0, если иначе. Так как первообразная при z > 0 - линейная функция
    return Z > 0


# Обратное распространение
def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):

    # Получаем матрицу One Hot Encoding
    one_hot_Y = one_hot(Y)

    # Погрешность второго слоя (выходного) = прогноз - реальные значения
    dZ2 = A2 - one_hot_Y

    # Погрешность весов 2-го слоя (производная)
    dW2 = 1 / m * dZ2.dot(A1.T)

    # Погрешность значений 2-го слоя (среднее из погрешностей)
    db2 = 1 / m * np.sum(dZ2)

    # Погрешность первого слоя (скрытого) = погрешность второго слоя с весами * на производную активационной ф-ии
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)

    # Погрешность весов 1-го слоя (производная)
    dW1 = 1 / m * dZ1.dot(X.T)

    # Погрешность значений 1-го слоя (среднее из погрешностей)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2


# Функция обновления параметров
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):

    # Просто обновляем параметры - веса и значения
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


# Получение прогнозов
def get_predictions(A2):

    # Получаем индекс максимального элемента вдоль 0-й оси (т.е. получаем ответ нейронной сети)
    return np.argmax(A2, 0)


# Получение точности
def get_accuracy(predictions, Y):

    # Составляем массив из прогнозов сети
    predictions = np.array([i + 1 for i in predictions])

    # Делим сумму верных значений на кол-во всех классов - получаем точность
    return np.sum(predictions == Y) / Y.size


# Градиентный спуск
def gradient_descent(X, Y, alpha, iterations):

    # Задаем начальные параметры
    W1, b1, W2, b2 = init_params()

    # Итеративно выполняем градиентный спуск
    for i in range(iterations):

        # Прямое распространение
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # Обратное распространение
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

        # Обновляем коэффициенты
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Каждые 10 итераций выводим информацию в консоль
        if i % 10 == 0:
            print("Итерация: ", i)
            predictions = get_predictions(A2)
            print("Текущая точность: ", get_accuracy(predictions, Y) * 100, '%', sep='')

    return W1, b1, W2, b2


# Градиентный спуск (условие окончания по времени)
def gradient_descent_max_time(X, Y, alpha, max_time_min):

    # Задаем начальные параметры
    W1, b1, W2, b2 = init_params()

    i = 0

    # Время
    timeout = time.time() + max_time_min * 60

    # Итеративно выполняем градиентный спуск
    while time.time() < timeout:

        # Прямое распространение
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # Обратное распространение
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

        # Обновляем коэффициенты
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Каждые 10 итераций выводим информацию в консоль
        if i % 10 == 0:
            print("Итерация: ", i)
            predictions = get_predictions(A2)
            print("Текущая точность: ", get_accuracy(predictions, Y) * 100, '%', sep='')

        i += 1

    return W1, b1, W2, b2


# Сделать прогноз
def make_predictions(X, W1, b1, W2, b2):

    # Прогоняем через сеть один экземпляр
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)

    # Получаем прогнозы сети
    predictions = get_predictions(A2)
    # print(A2)
    return predictions


# Протестировать прогноз (для проверки на базовом дата-сете)
def test_prediction(index, W1, b1, W2, b2):

    # Достаем нужную картинку из тестового дата-сета
    current_image = X_dev[:, index, None]

    # Получаем прогноз сети
    prediction = make_predictions(X_dev[:, index, None], W1, b1, W2, b2)

    # Получаем реальный ответ
    label = Y_dev[index]

    # Выводим информацию
    print("Прогноз: ", chr((prediction + 1)[0] + 96))
    print("Реальный ответ: ", chr(label + 96))

    # Показываем картинку для проверки
    current_image = current_image.reshape((28, 28)) * 255
    tr = scipy.ndimage.rotate(current_image, -90)
    tr = np.fliplr(tr)
    plt.gray()
    plt.imshow(tr, interpolation='nearest')
    plt.show()


# Сделать прогноз для реальной картинки
def neural_guess_image(image, W1, b1, W2, b2):

    # Передаем сети картинку, получаем прогноз
    prediction = make_predictions(image, W1, b1, W2, b2)

    # Выводим ответ
    print(chr((prediction + 1)[0] + 96))

    # Показываем картинку
    current_image = image.reshape((28, 28)) * 255
    tr = scipy.ndimage.rotate(current_image, -90)
    tr = np.fliplr(tr)
    plt.gray()
    plt.imshow(tr, interpolation='nearest')
    plt.show()

    # Возвращаем ответ
    return chr((prediction + 1)[0] + 96)


# Привести картинку к нужному виду для обработки нейронкой
def prepare_img_for_neural(image):

    # Поворачиваем изображение на 90 градусов и переварачиваем его
    tr = np.fliplr(image)
    tr = scipy.ndimage.rotate(tr, 90)

    # Объединяем двумерный массив в одномерный
    new_arr = [a for b in tr for a in b]

    # Инвертируем пиксели изображени
    corrected_img = list(map(lambda x: 255 - x, new_arr))

    # Транспонируем
    img_arr = np.array(corrected_img).reshape(784, 1) / 255

    return img_arr


# Главная функция
if __name__ == '__main__':
    letters_list, space_list = prepare_image("test_data/1.png")

    # Считываем дата сет для обучения
    col_names = ['label']
    col_names.extend(['pixel' + str(i) for i in range(784)])
    data = pd.read_csv('dataset/emnist-letters-train.csv', names=col_names, header=None)

    # Считываем дата сет для тестов
    data_test = pd.read_csv('dataset/emnist-letters-test.csv', names=col_names, header=None)

    # Записываем данные в массив numpy
    data = np.array(data)
    data_test = np.array(data_test)

    # Получаем размерность данных (настоящее кол-во столбцов меньше на 1)
    m, n = data.shape
    m1, n1 = data_test.shape

    # Перемешиваем данные
    # np.random.shuffle(data)

    # Разделяем данные на тестовые
    data_dev = data_test[0:m1].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n1]
    X_dev = X_dev / 255

    # И на тренировочные данные
    data_train = data[0:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255

    # Если все файлы с данными сети уже имеются, то пропускаем обучение
    if os.path.isfile("neural_coeffs/W1.csv") and \
            os.path.isfile("neural_coeffs/b1.csv") and \
            os.path.isfile("neural_coeffs/W2.csv") and \
            os.path.isfile("neural_coeffs/b2.csv"):

        # Считываем данные из файлов
        W1 = np.genfromtxt('neural_coeffs/W1.csv', delimiter=',')
        b1 = pd.read_csv('neural_coeffs/b1.csv', header=None)
        W2 = np.genfromtxt('neural_coeffs/W2.csv', delimiter=',')
        b2 = pd.read_csv('neural_coeffs/b2.csv', header=None)

        # Конвертируем данные в нужный формат
        b1 = b1.to_numpy()
        b2 = b2.to_numpy()

    # Если каких-то файлов нет, то обучаем нейронную сеть
    else:
        tic = time.perf_counter()

        # Запускаем метод градиентного спуска
        # W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.5, 2000)
        W1, b1, W2, b2 = gradient_descent_max_time(X_train, Y_train, 0.5, 60)

        toc = time.perf_counter()

        print(f"Обучение заняло {(toc - tic) / 60:0.2f} минут")

        # Записываем данные в файлы
        np.savetxt("neural_coeffs/W1.csv", W1, delimiter=",")
        np.savetxt("neural_coeffs/b1.csv", b1, delimiter=",")
        np.savetxt("neural_coeffs/W2.csv", W2, delimiter=",")
        np.savetxt("neural_coeffs/b2.csv", b2, delimiter=",")

        # От кол-ва нейронов на скрытом слое:
        #   26 нейронов (alpha = 0.25; iter = 2000) ~ 72% (12.12 минут)
        #   52 нейронов (alpha = 0.25; iter = 2000) ~ 77% (16 минут)
        #   104 нейронов (alpha = 0.25; iter = 2000) ~ 79% (25 минут)
        #   208 нейронов (alpha = 0.25; iter = 2000) ~ 80% (36 минут)
        #   416 нейронов (alpha = 0.25; iter = 2000) ~ 80.4% (66 минут)

        # От параметра насыщения:
        #   0.1 (52 нейрона, iter = 2000) ~ 71% (17 минут)
        #   0.5 (52 нейрона, iter = 2000) ~ 79% (18 минут)
        #   1.5 (52 нейрона, iter = 2000) ~ 77.5% (17.5 минут)
        #   3 (52 нейрона, iter = 2000) ~ 15% (17 минут)
        #   6 (52 нейрона, iter = 2000) ~ 5% (16 минут)

        # От времени обучения:
        #   3 минуты (52 нейрона, alpha = 0.5) ~ 65%
        #   5 минут (52 нейрона, alpha = 0.5) ~ 72%
        #   10 минут (52 нейрона, alpha = 0.5) ~ 77%
        #   30 минут (52 нейрона, alpha = 0.5) ~ 82%
        #   60 минут (52 нейрона, alpha = 0.5) ~ 83%


        # alpha = 0.1 iterations = 500 ~ 56% (2.5 минуты)
        # alpha = 0.2 iterations = 500 ~ 63% (3.13 минут)
        # alpha = 0.3 iterations = 500 ~ 66% (5 минут)
        # alpha = 0.2 iterations = 1500 ~ 74% (12.5 минут)
        # alpha = 0.2 iterations = 5000 ~ 79% (31.3 минут)
        # alpha = 0.2 iterations = 10000 ~ 82% (70  минут)

        # Данные при 52 нейронах на скрытом слое:
        # alpha = 0.2 iterations = 500 ~ 68% (4 минуты)
        # alpha = 0.25 iterations = 20000 ~ 89.5% (167 минут)
        # alpha = 0.25 iterations = 40000 ~ 91% (338 минут)


    # Передаем нейронной сети изображения букв на обработку (для теста)
    # test_prediction(10836, W1, b1, W2, b2)


    # Проверяем какая будет точность при обработке на dev дата-сете
    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    print('Точность на НЕ обучающем дата-сете: ', get_accuracy(dev_predictions, Y_dev) * 100, '%', sep='')


    # Передаем нейросети все изображения букв с целевой картинки
    # final_prediction = ''
    #
    # i = 0
    # for pict in letters_list:
    #     prepared_pic = prepare_img_for_neural(pict)
    #     result = neural_guess_image(prepared_pic, W1, b1, W2, b2)
    #     final_prediction += result
    #
    #     # Если здесь должен быть пробел, добавляем его
    #     if i < len(space_list):
    #         if space_list[i] == 1:
    #             final_prediction += " "
    #     i += 1
    #
    # print("Текст, распознанный нейронной сетью:", final_prediction)


    # Генерируем слово на НЕ обучающем дата-сете
    # final_prediction_1 = ''
    #
    # for i in range(random.randint(10, 20)):
    #     rand_num = random.randint(0, m1 - 1)
    #     tmp_pic = X_dev[:, rand_num, None]
    #     result = neural_guess_image(tmp_pic, W1, b1, W2, b2)
    #     final_prediction_1 += result
    #
    # print('Текст, распознанный нейронной сетью из НЕ обучаюшего дата-сета:', final_prediction_1)

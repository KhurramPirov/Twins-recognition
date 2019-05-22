## Запуск
Чтобы все запустить нужно поставить dlib. Для этого нужен C-make.

Для детектора лиц необходимы 2 файла. 
Один `mmod_human_face_detector.dat` (5мб)
(в репозитории и тут: https://drive.google.com/file/d/1AXrjHnNE1xmoaBJnOYmuasSu9j579_wi/view?usp=sharing)
Второй `shape_predictor_68_face_landmarks.dat` (95мб)
(Только тут: https://drive.google.com/file/d/1myGmKhk8tg1e-FXlp6cQw8AGZblTq1nn/view?usp=sharing)

Также необходим файл dlib_fd.py с классом. 
Возможно, потребуется обновить openCV, у меня от 4.3.0

## Работа
Пример работы в .ipynb ноутбуке. 
1. Инициализация класса. В качестве аргументов здесь пути к файлам .dat и флаг какой модуль использовать для поиска лица на картинке.
Если стоит True, то модель, основанная на CovnNet, если нет, то используется HOG-фильтры. Вторые более стабильные, но детектят обычно меньше лиц. 
`dfd = dlib_landmark_detector(cnn_flag='True', 
                             bb_detector_path = path+'/mmod_human_face_detector.dat', 
                             landmark_predictor_path = path+'/shape_predictor_68_face_landmarks.dat' )`

2. Подготовка изображения. На вход принимает 8битную картинку. Padding лучше делать медианой, но не нулями. Тогда челюсть располагается более менее адекватно. 
`# Convert image from batch 'float64' to 'uint8'
img_8bit = (255*img/img.max() ).round().astype('uint8')
img_8bit_pad = np.pad(img_8bit, ((0,40),(30,30),(0,0)), mode='median')
`

3. Использование детектора
`dfd.reset()'
dfd.face_detect(img_8bit_pad, visualize=True)
`
Перед запуском необходимо очистить значения, это функция `self.reset()`. В качестве параметра принимает картинку (см. выше) и флаги: писать подробный отчет (`verbose=True`) и показывать ли картинки лиц, задетектированных точек (`visualize=True`).

4. Чтобы получить координаты - функция get_original_size_landmarks
`# Finally, GET landmark coordinated as numpy array of [x, y] coords of each 68 point
img_shape = dfd.get_original_size_landmarks()`

5. Чтобы вырезать повернутое лицо (отцентрированное), используется функция 
`dlib.get_face_chip(dfd._rescaled_image.copy(), dfd.landmark_shape, size=256, padding=0.15)`
Здесь важно НЕ менять размер и padding для ВСЕХ лиц. Я подобрал padding=0.15, мне кажется, оптимально. 
В качестве параметров принимает внутренние переменные из детектора. Поэтому `dfd.reset()` делать строго перед запуском детектора.

## Workflow
Изображение -> Изображение uint8 -> dfd(img) -> face_crop -> dfd(face_crop) -> landmarks -> distance(shape1, shape2) -> plot


Чтобы все запустить нужно поставить dlib. Для этого нужен C-make.

Для детектора лиц необходимы 2 файла. 
Один `mmod_human_face_detector.dat` (5мб)
(в репозитории и тут: https://drive.google.com/file/d/1AXrjHnNE1xmoaBJnOYmuasSu9j579_wi/view?usp=sharing)
Второй `shape_predictor_68_face_landmarks.dat` (95мб)
(Только тут: https://drive.google.com/file/d/1myGmKhk8tg1e-FXlp6cQw8AGZblTq1nn/view?usp=sharing)

Также необходим файл dlib_fd.py с классом. 
Возможно, потребуется обновить openCV, у меня от 4.3.0

Пример работы в .ipynb ноутбуке. 

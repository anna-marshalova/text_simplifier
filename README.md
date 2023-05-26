# Упрощение текста

Сервис для автоматического упрощения текстов на русском языке. 
Проект в раках курса [My First Data Project](https://ai.itmo.ru/course) от университета ИТМО.

**Корпус**: [RuSimpleSentEval](https://github.com/dialogue-evaluation/RuSimpleSentEval) + [RuAdapt](https://github.com/Digital-Pushkin-Lab/RuAdapt).

**Бейзлайн:** берем несколько первых сиинтаксических уровней каждого текста, заменяем слова на более простые синонимы. 
Синонимы из [этого списка](https://github.com/Digital-Pushkin-Lab/RuAdapt_Word_Lists) + подбираются по текстам корпуса с помощью fastText. 
Код лежит в папке `baseline`, эксперименты - там же, в ноутбуке `baseline simplification.ipynb`.


**Модель:** t5 (несколько вариантов моделей). 
Код для обучения и инференса в папке `seq2seq`, эксперименты - там же, в ноутбуке `t5 for simplification.ipynb`.
Также провела эксперимент с созданием модели для усложнения текстов - код в ноутбуке `complication.ipynb`.

**Метрики**: [SARI](https://aclanthology.org/Q16-1029.pdf), [BLEU](http://aclanthology.lst.uni-saarland.de/P02-1040/), [FKGL](https://www.semanticscholar.org/paper/Derivation-of-New-Readability-Formulas-%28Automated-Kincaid-Fishburne/26d5981f7da4b508961aea01d53cd60e2202ff2d) (модифицированная для русского языка). 
Код для экспериментов и подсчета метрик - в папке `experiments`.
Чекпоинты метрик хранятся в файле с логами.

**Обертка**: cервис обернут в телеграм-бота, но доступен пока только локально. 
Код в папке `bot`. 

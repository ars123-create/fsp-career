# Источники, границы проверки и поиск аналогов

Обзор для плана от 7 октября 2026. Назначение — обосновать решения и отделить известные методы от собственных гипотез. Чтение страницы или README не равно воспроизведению результата, проверке безопасности либо интеграции.

## Материалы организаторов

- [Защищённая папка](https://mostech-cloud.mos.ru/s/w7JwETATCPHpZb3): ранее открыта в браузере с предоставленным пользователем доступом; скачаны два файла. Пароль в репозитории не сохраняется.
- [Исходное ТЗ, 9 страниц](../materials/ФСП%20спец.%20трек%20ТЗ.pdf). Извлечение текста повторно сверено с сохранённой копией; физические номера страниц использованы в матрице требований.
- [Шаблон презентации, 37 слайдов](../materials/ЛЦТ_2026_Шаблон%20презентации%20ФСП.pptx). Прочитаны инструкции и обязательные заготовки 7–11; остальные страницы являются набором примеров/макетов.
- Пост организаторов, переданный пользователем: дедлайн 11 октября 23:59 МСК; сохранить структуру/дизайн слайдов 7–11. Ссылку на отдельный публичный пост пользователь не предоставлял.

Нет предоставленных API/данных реестра, датасетов и сервера — это прямо следует из ТЗ, а не из неудачи поиска. Отдельный брендбук упомянут в ТЗ, но среди полученных файлов отсутствовал.

## Каркас и компоненты

| Источник | Глубина проверки | Для чего использован |
|---|---|---|
| [FastAPI template](https://github.com/fastapi/full-stack-fastapi-template) | README, лицензия MIT по репозиторию, отдельные исходники ниже; запуск NOT RUN | Предварительный выбор основы |
| [User models](https://raw.githubusercontent.com/fastapi/full-stack-fastapi-template/master/backend/app/models.py) | Прочитан код текущей master | Видны active/superuser, email; необходимы отдельные предметные роли и подтверждение |
| [User routes](https://raw.githubusercontent.com/fastapi/full-stack-fastapi-template/master/backend/app/api/routes/users.py) | Прочитан код, включая signup и редактирование email | Signup напрямую создаёт пользователя; готового требуемого потока подтверждения в этих маршрутах нет |
| [OpenATS](https://github.com/chamals3n4/OpenATS) | README/описание зависимостей и лицензии | Альтернативная основа; отклонена для данного объёма переделок |
| [FastAPI Users verification](https://fastapi-users.github.io/fastapi-users/latest/configuration/routers/verify/) | Документация маршрута | Готовый ориентир для подтверждения почты |
| [Mailpit](https://github.com/axllent/mailpit) | Описание проекта | Локальная проверка писем |
| [shadcn forms](https://ui.shadcn.com/docs/forms/tanstack-form) | Документация | Формы; конкретный выбор наследуем от каркаса |
| [pypdf](https://github.com/py-pdf/pypdf) | Документация/репозиторий, локальное извлечение текста ТЗ | Возможный импорт резюме, не оценка грейда |
| [WeasyPrint](https://github.com/Kozea/WeasyPrint) | README, BSD-3-Clause по репозиторию | Кандидат для стандартизированного PDF |
| [Hypothesis](https://github.com/HypothesisWorks/hypothesis) | Описание/документация проекта | Проверка генераторов заданий, пока не установлен |
| [catsim](https://github.com/douglasrizzo/catsim) | README, интерфейсы CAT, BSD-3-Clause по репозиторию | Ориентир и потенциальный независимый baseline, не запускался |
| [py-irt](https://github.com/nd-ball/py-irt) | README: модели, зависимости, MIT | Возможная дальнейшая калибровка при достаточных данных |
| [SciPy assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | Официальный API + выполненные проверки | Реально переиспользованный решатель; библиотечный алгоритм не объявляется авторским |
| [pgvector](https://github.com/pgvector/pgvector) | README | Отложенное расширение поиска; не установлен |
| [Sentence Transformers](https://github.com/huggingface/sentence-transformers) | README | Возможное сопоставление синонимов; модель и её лицензия не выбраны |
| [Judge0](https://github.com/judge0/judge0) | Описание проекта | Отложен вместе с исполнением произвольного кода |
| [Keycloak OIDC](https://www.keycloak.org/securing-apps/oidc-layers) | Официальная документация | План будущего адаптера внешней идентичности |
| [Playwright](https://github.com/microsoft/playwright), [Locust](https://github.com/locustio/locust) | Репозитории/описание возможностей | План будущих браузерных и нагрузочных проверок |
| [Matplotlib errorbar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html) | Официальный API; версия runtime 3.10.8 | Построение воспроизводимого научного графика из сохранённых чисел |

Точные версии внедряемых компонентов и файлы LICENSE фиксируются при изолированном запуске. Для ещё не внедрённых строк не придуманы SHA, результаты CI и сведения о production-эксплуатации.

## Научные первоисточники

1. Behpour, S., Liu, A., Ziebart, B. (2019). [Active Learning for Probabilistic Structured Prediction of Cuts and Matchings](https://proceedings.mlr.press/v97/behpour19a.html). ICML, PMLR 97:563–572. Изучены аннотация, вводная часть доступного PDF и [ссылка на авторский код](https://github.com/sima111b/ActiveStructuredPrediction). Полного воспроизведения нет. Это ближайший явный предшественник идеи активных запросов для сопоставления.
2. Yang, C. et al. (2024). [Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method](https://arxiv.org/abs/2408.09748). KDD 2024. Прочитаны аннотация/описание постановки и [репозиторий CRRS](https://github.com/RUCAIBox/CRRS). Нельзя называть наш совместимый по условиям подбор воспроизведением их причинной модели.
3. Zheng et al. (2023). [Reciprocal Sequential Recommendation](https://arxiv.org/abs/2306.14712). Прочитана аннотация; ориентир для сценариев с историей взаимодействий. Не выбран как база холодного старта.
4. McLeod, L. D., Lewis, C., Thissen, D. (2003). [A Bayesian Method for the Detection of Item Preknowledge in Computerized Adaptive Testing](https://www.ets.org/research/policy_research_reports/publications/article/2003/hjul.html). [Страница издателя](https://journals.sagepub.com/doi/10.1177/0146621602250534). Доступны карточка/аннотация; полный платный текст не получен. Утверждение ограничено тем, что моделирование предварительного знания уже исследовалось.
5. Jin et al. [Modeling Rapid Guessing Behaviors in Computer-Based Testlet Items](https://journals.sagepub.com/doi/10.1177/01466216221125177). Изучены доступная аннотация и описание; повторное открытие [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9679923/) встретило браузерную проверку. Не заявляем полный новый анализ статьи.
6. Jiang, S. et al. (2017). [Efficient Nonmyopic Active Search](https://proceedings.mlr.press/v70/jiang17d.html). ICML, PMLR 70:1714–1723. Прочитана аннотация. Использован как свидетельство известности планирования будущих запросов; его теоремы не переносятся на нашу постановку без доказательства.
7. Sharpnack, J. et al. (2026). [S2A3: Thompson Sampling and Stochastic Exposure Control for High-Stakes CATs](https://aclanthology.org/2026.aimecon-main.8/). AIME-Con, 76–82, карточка датирована октябрём 2026. Прочитана аннотация. Полный анализ современного метода остаётся пунктом последующей научной работы.
8. Michael, E., Wood, T. A., Manzie, C., Shames, I. (2020). [Global Sensitivity Analysis for the Linear Assignment Problem](https://arxiv.org/abs/2005.11792). Прочитана аннотация и описание вопроса об изменениях весов. Источник показывает, что анализ изменяемых строк/столбцов — известная область. Дополнительно прочитана часть раздела о вероятностной структурной модели в PDF Behpour et al.; полного воспроизведения по-прежнему нет.

## Поисковые направления

В ходе обзора проверялись сочетания запросов:

- computerized adaptive testing / item exposure / automatic item generation / hierarchical item response theory;
- reciprocal recommendation / recruitment / uncertainty / active learning;
- value of information / person-job / matching;
- decision-focused active learning / matching;
- adaptive testing / item generation / large language / calibration;
- testlet / item preknowledge / Bayesian model;
- recruitment / adaptive testing / matching / contamination;
- non-myopic / batch / value of information / Bayesian experimental design;
- family / preknowledge / reciprocal recommendation.
- value of information / matching / column; assignment problem / sensitivity analysis / column — перед точным кешированием расчётов.

Отбор для обоснований ограничен первоисточниками: статьи авторов, официальная документация и репозитории. Найдены близкие области и прямые предшественники; отсутствие в поиске точной копии нашей конструкции **не доказывает новизну**. Для статьи необходим полный обзор ближайших работ и сравнение постановок, а не только поиск совпадающего названия.

## Собственные доказательства

- [Протокол](../research/PROTOCOL.md) — записан до отложенного запуска; находится в списке хешируемых исходников.
- [Код](../research/core.py) — точная конечная модель; оптимизатор заимствован из установленного SciPy.
- [Проверки свойств](../research/test_core.py) — 6 тестов, включая аналитический результат и полный перебор.
- [Dev-результаты](../research/results/dev_v1.json) — не использовать как отложенные.
- [Отложенные результаты](../research/results/test_v1.json) — 2400 прогонов, полный trace, версии, хеши.
- [Автоматический отчёт](../research/results/test_v1_tables.md) — таблицы, без ручного выбора удачных запусков.
- [Протокол кеширования](../research/CACHE_PROTOCOL.md), [проверки эквивалентности](../research/test_fast_voi.py) — отдельная вычислительная проверка, не пересмотр качества test_v1.

Числа этого исследования подтверждают только поведение программы в описанных искусственных условиях. Они не являются измеренными свойствами ещё не созданного веб-приложения.

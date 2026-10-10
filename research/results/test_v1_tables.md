# Первая отложенная синтетическая проверка

Автоматически сформировано из test_v1.json. Ни одного реального кандидата.
Строк: 2400; рынков в каждом условии: 80.
Regret и Brier: меньше лучше. Ошибочные приглашения — число на рынок, не доля.
Distinct families — число уникальных троек кандидат/навык/семейство среди дополнительных ответов.

| Условие | Метод | Regret | Brier | Ошиб. приглашения | Все приглашения | Доп. ответы | Семейства | Время, с |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean | no_extra | 1.2509 | 0.1184 | 0.8125 | 3.3000 | 0.0000 | 0.0000 | 0.0001 |
| clean | random_naive | 0.9622 | 0.0988 | 0.5000 | 3.0625 | 12.0000 | 11.2625 | 0.0005 |
| clean | information_family | 0.5918 | 0.0784 | 0.2375 | 2.9625 | 12.0000 | 9.3625 | 0.0083 |
| clean | voi_naive | 0.4443 | 0.0831 | 0.1125 | 2.8625 | 12.0000 | 8.9125 | 0.0245 |
| clean | voi_family | 0.4443 | 0.0831 | 0.1125 | 2.8625 | 12.0000 | 8.9125 | 0.0246 |
| clean | pair_voi_family | 0.4088 | 0.0932 | 0.1375 | 2.9375 | 12.0000 | 10.2000 | 0.0610 |
| leakage | no_extra | 2.3403 | 0.1949 | 1.7000 | 3.7875 | 0.0000 | 0.0000 | 0.0001 |
| leakage | random_naive | 1.9178 | 0.1796 | 1.3250 | 3.5500 | 12.0000 | 11.2625 | 0.0005 |
| leakage | information_family | 1.3350 | 0.1354 | 0.5250 | 2.6500 | 12.0000 | 9.0750 | 0.0088 |
| leakage | voi_naive | 1.1007 | 0.1533 | 0.6750 | 3.2125 | 12.0000 | 8.8625 | 0.0263 |
| leakage | voi_family | 0.8459 | 0.1421 | 0.2875 | 2.7125 | 12.0000 | 10.5375 | 0.0259 |
| leakage | pair_voi_family | 0.8400 | 0.1427 | 0.3000 | 2.7500 | 12.0000 | 10.7750 | 0.0791 |
| misspecified_leakage | no_extra | 2.9237 | 0.2401 | 2.0625 | 3.8750 | 0.0000 | 0.0000 | 0.0001 |
| misspecified_leakage | random_naive | 2.4846 | 0.2277 | 1.7625 | 3.7500 | 12.0000 | 11.2625 | 0.0005 |
| misspecified_leakage | information_family | 2.3305 | 0.1905 | 1.6125 | 3.6375 | 12.0000 | 8.9875 | 0.0088 |
| misspecified_leakage | voi_naive | 1.9036 | 0.2110 | 1.2875 | 3.5000 | 12.0000 | 9.1625 | 0.0261 |
| misspecified_leakage | voi_family | 1.6391 | 0.1882 | 1.0125 | 3.2000 | 12.0000 | 10.3250 | 0.0261 |
| misspecified_leakage | pair_voi_family | 1.5770 | 0.1889 | 0.9875 | 3.2250 | 12.0000 | 10.5000 | 0.0788 |
| shifted_items | no_extra | 2.5968 | 0.2413 | 1.8000 | 3.6875 | 0.0000 | 0.0000 | 0.0001 |
| shifted_items | random_naive | 2.2539 | 0.2239 | 1.4125 | 3.3250 | 12.0000 | 11.2625 | 0.0005 |
| shifted_items | information_family | 1.7802 | 0.1914 | 0.6750 | 2.4125 | 12.0000 | 8.4500 | 0.0086 |
| shifted_items | voi_naive | 1.7180 | 0.2130 | 0.9000 | 2.9375 | 12.0000 | 8.4250 | 0.0256 |
| shifted_items | voi_family | 1.5781 | 0.1962 | 0.5375 | 2.3625 | 12.0000 | 9.9000 | 0.0252 |
| shifted_items | pair_voi_family | 1.6180 | 0.1980 | 0.5500 | 2.3500 | 12.0000 | 10.2000 | 0.0754 |
| correlated_skills | no_extra | 1.9391 | 0.1844 | 1.3375 | 3.7750 | 0.0000 | 0.0000 | 0.0001 |
| correlated_skills | random_naive | 1.7198 | 0.1722 | 1.1250 | 3.6500 | 12.0000 | 11.2625 | 0.0005 |
| correlated_skills | information_family | 1.5139 | 0.1331 | 0.6250 | 2.8750 | 12.0000 | 9.2500 | 0.0087 |
| correlated_skills | voi_naive | 1.0035 | 0.1535 | 0.6000 | 3.4125 | 12.0000 | 9.0125 | 0.0259 |
| correlated_skills | voi_family | 0.8715 | 0.1440 | 0.3500 | 3.0625 | 12.0000 | 10.9125 | 0.0256 |
| correlated_skills | pair_voi_family | 0.8528 | 0.1443 | 0.3625 | 3.1000 | 12.0000 | 11.0000 | 0.0788 |

## Парные описательные интервалы

Положительная разница: pair_voi_family имеет меньший regret. Без поправки на множественность.

| Условие | Сравнение с | Разница | Нижняя 95% | Верхняя 95% |
|---|---|---:|---:|---:|
| clean | no_extra | 0.8422 | 0.6400 | 1.0473 |
| clean | random_naive | 0.5534 | 0.3801 | 0.7329 |
| clean | information_family | 0.1830 | 0.0625 | 0.3118 |
| clean | voi_naive | 0.0356 | -0.0434 | 0.1168 |
| clean | voi_family | 0.0356 | -0.0415 | 0.1163 |
| leakage | no_extra | 1.5002 | 1.1996 | 1.8101 |
| leakage | random_naive | 1.0777 | 0.8113 | 1.3406 |
| leakage | information_family | 0.4950 | 0.2913 | 0.7080 |
| leakage | voi_naive | 0.2607 | 0.0596 | 0.4582 |
| leakage | voi_family | 0.0058 | -0.0473 | 0.0579 |
| misspecified_leakage | no_extra | 1.3467 | 1.0156 | 1.6867 |
| misspecified_leakage | random_naive | 0.9076 | 0.5910 | 1.2369 |
| misspecified_leakage | information_family | 0.7535 | 0.5036 | 1.0165 |
| misspecified_leakage | voi_naive | 0.3266 | -0.0191 | 0.6664 |
| misspecified_leakage | voi_family | 0.0621 | -0.0439 | 0.1839 |
| shifted_items | no_extra | 0.9788 | 0.6791 | 1.2870 |
| shifted_items | random_naive | 0.6359 | 0.3665 | 0.9119 |
| shifted_items | information_family | 0.1622 | -0.0790 | 0.4117 |
| shifted_items | voi_naive | 0.0999 | -0.0698 | 0.2751 |
| shifted_items | voi_family | -0.0399 | -0.1157 | 0.0356 |
| correlated_skills | no_extra | 1.0863 | 0.8023 | 1.3759 |
| correlated_skills | random_naive | 0.8670 | 0.6197 | 1.1303 |
| correlated_skills | information_family | 0.6611 | 0.4403 | 0.8852 |
| correlated_skills | voi_naive | 0.1507 | -0.0256 | 0.3385 |
| correlated_skills | voi_family | 0.0187 | -0.0508 | 0.0893 |

## Хеши исходников

- run_benchmark.py: 2c0b6a238004f4d9d902805b1424cd8de6e6c7d5e977d8f2c204961857a89132
- core.py: f812ccdae626a1e3b322b7ab044c7a2a65627de534cf282440c2a041cf665b6f
- PROTOCOL.md: ecdea073b231628e5a10bdea82827025c2d64f6d25a8cb9d77d8e24b262ca631

Версия Matplotlib для графика: 3.10.8.
Хеширование и проверки полноты результата выполнены при построении отчёта.

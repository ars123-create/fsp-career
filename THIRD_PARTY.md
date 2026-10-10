# Происхождение компонентов

Дата проверки: 7 октября 2026. Версии установки — backend/requirements.lock.txt и frontend/package-lock.json. Таблица ниже выделяет основные компоненты; полный перечень транзитивных пакетов — в lock-файлах.

| Компонент | Роль | Лицензия / происхождение |
|---|---|---|
| Full Stack FastAPI Template | Примеры Argon2/JWT, выбранные UI-примитивы | MIT, Sebastián Ramírez, commit e99fbaeba3b55190156ebae525ad9d481d76677e; licenses/fastapi-template-MIT.txt |
| shadcn/ui, Radix | Готовые React-компоненты, Dialog/Tabs/Button и др. | MIT, licenses/shadcn-ui-MIT.txt; код адаптирован из указанного каркаса |
| FastAPI, SQLModel, SQLAlchemy, Alembic | API, модели, транзакции, миграции | MIT; соответствующие лицензии в licenses/ |
| Pydantic 2.13.5 | Строгая схема ответов для отдельной предметной проверки | MIT; licenses/Pydantic-2.13.5-LICENSE.txt; библиотека не модифицируется |
| React, TanStack, Vite, Tailwind, Lucide, Sonner | Интерфейс и сборка | MIT, точные версии в package-lock.json; уведомления в установленных пакетах |
| ReportLab, pypdf | PDF экспорт/импорт | BSD; licenses/reportlab-0-LICENSE, licenses/pypdf-0-LICENSE |
| psycopg | Драйвер PostgreSQL | LGPL-3.0; неизменённая зависимость; licenses/psycopg-LICENSE.txt |
| pwdlib, PyJWT | Пароли и токены | MIT; licenses/ |
| DejaVu Sans | Встроенная кириллица PDF | licenses/DejaVu.txt |
| Manrope | Локальный веб-шрифт | SIL OFL 1.1; licenses/Manrope-OFL.txt |
| HTTPX | API-проверки и нагрузочный smoke | BSD-3-Clause; licenses/httpx-* |
| Hypothesis | Генеративные проверки эталонов | MPL-2.0; используется как неизменённая зависимость; licenses/hypothesis-0-LICENSE.txt |
| Python stdlib Decimal / sqlite3 | Числовая проверка и независимые SQL-эталоны | CPython 3.14.7, PSF-2.0; примеры документации PSF-2.0/0BSD; licenses/Python-3.14-LICENSE.txt. Интерпретатор не модифицируется |
| Prettier | Форматирование исходников | MIT; licenses/Prettier-MIT.txt |
| PostgreSQL / Mailpit | БД и локальный SMTP | PostgreSQL License / MIT; официальные контейнеры |
| NumPy / SciPy / HiGHS | Алгоритм устойчивой подборки и исследование | NumPy 2.4.4 / SciPy 1.18.1 (BSD-3-Clause), встроенный HiGHS 1.12.0 (MIT); licenses/SCIPY_LICENSE.txt, licenses/HIGHS_LICENSE.txt |
| Matplotlib | Только графики исследования | 3.10.8, Matplotlib license; не зависимость приложения |

Источники: https://github.com/fastapi/full-stack-fastapi-template ; https://github.com/shadcn-ui/ui ; https://github.com/axllent/mailpit ; https://www.postgresql.org/about/licence/ . Адресные пробы, адаптации и границы проверки — docs/IMPLEMENTATION_REUSE.md. READ в старом реестре не означает проверенный запуск.

Графика ФСП, логотипы и исходная презентация принадлежат соответствующим правообладателям и используются как предоставленные материалы трека. Не включаются в лицензию нашего кода. Сгенерированный дизайн сохранён отдельно от фактических скриншотов продукта.

Видеогид собран установленным FFmpeg 8.1.1 (сборка GPL v3 с libx264). Исполняемый FFmpeg не распространяется в проекте; [лицензионные сведения](https://ffmpeg.org/legal.html).
# Авторинг комплекта 0.4.4

python-docx 1.2.0 (MIT, Steve Canny) используется для редактируемой документации; лицензия: `licenses/python-docx-1.2.0-LICENSE.txt`. Он не добавлен в зависимости приложения. Bundled ReportLab 4.4.9 применяется для PDF слайдов; версия приложения закреплена отдельно.


## Optional product-film authoring (2026-10-10)

The separate authoring/fsp-product-film project uses Remotion 4.0.534, by Remotion AG, under its Free/Company License (not MIT); no Remotion runtime was added to the application. Exact notice: authoring/fsp-product-film/licenses/Remotion-LICENSE.md. Manrope font files retain their OFL notice alongside it. Source snapshots in public/screens were captured from fictional demo data in this application. Screen Studio and Arcade were editorial references only; their code and assets are not included.


## Live browser walkthrough (2026-10-10)

Separate authoring/fsp-live-demo uses Microsoft Playwright Core 1.64.0-alpha-1790635538000, Apache-2.0 (notice included alongside the recorder). Actual Chromium video is converted by installed FFmpeg 8.1.1. Geoscan recorder was inspected as a method reference; its code is not redistributed verbatim. This authoring dependency is not part of application runtime.

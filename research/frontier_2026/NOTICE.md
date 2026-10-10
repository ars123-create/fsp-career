# Авторство и лицензии исследовательского набора

В этом каталоге находятся собственные исследовательские адаптации и отдельно
сохранённые внешние компоненты. Существующая лицензия приложения не меняется.

- PABBO, Zhang/Huang/Kaski/Martinelli: AGPL-3.0, исходная лицензия и изменения
  в `research/pabbo_topk/LICENSE.txt` и `NOTICE.md`. Общие benchmark/registry и
  adapters, использующие этот компонент, не выдаются за полностью MIT-код.
- MoNNE, Hanyang Wang: MIT; pinned source/авторство сохранены в
  `components/monne/upstream/LICENSE` и `SOURCE.json`.
- SkewGP исходники: соответствующие upstream лицензии/хеши внутри
  `components/constrained_gp/`; наша SNIS адаптация прямо отделена от них.
- NumPy, SciPy, PyTorch, BoTorch сохраняют собственные лицензии. Runtime
  установлен ранее; package versions и source hashes входят в evidence.
- Lean/mathlib Apache-2.0 и лицензии зависимостей сохранены в `formal/`.

Копии `evidence/*/snapshot/` предназначены для воспроизведения конкретного
измерения и содержат исполняемые зависимости/веса этого измерения. Исходные
публикации, обучение, notices и аудит сохраняются также в основных research
каталогах. Snapshot не объявляется самостоятельно полным релизом приложения.

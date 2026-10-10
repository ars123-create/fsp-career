import { Button } from "@/components/ui/button";
import { WeightEditor } from "@/components/weight-editor";
import type { PreferenceAnswer, SearchResult } from "@/lib/types";

const percent = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 });

export function PreferencePanel({
  report,
  busy,
  hidden,
  onAnswer,
  onUndo,
  onReset,
  onHide,
  onWeights,
}: {
  report: NonNullable<SearchResult["preference"]>;
  busy: boolean;
  hidden: boolean;
  onAnswer: (choice: PreferenceAnswer["choice"]) => void;
  onUndo: () => void;
  onReset: () => void;
  onHide: (hidden: boolean) => void;
  onWeights: (weights: Record<string, number>) => void;
}) {
  const question = !hidden && report.question;
  const used = report.answered + report.skipped;
  const mixed = report.engine === "mixed-kg-v1";
  return (
    <section
      className="preference-panel"
      aria-label="Уточнение задачи"
      aria-busy={busy}
    >
      <div className="preference-heading">
        <div>
          <h2>
            {question
              ? "Что больше подходит вашей задаче?"
              : report.answered
                ? "Подборка с учётом ваших ответов"
                : "Подборка по результатам проверок"}
          </h2>
          <p>
            {question
              ? "Сравните два условных варианта при одинаковых остальных условиях. Проценты — результаты проверок навыков."
              : `Выбрано ${report.selected_count} из ${report.considered} подходящих специалистов.`}
          </p>
        </div>
        {question && (
          <span className="preference-step">Вопрос {used + 1} из 3</span>
        )}
      </div>
      {question && (
        <>
          <table
            className="preference-table"
            aria-label="Два сочетания результатов проверок"
          >
            <thead>
              <tr>
                <th scope="col">Навык</th>
                <th scope="col">Вариант А</th>
                <th scope="col">Вариант Б</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(question.left).map((skill) => (
                <tr key={skill}>
                  <th scope="row">{skill}</th>
                  {[question.left, question.right].map((variant, i) => (
                    <td key={i}>
                      <b>{percent.format(variant[skill])}%</b>
                      <div className="preference-track" aria-hidden="true">
                        <span style={{ width: `${variant[skill]}%` }} />
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <td />
                <td>
                  <Button
                    variant="outline"
                    disabled={busy}
                    onClick={() => onAnswer("left")}
                  >
                    Выбрать А
                  </Button>
                </td>
                <td>
                  <Button
                    variant="outline"
                    disabled={busy}
                    onClick={() => onAnswer("right")}
                  >
                    Выбрать Б
                  </Button>
                </td>
              </tr>
            </tfoot>
          </table>
          <div className="preference-actions">
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => onAnswer("skip")}
            >
              Пропустить вопрос
            </Button>
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => onHide(true)}
            >
              Посмотреть подборку
            </Button>
          </div>
        </>
      )}
      {!question && report.question && (
        <Button variant="outline" disabled={busy} onClick={() => onHide(false)}>
          Продолжить уточнение
        </Button>
      )}
      {!question && !report.question && report.status === "not_needed" && (
        <p className="preference-status">
          Дополнительный вопрос сейчас почти не меняет состав подборки.
        </p>
      )}
      {mixed ? (
        <p className="preference-status">
          {report.model}
        </p>
      ) : (
        <div className="preference-priorities">
          <span>
            {report.answered
              ? "Приоритеты по вашим ответам"
              : "Исходно навыки равнозначны"}
          </span>
          <div>
            {Object.entries(report.weights).map(([name, weight]) => (
              <span className="skill-tag" key={name}>
                {name} {Math.round(weight * 100)}%
              </span>
            ))}
          </div>
        </div>
      )}
      <div className="preference-status" role="status" aria-live="polite">
        {busy
          ? "Обновляем подборку…"
          : used
            ? `Учтено ответов: ${report.answered}. Пропущено: ${report.skipped}. По сравнению с исходной подборкой заменено профилей: ${report.changed_count}.`
            : `В подборке ${report.selected_count} из ${report.considered} подходящих профилей. Можно выбрать вариант или сразу перейти к специалистам.`}
      </div>
      {used > 0 && (
        <div className="preference-actions">
          <Button variant="ghost" size="sm" disabled={busy} onClick={onUndo}>
            Отменить последний ответ
          </Button>
          <Button variant="ghost" size="sm" disabled={busy} onClick={onReset}>
            Сбросить уточнения
          </Button>
        </div>
      )}
      {!mixed && (
        <div className="preference-actions">
          <WeightEditor
            weights={report.weights}
            busy={busy}
            onApply={onWeights}
            label="Задать приоритеты самому"
          />
        </div>
      )}
      <details className="preference-explanation">
        <summary>Как ответы влияют на подбор</summary>
        <p>
          Сначала применяются обязательные условия: специализация, грейд,
          формат, зарплатная вилка и проверенные навыки. Затем баллы проверок
          учитываются с приоритетами вашего запроса. Сравнения помогают уточнить
          эти приоритеты; пропуск ничего не добавляет. Достижения ФСП доступны в
          профиле и отдельном фильтре. Ответы не меняют грейд специалиста и не
          открывают его контакты.
        </p>
      </details>
    </section>
  );
}

import type { AssessmentAnswer, Question } from "@/lib/types";

export function QuestionTables({ question }: { question: Question }) {
  return question.tables?.map((table) => (
    <div
      className="assessment-table-scroll"
      key={table.name}
      tabIndex={0}
      role="region"
      aria-label={`Таблица ${table.name}`}
    >
      <table className="assessment-table">
        <caption>{table.name}</caption>
        <thead>
          <tr>
            {table.columns.map((column) => (
              <th scope="col" key={column}>
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i}>
              {row.map((value, j) => (
                <td key={j}>
                  {value === null ? (
                    <span className="muted">NULL</span>
                  ) : (
                    String(value)
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ));
}

export function AnswerText({
  question,
  value,
}: {
  question: Question;
  value?: AssessmentAnswer;
}) {
  if (question.answer_type === "fields") {
    return (
      <>
        {question.fields?.map((field, i) => (
          <span key={field.name}>
            {i > 0 ? "; " : ""}
            {field.label}:{" "}
            <b>
              {typeof value === "object" ? (value[field.name] ?? "—") : "—"}
            </b>
          </span>
        ))}
      </>
    );
  }
  return <>{typeof value === "number" ? question.choices[value] : "—"}</>;
}

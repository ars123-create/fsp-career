import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Field } from "@/components/common";

export function WeightEditor({
  weights,
  busy,
  onApply,
  label = "Изменить приоритеты",
}: {
  weights: Record<string, number>;
  busy: boolean;
  onApply: (weights: Record<string, number>) => void;
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const total = Object.values(values).reduce(
    (sum, value) => sum + Number(value),
    0,
  );
  function changeOpen(next: boolean) {
    if (next) {
      // The user reviews these rounded proposals before applying them.
      setValues(
        Object.fromEntries(
          Object.entries(weights).map(([skill, weight]) => [
            skill,
            String(Number((weight * 100).toFixed(2))),
          ]),
        ),
      );
      setError("");
    }
    setOpen(next);
  }
  function submit(event: FormEvent) {
    event.preventDefault();
    const entries = Object.entries(values).map(
      ([skill, value]) => [skill, Number(value)] as const,
    );
    if (
      !entries.length ||
      entries.some(
        ([skill, value]) =>
          values[skill] === "" ||
          !Number.isFinite(value) ||
          value < 0 ||
          value > 100,
      ) ||
      total <= 0
    ) {
      setError(
        "Укажите значения от 0 до 100. Хотя бы одно должно быть больше нуля.",
      );
      return;
    }
    onApply(Object.fromEntries(entries));
    setOpen(false);
  }
  return (
    <>
      <Button
        variant="outline"
        disabled={busy}
        onClick={() => changeOpen(true)}
      >
        {label}
      </Button>
      <Dialog open={open} onOpenChange={changeOpen}>
        <DialogContent className="form-dialog">
          <DialogHeader>
            <DialogTitle>Ваши приоритеты навыков</DialogTitle>
            <DialogDescription>
              Сравните пользу роста результата проверки с 50% до 100% при
              одинаковых остальных условиях. Важность 100 и 50 означает, что
              такой рост первого навыка для вас вдвое ценнее.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={submit} className="stack-form">
            <p className="tiny muted">
              Предложены округлённые текущие доли. Можно менять пропорции: сумма
              не обязана быть равна 100.
            </p>
            {Object.keys(weights).map((skill) => (
              <Field
                key={skill}
                label={`Важность: ${skill}`}
                hint={`Доля в расчёте: ${total > 0 && Number.isFinite(total) ? ((100 * Number(values[skill] || 0)) / total).toFixed(1) : "0"}%`}
              >
                <Input
                  type="number"
                  required
                  min={0}
                  max={100}
                  step="any"
                  value={values[skill] ?? ""}
                  onChange={(event) => {
                    setValues({ ...values, [skill]: event.target.value });
                    setError("");
                  }}
                />
              </Field>
            ))}
            <p className="tiny muted">
              Нулевой вес снижает влияние на порядок. Навык остаётся
              обязательным, грейд специалиста не меняется.
            </p>
            {error && (
              <p className="error-box" role="alert">
                {error}
              </p>
            )}
            <div className="preference-actions">
              <Button type="submit" disabled={busy}>
                Применить приоритеты
              </Button>
              <Button
                type="button"
                variant="ghost"
                onClick={() => {
                  setValues(
                    Object.fromEntries(
                      Object.keys(weights).map((skill) => [skill, "100"]),
                    ),
                  );
                  setError("");
                }}
              >
                Сделать равными
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}

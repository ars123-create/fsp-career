import { useEffect, useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDownToLine,
  ArrowRight,
  Bookmark,
  Check,
  ChevronDown,
  Clock3,
  Filter,
  LockKeyhole,
  MapPin,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Trophy,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PreferencePanel } from "@/components/preference-panel";
import { WeightEditor } from "@/components/weight-editor";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Busy,
  CheckField,
  Empty,
  ErrorBox,
  Field,
  Loading,
  PageTitle,
  Select,
  Verified,
} from "@/components/common";
import { api, download, post } from "@/lib/api";
import {
  type Candidate,
  type SearchInput,
  type SearchResult,
  type PreferenceAnswer,
  specs,
  formats,
  salary,
  date,
} from "@/lib/types";
import { toast } from "sonner";

const empty: SearchInput = {
  query: "",
  specialization: "",
  grade: "",
  work_format: "",
  skills: [],
  salary_min: 0,
  salary_max: 0,
  fsp_only: false,
  title: "Моя подборка",
  ranking: { mode: "mixed", priorities: [], comparisons: [] },
};

// Round guarantees upward so the displayed bound never understates the value.
const boundLabel = (value: number) => (Math.ceil(value * 10) / 10).toFixed(1);

export function SkillBars({ skills }: { skills: Candidate["skills"] }) {
  return (
    <div className="skill-bars">
      {Object.values(skills).some((s) => s.source === "family-latest-v1") && (
        <p className="tiny muted">
          Баллы учитывают последнее наблюдение по каждому семейству. Повтор
          одного семейства обновляет его результат и не увеличивает покрытие.
        </p>
      )}
      {Object.entries(skills).map(([name, value]) => (
        <div key={name}>
          <div className="skill-label">
            <span>{name}</span>
            <b>{value.score}%</b>
          </div>
          <div className="skill-track">
            <span style={{ width: `${value.score}%` }} />
          </div>
          <small>
            {value.families} семейств · {date(value.checked_at)}
            {value.latest_checked_at &&
            date(value.latest_checked_at) !== date(value.checked_at)
              ? ` — ${date(value.latest_checked_at)}`
              : ""}
          </small>
        </div>
      ))}
    </div>
  );
}

export function CandidateDetails({
  c,
  onInvite,
  close,
}: {
  c: Candidate;
  onInvite?: () => void;
  close?: () => void;
}) {
  return (
    <aside className="candidate-detail">
      <div className="detail-kicker">
        <span>ПРОФИЛЬ СПЕЦИАЛИСТА</span>
        {close && (
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Закрыть профиль"
            onClick={close}
          >
            <X />
          </Button>
        )}
      </div>
      <div className="detail-person">
        <div className="avatar large">
          {c.contacts
            ? c.name.slice(0, 1)
            : c.specialization === "backend"
              ? "Py"
              : "Da"}
        </div>
        <h2>{c.name}</h2>
        <p>{c.headline}</p>
        <div className="flex justify-center gap-2 flex-wrap">
          <span className="grade-tag">{c.grade}</span>
          {c.is_demo && <span className="demo-tag">Демо</span>}
        </div>
      </div>
      <div className="detail-section">
        <h3>
          Подтверждённые навыки <ShieldCheck size={17} />
        </h3>
        <SkillBars skills={c.skills} />
        <p className="tiny muted">
          Процент верных ответов в проверке. Не вероятность успешного найма.
        </p>
      </div>
      {c.reasons.length > 0 && (
        <div className="detail-section">
          <h3>Почему в подборке</h3>
          <ul className="reasons">
            {c.reasons.map((r) => (
              <li key={r}>
                <Check size={14} />
                {r}
              </li>
            ))}
          </ul>
          <p className="tiny muted">
            {c.match_score} из 100 баллов соответствия текущему запросу.
          </p>
        </div>
      )}
      <div className="detail-section">
        <h3>Об опыте</h3>
        <p>{c.about || "Описание не заполнено."}</p>
        <div className="meta-line">
          <MapPin size={14} />
          {c.city || "Город не указан"} · {formats[c.work_format]}
        </div>
        <p className="detail-salary">{salary(c.salary_min, c.salary_max)}</p>
        <div className="tags">
          {c.stack.map((s) => (
            <span key={s}>{s}</span>
          ))}
        </div>
      </div>
      <div className="detail-section">
        <h3>
          Достижения ФСП <Trophy size={16} />
        </h3>
        {c.achievements.length ? (
          c.achievements.map((a, i) => (
            <div className="achievement" key={i}>
              <Trophy size={18} />
              <div>
                <b>{a.title}</b>
                <small>
                  {a.year} · {a.source}
                </small>
              </div>
            </div>
          ))
        ) : (
          <p className="muted">
            Подтверждённых достижений пока нет. Они не обязательны для подбора.
          </p>
        )}
      </div>
      <div className="detail-section">
        <h3>
          {c.contacts ? "Контакты открыты" : "Контакты защищены"}{" "}
          <LockKeyhole size={16} />
        </h3>
        {c.contacts ? (
          <div className="contacts">
            <a href={`mailto:${c.contacts.email}`}>{c.contacts.email}</a>
            <p>{c.contacts.phone}</p>
            <p>{c.contacts.telegram}</p>
          </div>
        ) : (
          <p className="muted">
            Будут доступны после принятия вашего приглашения или отклика
            кандидата.
          </p>
        )}
        {onInvite && (
          <Button className="w-full mt-4" onClick={onInvite}>
            <Send size={16} />
            Пригласить в команду
          </Button>
        )}
        <Button
          className="w-full mt-2"
          variant="outline"
          onClick={() =>
            download(`/candidates/${c.id}/pdf`, "fsp-profile.pdf").catch((e) =>
              toast.error(e.message),
            )
          }
        >
          <ArrowDownToLine size={16} />
          Скачать профиль PDF
        </Button>
      </div>
    </aside>
  );
}

export function InviteDialog({
  candidate,
  onClose,
}: {
  candidate: Candidate | null;
  onClose: () => void;
}) {
  const company = useQuery({
    queryKey: ["company"],
    queryFn: () => api("/company"),
    enabled: !!candidate,
  });
  const qc = useQueryClient();
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [contact, setContact] = useState("");
  useEffect(() => {
    if (company.data) setContact(company.data.contact);
    setError(null);
  }, [company.data, candidate?.id]);
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const d = new FormData(e.currentTarget);
    setBusy(true);
    setError(null);
    try {
      await post("/invitations", {
        candidate_id: candidate!.id,
        title: d.get("title"),
        description: d.get("description"),
        salary_min: Number(d.get("salary_min")),
        salary_max: Number(d.get("salary_max")),
        currency: "RUB",
        contact,
      });
      qc.invalidateQueries({ queryKey: ["invitations"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
      toast.success("Приглашение отправлено кандидату");
      onClose();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog
      open={!!candidate}
      onOpenChange={(v) => {
        if (!v) onClose();
      }}
    >
      <DialogContent className="modal-scroll">
        <DialogHeader>
          <DialogTitle>Пригласить в команду</DialogTitle>
          <DialogDescription>
            {candidate?.name} · Укажите условия до знакомства.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={submit} className="form-stack">
          <Field label="Роль в команде">
            <Input
              name="title"
              required
              minLength={3}
              maxLength={150}
              defaultValue={
                candidate ? `${candidate.headline} · ${candidate.grade}` : ""
              }
            />
          </Field>
          <Field label="Команда, задачи и условия">
            <textarea
              name="description"
              required
              minLength={10}
              maxLength={5000}
              rows={4}
              placeholder="Что предстоит делать, с кем работать и какие условия вы предлагаете"
            />
          </Field>
          <div className="form-grid">
            <Field label="Зарплата от, ₽ / месяц">
              <Input
                name="salary_min"
                type="number"
                min={1}
                max={10000000}
                defaultValue={candidate?.salary_min || 100000}
                required
              />
            </Field>
            <Field label="До, ₽ / месяц">
              <Input
                name="salary_max"
                type="number"
                min={1}
                max={10000000}
                defaultValue={candidate?.salary_max || 180000}
                required
              />
            </Field>
          </div>
          <Field label="Как с вами связаться">
            <Input
              value={contact}
              onChange={(e) => setContact(e.target.value)}
              required
              minLength={3}
              maxLength={200}
            />
          </Field>
          <p className="tiny muted">
            Предложение действует 14 дней. Контакты кандидата откроются вашей
            компании только после принятия. Опубликованная вакансия не
            требуется.
          </p>
          <ErrorBox error={error || company.error} />
          <Busy type="submit" busy={busy}>
            <Send />
            Отправить приглашение
          </Busy>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function SearchPage() {
  const [draft, setDraft] = useState<SearchInput>({
    ...empty,
    specialization: "backend",
    grade: "Middle",
  });
  const [applied, setApplied] = useState<SearchInput>({
    ...empty,
    specialization: "backend",
    grade: "Middle",
  });
  const [selected, setSelected] = useState("");
  const [invite, setInvite] = useState<Candidate | null>(null);
  const [mobileDetail, setMobileDetail] = useState(false);
  const [filters, setFilters] = useState(false);
  const [savedOpen, setSavedOpen] = useState(false);
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveTitle, setSaveTitle] = useState("");
  const [previous, setPrevious] = useState<Candidate[] | null>(null);
  const [previousLabel, setPreviousLabel] = useState("");
  const [operationError, setOperationError] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const [questionHidden, setQuestionHidden] = useState(false);
  const query = useQuery<SearchResult>({
    queryKey: ["search", applied],
    queryFn: () => post("/search", applied),
    placeholderData: (previousData) => previousData,
  });
  const saved = useQuery<any[]>({
    queryKey: ["needs"],
    queryFn: () => api("/needs"),
  });
  const qc = useQueryClient();
  useEffect(() => {
    if (query.data) setDraft(query.data.normalized);
  }, [query.data]);
  const candidates = previous ?? query.data?.candidates ?? [];
  const candidate = candidates.find((c) => c.id === selected) || candidates[0];
  function set<K extends keyof SearchInput>(k: K, v: SearchInput[K]) {
    setDraft((d) => ({
      ...d,
      [k]: v,
      ...(k === "specialization" ? { skills: [], query: "" } : {}),
      ...([
        "skills",
        "specialization",
        "grade",
        "work_format",
        "salary_min",
        "salary_max",
        "fsp_only",
      ].includes(k)
        ? {
            ranking: {
              ...d.ranking,
              priorities: [],
              comparisons: [],
              weights: ["skills", "specialization"].includes(k)
                ? {}
                : d.ranking.weights,
            },
          }
        : {}),
    }));
  }
  function submit(e?: FormEvent) {
    e?.preventDefault();
    setPrevious(null);
    setApplied({ ...draft });
    setFilters(false);
    setQuestionHidden(false);
  }
  function prefer(first: string, second: string) {
    if (!query.data) return;
    const q = query.data.normalized;
    setApplied({
      ...q,
      ranking: {
        mode: "robust",
        priorities: [...q.ranking.priorities, [first, second]],
        comparisons: [],
      },
    });
    setPrevious(null);
    setQuestionHidden(false);
  }
  function applyComparisons(comparisons: PreferenceAnswer[]) {
    if (!query.data || query.isFetching) return;
    setApplied({
      ...query.data.normalized,
      ranking: {
        mode:
          query.data.normalized.ranking.mode === "mixed"
            ? "mixed"
            : "preference",
        priorities: [],
        comparisons,
      },
    });
    setPrevious(null);
    setQuestionHidden(false);
  }
  function answerPreference(choice: PreferenceAnswer["choice"]) {
    const question = query.data?.preference?.question;
    if (!question || !query.data) return;
    applyComparisons([
      ...query.data.normalized.ranking.comparisons,
      {
        left: question.left,
        right: question.right,
        choice,
      },
    ]);
  }
  function applyWeights(weights: Record<string, number>) {
    if (!query.data || query.isFetching) return;
    setApplied({
      ...query.data.normalized,
      ranking: { mode: "weighted", priorities: [], comparisons: [], weights },
    });
    setPrevious(null);
    setQuestionHidden(false);
  }
  async function save(e: FormEvent) {
    e.preventDefault();
    setSaving(true);
    setOperationError(null);
    try {
      await post("/needs", {
        ...query.data?.normalized,
        title: saveTitle || "Моя подборка",
      });
      await qc.invalidateQueries({ queryKey: ["needs"] });
      setSaveOpen(false);
      toast.success("Подборка и её состав сохранены");
    } catch (e) {
      setOperationError(e);
    } finally {
      setSaving(false);
    }
  }
  async function restore(id: string) {
    setOperationError(null);
    try {
      const r = await api(`/needs/${id}`);
      setDraft(r.current.normalized);
      setApplied(r.current.normalized);
      setPrevious(r.previous);
      setPreviousLabel(r.need.title);
      setSavedOpen(false);
    } catch (e) {
      setOperationError(e);
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="ПОИСК ПО ПОДТВЕРЖДЁННЫМ НАВЫКАМ"
        title="Найдите своего специалиста"
        description="Задайте потребность команды. Посмотрите, чем подтверждено совпадение."
        action={
          <Button variant="outline" onClick={() => setSavedOpen(true)}>
            <Bookmark />
            Мои подборки{" "}
            <span className="count-small">{saved.data?.length || 0}</span>
          </Button>
        }
      />
      <form
        onSubmit={submit}
        className="search-form"
        aria-busy={query.isFetching}
      >
        <fieldset disabled={query.isFetching} className="contents">
          <div className="search-input">
            <Search size={20} />
            <Input
              aria-label="Потребность команды"
              value={draft.query}
              onChange={(e) =>
                setDraft({
                  ...empty,
                  query: e.target.value,
                  ranking: {
                    mode: draft.ranking.mode,
                    priorities: [],
                    comparisons: [],
                  },
                })
              }
              placeholder="Например: Python backend, SQL, Middle, удалённо, 180–250 тыс."
            />
            <Button type="submit" size="lg">
              Найти специалистов <ArrowRight />
            </Button>
          </div>
          <div className="filter-strip">
            <Button
              variant="outline"
              type="button"
              onClick={() => setFilters(!filters)}
            >
              <Filter />
              Фильтры
              <ChevronDown size={14} />
            </Button>
            <Select
              aria-label="Специализация"
              value={draft.specialization}
              onChange={(e) => set("specialization", e.target.value)}
            >
              <option value="">Все специализации</option>
              {Object.entries(specs).map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </Select>
            <Select
              aria-label="Грейд"
              value={draft.grade}
              onChange={(e) => set("grade", e.target.value)}
            >
              <option value="">Любой грейд</option>
              <option>Junior</option>
              <option>Middle</option>
            </Select>
            <Select
              aria-label="Формат работы"
              value={draft.work_format}
              onChange={(e) => set("work_format", e.target.value)}
            >
              <option value="">Любой формат</option>
              {Object.entries(formats).map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </Select>
            <CheckField
              checked={draft.fsp_only}
              onChange={(v) => set("fsp_only", v)}
            >
              С достижениями ФСП
            </CheckField>
            <Select
              aria-label="Способ подбора"
              value={draft.ranking.mode}
              onChange={(e) =>
                set("ranking", {
                  mode: e.target.value as SearchInput["ranking"]["mode"],
                  priorities: [],
                  comparisons: [],
                })
              }
            >
              <option value="mixed">Умный подбор с уточнениями</option>
              <option value="evidence">По результатам проверок</option>
              <option value="preference">Подбор с уточнениями</option>
              <option value="weighted">Задать приоритеты самому</option>
              <option value="robust">Устойчивая пятёрка</option>
            </Select>
          </div>
          {filters && (
            <div className="expanded-filters">
              <Field
                label="Обязательные проверенные навыки"
                hint="Python, SQL, API, Аналитика, Качество данных"
              >
                <Input
                  value={draft.skills.join(", ")}
                  onChange={(e) =>
                    set(
                      "skills",
                      e.target.value
                        .split(",")
                        .map((s) => s.trim())
                        .filter(Boolean),
                    )
                  }
                />
              </Field>
              <Field label="Предложение от, ₽">
                <Input
                  type="number"
                  min={0}
                  value={draft.salary_min || ""}
                  onChange={(e) => set("salary_min", Number(e.target.value))}
                />
              </Field>
              <Field label="До, ₽">
                <Input
                  type="number"
                  min={0}
                  value={draft.salary_max || ""}
                  onChange={(e) => set("salary_max", Number(e.target.value))}
                />
              </Field>
              <Button type="submit">Применить</Button>
            </div>
          )}
        </fieldset>
      </form>
      <ErrorBox error={query.error || operationError} />
      {query.data?.notices.length ? (
        <p className="query-notice">
          <Sparkles size={14} />
          {query.data.notices.join(" ")}
        </p>
      ) : null}
      {!previous &&
        query.data?.preference &&
        query.data.preference.considered > 0 && (
          <PreferencePanel
            report={query.data.preference}
            busy={query.isFetching}
            hidden={questionHidden}
            onAnswer={answerPreference}
            onUndo={() =>
              applyComparisons(
                query.data!.normalized.ranking.comparisons.slice(0, -1),
              )
            }
            onReset={() => applyComparisons([])}
            onHide={setQuestionHidden}
            onWeights={applyWeights}
          />
        )}
      {!previous && query.data?.weighted && (
        <section
          className="preference-panel"
          aria-label="Подбор по вашим приоритетам"
          aria-busy={query.isFetching}
        >
          <div className="preference-heading">
            <div>
              <h2>Подборка по вашим приоритетам</h2>
              <p>
                Выбрано {query.data.weighted.selected_count} из{" "}
                {query.data.weighted.considered} подходящих специалистов.
                Важность навыков задаёте вы.
              </p>
            </div>
          </div>
          <div className="preference-priorities">
            <span>Доли в расчёте результата проверок</span>
            <div>
              {Object.entries(query.data.weighted.weights).map(
                ([skill, weight]) => (
                  <span className="skill-tag" key={skill}>
                    {skill}{" "}
                    {(weight * 100).toLocaleString("ru-RU", {
                      maximumFractionDigits: 1,
                    })}
                    %
                  </span>
                ),
              )}
            </div>
          </div>
          <p className="preference-status" role="status">
            {query.isFetching
              ? "Обновляем подборку…"
              : `По сравнению с равными приоритетами заменено профилей: ${query.data.weighted.changed_count}.`}
          </p>
          <div className="preference-actions">
            <WeightEditor
              weights={query.data.weighted.weights}
              busy={query.isFetching}
              onApply={applyWeights}
            />
            <Button
              variant="ghost"
              disabled={query.isFetching}
              onClick={() => applyComparisons([])}
            >
              Уточнить через вопросы
            </Button>
          </div>
          <details className="preference-explanation">
            <summary>Как устроен расчёт</summary>
            <p>
              Сначала проверяются специализация, грейд, зарплатная вилка, формат
              и обязательные навыки. Затем результаты проверок умножаются на
              заданные вами доли и складываются. При равенстве учитывается
              подтверждённое достижение ФСП. Веса не меняют грейд и доступ к
              контактам; результат не является вероятностью успешного найма.
            </p>
          </details>
        </section>
      )}
      {!previous && query.data?.robust && (
        <section
          className="robust-panel"
          aria-label="Устойчивость подборки"
          aria-busy={query.isFetching}
        >
          <div className="robust-heading">
            <div>
              <p className="eyebrow">ПРИОРИТЕТЫ МОГУТ ИЗМЕНИТЬСЯ</p>
              <h2>Подбор при разных приоритетах</h2>
              <p>
                Выбрано {query.data.robust.selected_count} из{" "}
                {query.data.robust.considered} подходящих профилей.
              </p>
            </div>
            <ShieldCheck size={26} aria-hidden="true" />
          </div>
          <div className="robust-comparison">
            <div>
              <span>Обычная пятёрка · худший проигрыш</span>
              <strong>
                {boundLabel(query.data.robust.baseline_worst_loss_points)}
                <small>балла / профиль</small>
              </strong>
            </div>
            <div>
              <span>Эта пятёрка · худший проигрыш</span>
              <strong>
                {boundLabel(query.data.robust.worst_loss_points)}
                <small>балла / профиль</small>
              </strong>
            </div>
          </div>
          <p className="tiny muted">
            Проигрыш лучшей подборке при допустимых приоритетах навыков, в
            среднем на профиль. Исходная средняя оценка снижена на{" "}
            {query.data.robust.nominal_loss_points.toFixed(2)} балла из
            разрешённого 1. Это сравнение наблюдаемых результатов проверок, не
            прогноз найма.
          </p>
          {query.data.robust.status !== "optimal" && (
            <p className="query-notice">
              Лимит расчёта: допустимая подборка найдена; оптимальность не
              подтверждена. Запас до нижней границы решателя:{" "}
              {query.data.robust.optimization_gap_points.toFixed(2)} балла /
              профиль.
            </p>
          )}
          {!!query.data.normalized.ranking.priorities.length && (
            <div className="robust-priorities">
              {query.data.normalized.ranking.priorities.map(([a, b]) => (
                <span className="grade-tag" key={`${a}-${b}`}>
                  {a} ≥ {b}
                </span>
              ))}
              <Button
                size="sm"
                variant="ghost"
                disabled={query.isFetching}
                onClick={() => {
                  setApplied({
                    ...query.data!.normalized,
                    ranking: {
                      mode: "robust",
                      priorities: [],
                      comparisons: [],
                    },
                  });
                  setQuestionHidden(false);
                }}
              >
                Сбросить приоритеты
              </Button>
            </div>
          )}
          {query.data.robust.question && !questionHidden && (
            <div className="robust-question">
              <h3>Что важнее для этой роли?</h3>
              <p>
                После ответа худший проигрыш — не более{" "}
                {boundLabel(query.data.robust.question.worst_after_points)}{" "}
                балла / профиль в заданной модели.
              </p>
              <div className="flex gap-2 flex-wrap">
                {query.data.robust.question.skills.map(
                  (skill, index, skills) => (
                    <Button
                      key={skill}
                      variant="outline"
                      disabled={query.isFetching}
                      onClick={() => prefer(skill, skills[1 - index])}
                    >
                      {skill} важнее
                    </Button>
                  ),
                )}
                <Button
                  variant="ghost"
                  disabled={query.isFetching}
                  onClick={() => setQuestionHidden(true)}
                >
                  Пока не знаю
                </Button>
              </div>
            </div>
          )}
          <details className="robust-explanation">
            <summary>Как получена граница</summary>
            <p>
              Мы проверяем крайние допустимые приоритеты. Самый неблагоприятный
              для этой подборки:{" "}
              {Object.entries(query.data.robust.worst_weights)
                .map(([skill, w]) => `${skill} ${Math.round(w * 100)}%`)
                .join(", ")}
              . Граница относится к составу подборки; порядок внутри — по
              исходной оценке. Проверки грейда и доступ к контактам не зависят
              от этого расчёта.
            </p>
          </details>
        </section>
      )}
      <div className="results-heading">
        <h2>
          {previous
            ? `Сохранённая подборка: ${previousLabel}`
            : "Подходящие специалисты"}{" "}
          <span>
            {previous ? candidates.length : (query.data?.total ?? "…")}
          </span>
        </h2>
        <div className="flex gap-2">
          {previous && (
            <Button variant="outline" onClick={() => setPrevious(null)}>
              Обновить состав
            </Button>
          )}
          <Button
            variant="ghost"
            disabled={!query.data || query.isFetching}
            onClick={() => {
              setSaveTitle(draft.query.slice(0, 60));
              setSaveOpen(true);
            }}
          >
            <Bookmark size={16} />
            Сохранить подборку
          </Button>
        </div>
      </div>
      {previous && (
        <p className="tiny muted mb-4">
          Исходный состав сохранён. Скрытые кандидатом профили исключены;
          текущие данные и права проверены заново.
        </p>
      )}
      {query.isPending ? (
        <Loading />
      ) : !candidates.length ? (
        <Empty
          title="Пока нет точного совпадения"
          text="Попробуйте расширить грейд, формат работы или убрать навык, для которого ещё нет проверки."
          action={
            <Button
              variant="outline"
              onClick={() => {
                setDraft({ ...empty });
                setApplied({ ...empty });
                setPrevious(null);
              }}
            >
              Сбросить фильтры
            </Button>
          }
        />
      ) : (
        <div className="search-layout">
          <div className="candidate-list">
            {candidates.map((c) => (
              <article
                key={c.id}
                className={`candidate-row ${candidate?.id === c.id ? "selected" : ""}`}
              >
                <button
                  type="button"
                  className="candidate-main"
                  onClick={() => {
                    setSelected(c.id);
                    if (window.matchMedia("(max-width:1000px)").matches)
                      setMobileDetail(true);
                  }}
                  aria-label={`Открыть ${c.name}`}
                >
                  <div
                    className={`avatar ${c.specialization === "data" ? "avatar-mint" : ""}`}
                  >
                    {c.specialization === "backend" ? "Py" : "Da"}
                  </div>
                  <div className="candidate-summary">
                    <div className="candidate-heading">
                      <h3>{c.headline}</h3>
                      <span className="grade-tag">{c.grade}</span>
                    </div>
                    <p className="candidate-name">
                      {c.name} {c.is_demo && <span>· демо</span>}
                    </p>
                    <p className="meta-line">
                      <MapPin size={13} />
                      {c.city} <span>·</span> {formats[c.work_format]}{" "}
                      <span>·</span> {c.experience_years} г. опыта
                    </p>
                    <div className="tags">
                      {Object.keys(c.skills).map((s) => (
                        <span className="skill-tag" key={s}>
                          <Check size={11} />
                          {s}
                        </span>
                      ))}
                      {c.achievements.length > 0 && (
                        <span className="fsp-tag">
                          <Trophy size={11} />
                          ФСП · демо
                        </span>
                      )}
                    </div>
                  </div>
                  <span className="match-number">
                    {c.match_score || "—"}
                    <small>баллов</small>
                  </span>
                </button>
                <div className="candidate-bottom">
                  <div>
                    <Verified />
                    <span className="salary-label">
                      {salary(c.salary_min, c.salary_max)}
                    </span>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setInvite(c)}
                  >
                    Пригласить <ArrowRight size={14} />
                  </Button>
                </div>
              </article>
            ))}
            <p className="list-footnote">
              <ShieldCheck size={16} />
              Ранжирование опирается на задания и источники достижений. Условия
              можно обсудить в приглашении.
            </p>
          </div>
          {candidate && (
            <div className="desktop-detail">
              <CandidateDetails
                c={candidate}
                onInvite={() => setInvite(candidate)}
              />
            </div>
          )}
        </div>
      )}
      <Dialog open={mobileDetail} onOpenChange={setMobileDetail}>
        <DialogContent className="modal-scroll">
          <DialogHeader>
            <DialogTitle>Профиль специалиста</DialogTitle>
            <DialogDescription>Подтверждения, опыт и условия</DialogDescription>
          </DialogHeader>
          {candidate && (
            <CandidateDetails
              c={candidate}
              onInvite={() => {
                setMobileDetail(false);
                setInvite(candidate);
              }}
            />
          )}
        </DialogContent>
      </Dialog>
      <InviteDialog candidate={invite} onClose={() => setInvite(null)} />
      <Dialog open={savedOpen} onOpenChange={setSavedOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Мои подборки</DialogTitle>
            <DialogDescription>
              Сохранённые запросы и состав выдачи на момент сохранения.
            </DialogDescription>
          </DialogHeader>
          {saved.data?.length ? (
            <div className="saved-list">
              {saved.data.map((n) => (
                <button key={n.id} onClick={() => restore(n.id)}>
                  <Bookmark size={18} />
                  <div>
                    <b>{n.title}</b>
                    <small>
                      {n.saved_ids.length} профилей · {date(n.created_at)}
                    </small>
                  </div>
                  <ArrowRight size={17} />
                </button>
              ))}
            </div>
          ) : (
            <p className="muted">
              Сохраните первый результат поиска, чтобы вернуться к нему позже.
            </p>
          )}
          <ErrorBox error={saved.error} />
        </DialogContent>
      </Dialog>
      <Dialog open={saveOpen} onOpenChange={setSaveOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Сохранить подборку</DialogTitle>
            <DialogDescription>
              Сохраним фильтры и текущий состав кандидатов.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={save} className="form-stack">
            <Field label="Название">
              <Input
                autoFocus
                value={saveTitle}
                onChange={(e) => setSaveTitle(e.target.value)}
                required
                maxLength={120}
                placeholder="Backend для команды платформы"
              />
            </Field>
            <ErrorBox error={operationError} />
            <Busy type="submit" busy={saving}>
              Сохранить
            </Busy>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}

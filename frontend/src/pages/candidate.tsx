import { useEffect, useState, type FormEvent } from "react";
import { useForm } from "react-hook-form";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  ArrowDownToLine,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Code2,
  FileUp,
  Link2,
  LockKeyhole,
  Play,
  RotateCcw,
  Save,
  ShieldCheck,
  Trophy,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  StepLink,
} from "@/components/common";
import { api, download, post, put } from "@/lib/api";
import {
  type AssessmentAnswer,
  type AssessmentCatalog,
  type Attempt,
  date,
  specs,
  formats,
  statuses,
} from "@/lib/types";
import { AnswerText, QuestionTables } from "@/components/assessment-content";
import { useAuth } from "@/lib/auth";
import { SkillBars } from "./search";
import { MailNotice } from "./auth";

export function Dashboard() {
  const { user } = useAuth();
  const q = useQuery({
    queryKey: ["dashboard"],
    queryFn: () => api("/dashboard"),
  });
  if (q.isPending) return <Loading />;
  if (q.error) return <ErrorBox error={q.error} />;
  const d = q.data;
  const candidate = user?.role === "candidate";
  return (
    <>
      <PageTitle
        eyebrow={candidate ? "ВАША КАРЬЕРА" : "ВАША КОМАНДА"}
        title={
          candidate ? "Навыки открывают двери" : "Встречайте будущих коллег"
        }
        description={
          candidate
            ? "Проверьте профиль, подтвердите навыки и выбирайте предложения."
            : "Находите специалистов по проверенным навыкам и обсуждайте конкретные условия."
        }
      />
      <div className="overview-metrics">
        <div>
          <span>
            {candidate ? "Текущая категория" : "Подтверждённых профилей"}
          </span>
          <strong>
            {candidate ? d.grade || "Не подтверждена" : d.candidates}
          </strong>
          <small>
            {candidate
              ? specs[d.specialization] || "Начните с первой проверки"
              : "Доступны для подбора"}
          </small>
        </div>
        <div>
          <span>Ожидают ответа</span>
          <strong>{d.pending}</strong>
          <small>Актуальные приглашения</small>
        </div>
        <div>
          <span>Приглашения приняты</span>
          <strong>{d.accepted}</strong>
          <small>Можно продолжить общение</small>
        </div>
      </div>
      <div className="overview-columns">
        <section className="panel">
          <p className="eyebrow">СЛЕДУЮЩИЙ ШАГ</p>
          <h2>
            {candidate
              ? "Профиль, которому доверяют"
              : "От потребности до знакомства"}
          </h2>
          {candidate ? (
            <>
              <StepLink
                to="/profile"
                n="01"
                title="Заполните профиль"
                text={
                  d.published
                    ? "Профиль опубликован. Предпочтения можно изменить."
                    : "Расскажите об опыте и включите публикацию."
                }
                done={d.published}
              />
              <StepLink
                to="/assessments"
                n="02"
                title="Подтвердите категорию"
                text={
                  d.grade
                    ? "Есть подтверждение. Можно добровольно проверить другой уровень."
                    : "12 заданий по выбранной специализации и грейду."
                }
                done={!!d.grade}
              />
              <StepLink
                to="/invitations"
                n="03"
                title="Рассмотрите предложения"
                text="Вилка, задачи и компания — до передачи контактов."
              />
            </>
          ) : (
            <>
              <StepLink
                to="/company"
                n="01"
                title="Представьте компанию"
                text="Опишите команду и оставьте способ связи."
              />
              <StepLink
                to="/search"
                n="02"
                title="Найдите специалиста"
                text="Фильтры и понятные основания каждой рекомендации."
              />
              <StepLink
                to="/invitations"
                n="03"
                title="Начните разговор"
                text="После принятия приглашения контакты станут доступны."
              />
            </>
          )}
        </section>
        <section className="panel purple-panel">
          <ShieldCheck size={34} />
          <h2>
            {candidate
              ? "Ваши контакты — ваш выбор"
              : "За каждым баллом — основание"}
          </h2>
          <p>
            {candidate
              ? "Работодатель увидит контакты только после вашего согласия. ФСП ID не нужен, чтобы начать искать работу."
              : "В профиле видны проверенные навыки, число семейств задач и дата оценки. Заявленный опыт показан отдельно."}
          </p>
          {candidate && d.skills && Object.keys(d.skills).length > 0 ? (
            <SkillBars skills={d.skills} />
          ) : (
            <div className="principle-list">
              <span>
                <Check />
                Прозрачные условия
              </span>
              <span>
                <Check />
                Подтверждённая категория
              </span>
              <span>
                <Check />
                Объяснимый выбор
              </span>
            </div>
          )}
          <Button variant="outline" asChild>
            <Link to={candidate ? "/invitations" : "/search"}>
              {candidate ? "К предложениям" : "Начать подбор"}
              <ArrowRight />
            </Link>
          </Button>
        </section>
      </div>
    </>
  );
}

const profileFields = [
  "full_name",
  "headline",
  "about",
  "city",
  "industry",
  "phone",
  "telegram",
  "experience_years",
  "declared_specialization",
  "declared_grade",
  "work_format",
  "employment",
  "salary_min",
  "salary_max",
  "published",
  "processing_consent",
  "publication_consent",
];
export function ProfilePage() {
  const q = useQuery({ queryKey: ["profile"], queryFn: () => api("/profile") });
  const qc = useQueryClient();
  const { register, reset, handleSubmit, watch, setValue } =
    useForm<Record<string, any>>();
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [imported, setImported] = useState<any>(null);
  const [importBusy, setImportBusy] = useState(false);
  const [fsp, setFsp] = useState("");
  useEffect(() => {
    if (q.data) {
      reset({
        ...q.data,
        stack_text: q.data.stack.join(", "),
        soft_text: q.data.soft_skills.join(", "),
      });
      setFsp(q.data.fsp_id);
    }
  }, [q.data, reset]);
  async function save(v: Record<string, any>) {
    setBusy(true);
    setError(null);
    const data = Object.fromEntries(profileFields.map((k) => [k, v[k]]));
    Object.assign(data, {
      stack: (v.stack_text || "")
        .split(",")
        .map((s: string) => s.trim())
        .filter(Boolean),
      soft_skills: (v.soft_text || "")
        .split(",")
        .map((s: string) => s.trim())
        .filter(Boolean),
    });
    try {
      await put("/profile", data);
      await qc.invalidateQueries({ queryKey: ["profile"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
      toast.success("Профиль сохранён");
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  async function upload(file?: File) {
    if (!file) return;
    setImportBusy(true);
    setError(null);
    try {
      const data = new FormData();
      data.append("file", file);
      setImported(
        await api("/profile/import-pdf", { method: "POST", body: data }),
      );
    } catch (e) {
      setError(e);
    } finally {
      setImportBusy(false);
    }
  }
  async function link(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await post("/profile/fsp", { fsp_id: fsp });
      toast.success(r.message);
      qc.invalidateQueries({ queryKey: ["profile"] });
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  if (q.isPending) return <Loading />;
  if (q.error) return <ErrorBox error={q.error} />;
  return (
    <>
      <PageTitle
        eyebrow="ЛИЧНЫЙ ПРОФИЛЬ"
        title="Расскажите о себе"
        description="Опыт помогает познакомиться. Категорию подтверждает отдельная проверка."
        action={
          <Button
            variant="outline"
            onClick={() =>
              download("/profile/pdf", "my-fsp-profile.pdf").catch((e) =>
                toast.error(e.message),
              )
            }
          >
            <ArrowDownToLine />
            Профиль PDF
          </Button>
        }
      />
      <div className="profile-layout">
        <div>
          <form className="panel form-stack" onSubmit={handleSubmit(save)}>
            <div className="section-heading">
              <h2>Опыт и предпочтения</h2>
              <label className="upload-control">
                <FileUp size={16} />
                {importBusy ? "Читаем PDF…" : "Импорт PDF"}
                <input
                  disabled={importBusy}
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => {
                    upload(e.target.files?.[0]);
                    e.target.value = "";
                  }}
                />
              </label>
            </div>
            <Field
              label="Имя и фамилия"
              hint="До согласия работодатели видят только код кандидата."
            >
              <Input
                required
                minLength={2}
                maxLength={100}
                {...register("full_name")}
              />
            </Field>
            <Field label="Заголовок профиля">
              <Input
                maxLength={120}
                placeholder="Python-разработчик"
                {...register("headline")}
              />
            </Field>
            <Field
              label="О себе и своём опыте"
              hint="Контакты и ссылки укажите в закрытых полях ниже."
            >
              <textarea rows={4} maxLength={2000} {...register("about")} />
            </Field>
            <div className="form-grid">
              <Field label="Город">
                <Input maxLength={80} {...register("city")} />
              </Field>
              <Field label="Отрасль">
                <Input maxLength={100} {...register("industry")} />
              </Field>
              <Field label="Специализация">
                <Select {...register("declared_specialization")}>
                  {Object.entries(specs).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Предполагаемый грейд">
                <Select {...register("declared_grade")}>
                  <option>Junior</option>
                  <option>Middle</option>
                </Select>
              </Field>
              <Field label="Опыт, лет">
                <Input
                  type="number"
                  min={0}
                  max={60}
                  {...register("experience_years", { valueAsNumber: true })}
                />
              </Field>
              <Field label="Формат работы">
                <Select {...register("work_format")}>
                  {Object.entries(formats).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Занятость">
                <Select {...register("employment")}>
                  <option value="full_time">Полная</option>
                  <option value="part_time">Частичная</option>
                  <option value="project">Проектная</option>
                </Select>
              </Field>
            </div>
            <Field
              label="Технологии через запятую"
              hint="Это ваш заявленный стек, он не присваивает подтверждённую категорию."
            >
              <Input
                {...register("stack_text")}
                placeholder="Python, SQL, Docker"
              />
            </Field>
            <Field label="Гибкие навыки">
              <Input
                {...register("soft_text")}
                placeholder="Командная работа, презентация решений"
              />
            </Field>
            <div className="form-grid">
              <Field label="Зарплата от, ₽ / месяц">
                <Input
                  type="number"
                  min={0}
                  max={10000000}
                  {...register("salary_min", { valueAsNumber: true })}
                />
              </Field>
              <Field label="До, ₽ / месяц">
                <Input
                  type="number"
                  min={0}
                  max={10000000}
                  {...register("salary_max", { valueAsNumber: true })}
                />
              </Field>
            </div>
            <hr />
            <h2>
              <LockKeyhole size={18} />
              Закрытые контакты
            </h2>
            <div className="form-grid">
              <Field label="Телефон">
                <Input type="tel" maxLength={40} {...register("phone")} />
              </Field>
              <Field label="Мессенджер">
                <Input maxLength={80} {...register("telegram")} />
              </Field>
            </div>
            <p className="tiny muted">
              Email: {q.data.email}. Эти данные открываются только компании с
              вашим разрешением.
            </p>
            <hr />
            <h2>Публикация и согласия</h2>
            <CheckField
              checked={!!watch("processing_consent")}
              onChange={(v) => setValue("processing_consent", v)}
            >
              Разрешаю обработку данных для работы платформы.
            </CheckField>
            <CheckField
              checked={!!watch("publication_consent")}
              onChange={(v) => setValue("publication_consent", v)}
            >
              Разрешаю показывать обезличенный профиль работодателям.
            </CheckField>
            <CheckField
              checked={!!watch("published")}
              onChange={(v) => setValue("published", v)}
            >
              Опубликовать профиль в поиске.
            </CheckField>
            {!q.data.grade && (
              <p className="notice-inline">
                Профиль появится в подборе после первой успешной проверки
                категории. На вакансии можно откликаться уже сейчас.
              </p>
            )}
            <ErrorBox error={error} />
            <div className="form-actions">
              <Busy type="submit" busy={busy}>
                <Save />
                Сохранить профиль
              </Busy>
              <Link to="/privacy" className="text-link">
                Правила приватности
              </Link>
            </div>
          </form>
        </div>
        <div className="profile-side">
          <section className="panel">
            <p className="eyebrow">ПОДТВЕРЖДЁННАЯ КАТЕГОРИЯ</p>
            <h2>{q.data.grade || "Пока нет категории"}</h2>
            <p>{specs[q.data.specialization] || "Пройдите первую проверку"}</p>
            {q.data.grade && (
              <p className="tiny muted">
                Смена грейда доступна с {date(q.data.next_grade_change_at)}.
                Неудачная попытка не понижает уровень.
              </p>
            )}
            <Button asChild className="w-full">
              <Link to="/assessments">
                <ShieldCheck />
                Проверить навыки
              </Link>
            </Button>
          </section>
          <form onSubmit={link} className="panel form-stack">
            <h2>
              <Trophy size={19} />
              История ФСП
            </h2>
            <p className="muted">
              ФСП ID не обязателен. Его ввод сохраняет заявку на связь с
              официальным реестром.
            </p>
            <Field label="ФСП ID">
              <Input
                value={fsp}
                onChange={(e) => setFsp(e.target.value)}
                maxLength={100}
                placeholder="Ваш идентификатор"
              />
            </Field>
            <p className="tiny">
              Статус:{" "}
              {q.data.fsp_status === "demo"
                ? "демонстрационный источник"
                : q.data.fsp_status === "pending"
                  ? "ожидает подключения реестра"
                  : "не связан"}
            </p>
            <Busy type="submit" busy={busy} variant="outline">
              <Link2 />
              {fsp ? "Сохранить заявку" : "Убрать связь"}
            </Busy>
            {q.data.achievements.map((a: any, i: number) => (
              <p className="tiny muted" key={i}>
                {a.title} · {a.source}
              </p>
            ))}
          </form>
        </div>
      </div>
      <Dialog
        open={!!imported}
        onOpenChange={(v) => {
          if (!v) setImported(null);
        }}
      >
        <DialogContent className="modal-scroll">
          <DialogHeader>
            <DialogTitle>Проверьте импорт</DialogTitle>
            <DialogDescription>
              Изменения попадут в форму; затем нажмите «Сохранить профиль».
            </DialogDescription>
          </DialogHeader>
          <p>{imported?.message}</p>
          <pre className="import-preview">{imported?.preview}</pre>
          <p>Телефон: {imported?.suggested.phone || "не найден"}</p>
          <p>Навыки: {imported?.suggested.stack.join(", ") || "не найдены"}</p>
          <Button
            onClick={() => {
              if (imported.suggested.phone)
                setValue("phone", imported.suggested.phone);
              if (imported.suggested.stack.length)
                setValue("stack_text", imported.suggested.stack.join(", "));
              setImported(null);
              toast.success("Поля перенесены в форму. Проверьте и сохраните.");
            }}
          >
            Перенести найденные поля
          </Button>
        </DialogContent>
      </Dialog>
    </>
  );
}

export function AssessmentPage() {
  const catalog = useQuery<AssessmentCatalog>({
    queryKey: ["assessment-catalog"],
    queryFn: () => api("/assessments/catalog"),
  });
  const history = useQuery<Attempt[]>({
    queryKey: ["assessments"],
    queryFn: () => api("/assessments"),
  });
  const profile = useQuery({
    queryKey: ["profile"],
    queryFn: () => api("/profile"),
  });
  const [specialization, setSpec] = useState("backend");
  const [grade, setGrade] = useState("Junior");
  const [active, setActive] = useState<Attempt | null>(null);
  const [fieldDrafts, setFieldDrafts] = useState<
    Record<string, Record<string, string>>
  >({});
  const [index, setIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [tick, setTick] = useState(Date.now());
  const qc = useQueryClient();
  useEffect(() => {
    const t = setInterval(() => setTick(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (profile.data) {
      setSpec(profile.data.declared_specialization);
      setGrade(profile.data.declared_grade);
    }
  }, [profile.data]);
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["assessments"] });
    qc.invalidateQueries({ queryKey: ["profile"] });
    qc.invalidateQueries({ queryKey: ["dashboard"] });
  };
  async function start(kind = "grade") {
    setBusy(true);
    setError(null);
    try {
      const a = await post<Attempt>("/assessments", {
        specialization:
          kind === "challenge" ? profile.data.specialization : specialization,
        grade: kind === "challenge" ? profile.data.grade : grade,
        kind,
      });
      setActive(a);
      setFieldDrafts({});
      setTick(Date.now());
      setIndex(0);
      refresh();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  async function answer(value: AssessmentAnswer) {
    if (!active) return;
    setBusy(true);
    setError(null);
    const question = active.questions[index];
    const savedValue =
      typeof value === "number"
        ? value
        : Object.fromEntries(
            Object.entries(value).map(([k, v]) => [k, v.trim()]),
          );
    try {
      await post(`/assessments/${active.id}/answer`, {
        question_id: question.id,
        ...(typeof savedValue === "number"
          ? { choice: savedValue }
          : { values: savedValue }),
      });
      setActive((previous) =>
        previous?.id === active.id
          ? {
              ...previous,
              answers: { ...previous.answers, [question.id]: savedValue },
            }
          : previous,
      );
      if (typeof savedValue === "object")
        setFieldDrafts((drafts) => ({ ...drafts, [question.id]: savedValue }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  async function action(verb: string) {
    if (!active) return;
    setBusy(true);
    setError(null);
    try {
      const a = await post<Attempt>(`/assessments/${active.id}/${verb}`);
      setActive(a);
      setFieldDrafts({});
      refresh();
      if (verb === "cancel") setActive(null);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  const running = active?.status === "in_progress";
  const remaining = active
    ? Math.max(0, active.expires_at - Math.floor(tick / 1000))
    : 0;
  const resumable = history.data?.find(
    (a) => a.status === "in_progress" && a.expires_at * 1000 > tick,
  );
  const question = active?.questions[index];
  const savedFields = question && active?.answers[question.id];
  const fieldValues = question
    ? (fieldDrafts[question.id] ??
      (typeof savedFields === "object" ? savedFields : {}))
    : {};
  const dirtyIds =
    active?.questions
      .filter((q) => {
        const draft = fieldDrafts[q.id];
        const saved = active.answers[q.id];
        return (
          draft &&
          q.fields?.some(
            (f) =>
              draft[f.name] !==
              (typeof saved === "object" ? saved[f.name] : undefined),
          )
        );
      })
      .map((q) => q.id) ?? [];
  const currentDirty = question ? dirtyIds.includes(question.id) : false;
  return (
    <>
      <PageTitle
        eyebrow="ПРОВЕРКА КОМПЕТЕНЦИЙ"
        title="Покажите, что умеете"
        description="Варианты генерируются на сервере. Результаты и основания сохраняются в вашем профиле."
      />
      <ErrorBox
        error={error || history.error || profile.error || catalog.error}
      />
      {active ? (
        <div className="assessment-layout">
          <section className="panel question-panel">
            <div className="section-heading">
              <span className="grade-tag">
                {specs[active.specialization]} · {active.grade}
              </span>
              {running && (
                <span className={`timer ${remaining < 120 ? "urgent" : ""}`}>
                  <Clock3 size={16} />
                  {Math.floor(remaining / 60)}:
                  {String(remaining % 60).padStart(2, "0")}
                </span>
              )}
            </div>
            {active.status === "completed" ? (
              <>
                <div
                  className={`result-header ${active.result.passed ? "passed" : ""}`}
                >
                  <ShieldCheck size={42} />
                  <p className="eyebrow">РЕЗУЛЬТАТ ПРОВЕРКИ</p>
                  <h2>{active.result.score}% заданий зачтено</h2>
                  <p>
                    {active.kind === "challenge"
                      ? active.result.evidence_outcome === "updated_families"
                        ? "Обновлены проверенные семейства. Остальные результаты сохранены. Грейд не изменился."
                        : active.result.evidence_outcome ===
                            "needs_full_assessment"
                          ? "Результат сохранён в истории. Для обновления общего профиля навыков нужна основная проверка этой категории и рубрики. Прежние оценки и грейд сохранены."
                          : "Результат короткой проверки сохранён. Грейд не изменился."
                      : active.result.grade_outcome === "awarded"
                        ? `Подтверждена категория ${active.grade}`
                        : active.result.grade_outcome === "cooldown"
                          ? `Проверка пройдена. Смена грейда доступна ${date(active.result.next_grade_change_at)}.`
                          : "Порог пока не достигнут. Текущий грейд сохранён."}
                  </p>
                </div>
                {!!Object.keys(active.result.practical_skills ?? {}).length && (
                  <>
                    <h3>Практические задания</h3>
                    <div className="result-skills">
                      {Object.entries(active.result.practical_skills ?? {}).map(
                        ([s, v]) => (
                          <div key={s}>
                            <span>{s}</span>
                            <b>{v}%</b>
                          </div>
                        ),
                      )}
                    </div>
                  </>
                )}
                <h3>Все задания по навыкам</h3>
                <div className="result-skills">
                  {Object.entries(active.result.skills).map(([s, v]) => (
                    <div key={s}>
                      <span>{s}</span>
                      <b>{v}%</b>
                    </div>
                  ))}
                </div>
                {active.result.grade_outcome === "cooldown" && (
                  <Busy
                    busy={busy}
                    variant="outline"
                    disabled={active.result.next_grade_change_at * 1000 > tick}
                    onClick={() => action("claim-grade")}
                  >
                    Применить грейд после интервала
                  </Busy>
                )}
                <p className="notice-inline">
                  Это результат демонстрационного банка, а не калиброванная
                  оценка квалификации на реальных данных.
                </p>
                <h3>Разбор заданий</h3>
                <div className="answer-review">
                  {active.questions.map((q, i) => (
                    <details key={q.id}>
                      <summary>
                        <span
                          className={q.is_correct ? "answer-ok" : "answer-no"}
                        >
                          {q.is_correct ? "✓" : "×"}
                        </span>{" "}
                        {i + 1}. {q.title}
                      </summary>
                      <p>{q.prompt}</p>
                      <QuestionTables question={q} />
                      {q.code && <pre>{q.code}</pre>}
                      <p>
                        Ваш ответ:{" "}
                        <AnswerText question={q} value={active.answers[q.id]} />
                      </p>
                      <p>
                        Правильный ответ:{" "}
                        <AnswerText
                          question={q}
                          value={q.correct_values ?? q.correct_choice}
                        />
                      </p>
                      <p className="muted">{q.explanation}</p>
                    </details>
                  ))}
                </div>
                <Button variant="outline" onClick={() => setActive(null)}>
                  К истории проверок
                </Button>
              </>
            ) : running && question ? (
              <>
                <div className="question-progress">
                  {active.questions.map((q, i) => (
                    <button
                      key={q.id}
                      onClick={() => setIndex(i)}
                      disabled={busy}
                      aria-label={`Задание ${i + 1}${dirtyIds.includes(q.id) ? ", не сохранено" : ""}`}
                      aria-current={i === index ? "step" : undefined}
                      className={`${i === index ? "current" : ""} ${active.answers[q.id] !== undefined ? "answered" : ""} ${dirtyIds.includes(q.id) ? "unsaved" : ""}`}
                    >
                      {i + 1}
                    </button>
                  ))}
                </div>
                <p className="eyebrow">
                  ЗАДАНИЕ {index + 1} / {active.questions.length} ·{" "}
                  {question.skill}
                </p>
                <h2>{question.title}</h2>
                <p className="question-prompt">{question.prompt}</p>
                <QuestionTables question={question} />
                {question.code && (
                  <pre className="code-block">{question.code}</pre>
                )}
                {question.answer_type === "fields" ? (
                  <form
                    className="practical-answer form-stack"
                    onSubmit={(e) => {
                      e.preventDefault();
                      void answer(fieldValues);
                    }}
                  >
                    <div className="form-grid">
                      {question.fields?.map((field) => (
                        <Field
                          key={field.name}
                          label={field.label}
                          hint={
                            field.places
                              ? `До ${field.places} знаков после запятой`
                              : "Целое число"
                          }
                        >
                          <Input
                            required
                            type="text"
                            inputMode="decimal"
                            maxLength={40}
                            autoComplete="off"
                            disabled={busy || !remaining}
                            value={fieldValues[field.name] ?? ""}
                            onChange={(e) =>
                              setFieldDrafts((drafts) => ({
                                ...drafts,
                                [question.id]: {
                                  ...fieldValues,
                                  [field.name]: e.target.value,
                                },
                              }))
                            }
                          />
                        </Field>
                      ))}
                    </div>
                    <p className="tiny muted">
                      Задание засчитывается, когда верны оба значения. Введите
                      числа без единиц и формул; можно использовать запятую или
                      точку.
                    </p>
                    <Busy
                      type="submit"
                      busy={busy}
                      variant="outline"
                      disabled={
                        !remaining ||
                        !currentDirty ||
                        question.fields?.some(
                          (f) => !fieldValues[f.name]?.trim(),
                        )
                      }
                    >
                      <Save size={16} />
                      Сохранить ответ
                    </Busy>
                  </form>
                ) : (
                  <div className="answer-options">
                    {question.choices.map((choice, i) => (
                      <button
                        key={i}
                        disabled={busy || !remaining}
                        type="button"
                        className={
                          active.answers[question.id] === i ? "chosen" : ""
                        }
                        onClick={() => answer(i)}
                      >
                        <span>{String.fromCharCode(65 + i)}</span>
                        <b>{choice}</b>
                        {active.answers[question.id] === i && (
                          <CheckCircle2 size={19} />
                        )}
                      </button>
                    ))}
                  </div>
                )}
                <p className="tiny muted" role="status">
                  {busy
                    ? "Сохраняем ответ…"
                    : currentDirty
                      ? "Изменения не сохранены. Нажмите «Сохранить ответ» перед выходом или перезагрузкой."
                      : active.answers[question.id] !== undefined
                        ? "Ответ сохранён. Его можно изменить до завершения."
                        : question.answer_type === "fields"
                          ? "Заполните оба поля и сохраните ответ."
                          : "Выберите один ответ."}
                </p>
                {dirtyIds.length > 0 &&
                  index === active.questions.length - 1 && (
                    <p className="notice-inline">
                      Есть несохранённые изменения. Откройте задания с оранжевой
                      отметкой и сохраните ответы перед завершением.
                    </p>
                  )}
                {!remaining && (
                  <div className="error-box">
                    Время истекло. Отмените эту попытку и начните новую.
                  </div>
                )}
                <div className="question-actions">
                  <Button
                    variant="outline"
                    onClick={() => setIndex(Math.max(0, index - 1))}
                    disabled={!index || busy}
                  >
                    <ChevronLeft />
                    Назад
                  </Button>
                  {index < active.questions.length - 1 ? (
                    <Button onClick={() => setIndex(index + 1)} disabled={busy}>
                      Далее
                      <ChevronRight />
                    </Button>
                  ) : (
                    <Busy
                      busy={busy}
                      disabled={
                        Object.keys(active.answers).length !==
                          active.questions.length ||
                        !remaining ||
                        dirtyIds.length > 0
                      }
                      onClick={() => action("finish")}
                    >
                      Завершить проверку
                      <Check />
                    </Busy>
                  )}
                </div>
              </>
            ) : (
              <>
                <p>Попытка {statuses[active.status]?.toLowerCase()}.</p>
                <Button variant="outline" onClick={() => setActive(null)}>
                  К проверкам
                </Button>
              </>
            )}
          </section>
          <aside className="panel assessment-rules">
            <ShieldCheck size={28} />
            <h3>Понятные правила</h3>
            <p>
              Проходной порог: {active.rubric.pass_score}% в целом и не менее{" "}
              {active.rubric.min_skill_score}% по каждому навыку.
            </p>
            {!!active.rubric.min_practical_skill_score && (
              <p>
                Дополнительно — не менее{" "}
                {active.rubric.min_practical_skill_score}% практических заданий
                по каждому навыку. У практического задания должны быть верны оба
                значения.
              </p>
            )}
            {active.kind === "challenge" && (
              <p>
                Короткая проверка обновляет результаты навыков и не меняет
                грейд.
              </p>
            )}
            <p>Неудачная попытка не понижает уже подтверждённый грейд.</p>
            <p>
              Менять грейд можно не чаще одного раза в 90 дней. Пройденный тест
              внутри интервала сохраняется.
            </p>
            <p className="tiny muted">
              Рубрика {active.rubric.version}. Ответы доступны только после
              завершения собственной попытки.
            </p>
            <p className="tiny muted">{active.rubric.notice}</p>
            {running && (
              <Busy
                busy={busy}
                variant="outline"
                onClick={() => action("cancel")}
              >
                Отменить попытку
              </Busy>
            )}
          </aside>
        </div>
      ) : (
        <>
          <div className="assessment-layout">
            <section className="panel form-stack">
              <div className="assessment-icon">
                <Code2 size={30} />
              </div>
              <h2>Подтвердите специализацию</h2>
              <p className="muted">
                {catalog.data
                  ? `${catalog.data.question_count} заданий, ${catalog.data.duration_minutes} минут. Из них ${catalog.data.practical_count} — практические случаи с вводом ответа. `
                  : "Загружаем условия проверки. "}
                Анализ кода, SQL и работа с данными. Вы сами выбираете уровень.
              </p>
              <div className="form-grid">
                <Field label="Специализация">
                  <Select
                    value={specialization}
                    onChange={(e) => setSpec(e.target.value)}
                  >
                    {Object.entries(specs).map(([k, v]) => (
                      <option key={k} value={k}>
                        {v}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Грейд">
                  <Select
                    value={grade}
                    onChange={(e) => setGrade(e.target.value)}
                  >
                    <option>Junior</option>
                    <option>Middle</option>
                  </Select>
                </Field>
              </div>
              <p className="notice-inline">
                {catalog.data &&
                  `${catalog.data.rubric.pass_score}% в целом, минимум ${catalog.data.rubric.min_skill_score}% по каждому навыку и ${catalog.data.rubric.min_practical_skill_score}% практических заданий по каждому навыку. `}
                Банк требует калибровки на независимых ответах людей.
              </p>
              {resumable ? (
                <Button
                  onClick={() => {
                    setActive(resumable);
                    setFieldDrafts({});
                    setIndex(0);
                  }}
                >
                  <Play />
                  Продолжить начатую проверку
                </Button>
              ) : (
                <Busy
                  busy={busy}
                  disabled={!catalog.data}
                  onClick={() => start()}
                >
                  <ArrowRight />
                  Начать проверку
                </Busy>
              )}
            </section>
            <section className="panel">
              <Clock3 size={28} className="text-purple" />
              <h2>Поддерживайте актуальность</h2>
              <p className="muted">
                Короткая проверка
                {catalog.data
                  ? `: ${catalog.data.challenge_count} практических задания за ${catalog.data.challenge_duration_minutes} минут. Она`
                  : ""}{" "}
                обновляет свидетельства о навыках. Доступна раз в неделю после
                подтверждения категории и не меняет грейд. Повтор задания
                обновляет только своё семейство. Если для текущей рубрики нет
                полной проверки, результат останется в истории до основной
                проверки.
              </p>
              <Busy
                busy={busy}
                disabled={!profile.data?.grade || !!resumable || !catalog.data}
                variant="outline"
                onClick={() => start("challenge")}
              >
                <RotateCcw />
                Короткая проверка
              </Busy>
              <p className="tiny muted mt-4">
                В текущем прототипе задания из общего банка. Отдельные задания
                работодателей пока не подключены.
              </p>
            </section>
          </div>
          <section className="panel mt-6">
            <h2>История проверок</h2>
            {history.isPending ? (
              <Loading />
            ) : !history.data?.length ? (
              <Empty
                title="Здесь появится ваш прогресс"
                text="Первая попытка сохранится после начала проверки."
              />
            ) : (
              <div className="history-list">
                {history.data.map((a) => (
                  <button
                    key={a.id}
                    onClick={() => {
                      setActive(a);
                      setFieldDrafts({});
                      setIndex(0);
                    }}
                  >
                    <span className="history-icon">
                      <ShieldCheck size={19} />
                    </span>
                    <div>
                      <b>
                        {specs[a.specialization]} · {a.grade}
                      </b>
                      <small>
                        {date(a.created_at)} ·{" "}
                        {a.kind === "challenge"
                          ? "Короткая проверка"
                          : "Категория"}
                        {a.is_demo ? " · демо" : ""}
                      </small>
                    </div>
                    <span>
                      {a.status === "completed"
                        ? `${a.result.score}%`
                        : statuses[a.status]}
                    </span>
                    <ChevronRight size={17} />
                  </button>
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </>
  );
}

export function CompanyPage() {
  const q = useQuery({ queryKey: ["company"], queryFn: () => api("/company") });
  const qc = useQueryClient();
  const { register, reset, handleSubmit } = useForm<Record<string, any>>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    if (q.data) reset(q.data);
  }, [q.data, reset]);
  async function save(data: Record<string, any>) {
    setBusy(true);
    setError(null);
    try {
      await put(
        "/company",
        Object.fromEntries(
          ["name", "description", "industry", "website", "contact"].map((k) => [
            k,
            data[k],
          ]),
        ),
      );
      qc.invalidateQueries({ queryKey: ["company"] });
      toast.success("Профиль компании сохранён");
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="О КОМПАНИИ"
        title="Представьтесь будущим коллегам"
        description="Эти сведения кандидат увидит вместе с вашим предложением."
      />
      {q.isPending ? (
        <Loading />
      ) : (
        <form onSubmit={handleSubmit(save)} className="panel form-stack narrow">
          <Field label="Название компании">
            <Input
              required
              minLength={2}
              maxLength={120}
              {...register("name")}
            />
          </Field>
          <Field label="О компании и команде">
            <textarea rows={5} maxLength={3000} {...register("description")} />
          </Field>
          <Field label="Направление работы">
            <Input maxLength={120} {...register("industry")} />
          </Field>
          <Field label="Сайт">
            <Input
              type="url"
              maxLength={250}
              placeholder="https://example.ru"
              {...register("website")}
            />
          </Field>
          <Field label="Способ связи с работодателем">
            <Input
              required
              minLength={3}
              maxLength={200}
              {...register("contact")}
            />
          </Field>
          <p className="notice-inline">
            Регистрация подтверждает email. Юридическая верификация компании в
            прототипе ещё не подключена.
          </p>
          <ErrorBox error={error || q.error} />
          <Busy busy={busy} type="submit">
            <Save />
            Сохранить компанию
          </Busy>
        </form>
      )}
    </>
  );
}

export function AccountPage() {
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const { user } = useAuth();
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const d = new FormData(e.currentTarget);
    setBusy(true);
    setError(null);
    try {
      setResult(
        await post("/auth/change-email", {
          email: d.get("email"),
          password: d.get("password"),
        }),
      );
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageTitle
        title="Настройки аккаунта"
        description="Текущий подтверждённый email сохраняется до подтверждения нового адреса."
      />
      <form onSubmit={submit} className="panel form-stack narrow">
        <p>
          Текущий адрес: <b>{user?.email}</b>
        </p>
        <Field label="Новый email">
          <Input type="email" name="email" required />
        </Field>
        <Field label="Текущий пароль">
          <Input
            type="password"
            name="password"
            autoComplete="current-password"
            required
          />
        </Field>
        <ErrorBox error={error} />
        {result && (
          <MailNotice
            receipt={result.local_mail_receipt}
            message={result.message}
          />
        )}
        <Busy busy={busy} type="submit">
          Отправить подтверждение
        </Busy>
        <Link to="/forgot-password" className="text-link">
          Восстановить пароль
        </Link>
      </form>
    </>
  );
}

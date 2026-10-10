import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  ArrowDownToLine,
  ArrowRight,
  BriefcaseBusiness,
  Check,
  ChevronRight,
  Inbox,
  LockKeyhole,
  MapPin,
  Plus,
  Send,
  X,
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
} from "@/components/common";
import { api, post, put, download } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { specs, formats, salary, date, statuses } from "@/lib/types";

export function InvitationsPage() {
  const { user } = useAuth();
  const employer = user?.role === "employer";
  const q = useQuery<any[]>({
    queryKey: ["invitations"],
    queryFn: () => api("/invitations"),
    refetchInterval: 15000,
  });
  const [filter, setFilter] = useState("all");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState<unknown>(null);
  const qc = useQueryClient();
  async function action(id: string, action: string) {
    setBusy(id);
    setError(null);
    try {
      await post(`/invitations/${id}/action`, { action });
      await qc.invalidateQueries({ queryKey: ["invitations"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
      qc.invalidateQueries({ queryKey: ["search"] });
      toast.success("Статус приглашения обновлён");
    } catch (e) {
      setError(e);
    } finally {
      setBusy("");
    }
  }
  const rows =
    q.data?.filter(
      (i) =>
        filter === "all" ||
        (filter === "pending"
          ? ["sent", "viewed"].includes(i.status)
          : i.status === filter),
    ) || [];
  return (
    <>
      <PageTitle
        eyebrow={employer ? "ОБРАТНЫЙ НАЙМ" : "ПРЕДЛОЖЕНИЯ ДЛЯ ВАС"}
        title="Приглашения в команду"
        description={
          employer
            ? "Следите за ответами. Контакты открываются после принятия предложения."
            : "Сначала условия. Затем ваше решение о знакомстве."
        }
        action={
          employer && (
            <Button asChild>
              <Link to="/search">
                <Plus />
                Найти кандидата
              </Link>
            </Button>
          )
        }
      />
      <div className="segmented" role="group" aria-label="Статусы приглашений">
        {[
          ["all", "Все"],
          ["pending", "Ожидают ответа"],
          ["accepted", "Принятые"],
          ["declined", "Отклонённые"],
        ].map(([k, l]) => (
          <button
            key={k}
            className={filter === k ? "active" : ""}
            onClick={() => setFilter(k)}
          >
            {l}
          </button>
        ))}
      </div>
      <ErrorBox error={error || q.error} />
      {q.isPending ? (
        <Loading />
      ) : !rows.length ? (
        <Empty
          title="Здесь пока тихо"
          text={
            employer
              ? "Найдите специалиста и отправьте адресное предложение. Публиковать вакансию необязательно."
              : "Подтвердите навыки и опубликуйте профиль, чтобы получать приглашения."
          }
          action={
            <Button variant="outline" asChild>
              <Link to={employer ? "/search" : "/profile"}>
                {employer ? "К поиску" : "Проверить профиль"}
                <ArrowRight />
              </Link>
            </Button>
          }
        />
      ) : (
        <div className="offer-list">
          {rows.map((i) => (
            <article className="offer panel" key={i.id}>
              <div className="offer-header">
                <div className="company-icon">
                  <BriefcaseBusiness size={22} />
                </div>
                <div>
                  <p className="eyebrow">
                    {employer
                      ? i.candidate?.name || "Профиль скрыт кандидатом"
                      : i.company.name}
                  </p>
                  <h2>{i.title}</h2>
                </div>
                <span className={`status status-${i.status}`}>
                  {statuses[i.status]}
                </span>
              </div>
              <div className="offer-columns">
                <div>
                  <p className="offer-salary">
                    {salary(i.salary_min, i.salary_max)} <small>/ месяц</small>
                  </p>
                  <p className="offer-description">{i.description}</p>
                  <div className="offer-meta">
                    <span>{i.company.industry}</span>
                    <span>Отправлено {date(i.created_at)}</span>
                    {["sent", "viewed"].includes(i.status) && (
                      <span>Ответить до {date(i.expires_at)}</span>
                    )}
                  </div>
                  <details className="company-description">
                    <summary>О компании и способе связи</summary>
                    <p>{i.company.description}</p>
                    <p>Связь: {i.contact}</p>
                    {i.company.website && (
                      <a
                        href={i.company.website}
                        target="_blank"
                        rel="noreferrer"
                        className="text-link"
                      >
                        Сайт компании ↗
                      </a>
                    )}
                    <p className="tiny muted">
                      Юридическая верификация компании пока не подключена.
                    </p>
                  </details>
                </div>
                <div className="offer-contact">
                  <LockKeyhole size={20} />
                  <h3>
                    {i.candidate?.contacts
                      ? "Контакты доступны"
                      : i.status === "accepted" && !i.contact_revoked
                        ? "Знакомство разрешено"
                        : "Контакты защищены"}
                  </h3>
                  {i.candidate?.contacts ? (
                    <>
                      <a
                        className="text-link"
                        href={`mailto:${i.candidate.contacts.email}`}
                      >
                        {i.candidate.contacts.email}
                      </a>
                      <p>{i.candidate.contacts.phone}</p>
                      <p>{i.candidate.contacts.telegram}</p>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() =>
                          download(
                            `/candidates/${i.candidate_id}/pdf`,
                            "fsp-profile.pdf",
                          ).catch((e) => toast.error(e.message))
                        }
                      >
                        <ArrowDownToLine />
                        Профиль PDF
                      </Button>
                    </>
                  ) : (
                    <p>
                      {i.contact_revoked
                        ? "Разрешение по этому приглашению отозвано."
                        : i.status === "accepted"
                          ? "Эта компания может связаться с кандидатом."
                          : "До принятия предложения email, имя и телефон кандидата скрыты."}
                    </p>
                  )}
                </div>
              </div>
              <div className="offer-actions">
                {["sent", "viewed"].includes(i.status) &&
                  (employer ? (
                    <Busy
                      busy={busy === i.id}
                      variant="outline"
                      onClick={() => action(i.id, "withdraw")}
                    >
                      Отозвать приглашение
                    </Busy>
                  ) : (
                    <>
                      {i.status === "sent" && (
                        <Button
                          disabled={!!busy}
                          variant="ghost"
                          onClick={() => action(i.id, "view")}
                        >
                          Отметить просмотренным
                        </Button>
                      )}
                      <Busy
                        busy={busy === i.id}
                        variant="outline"
                        onClick={() => action(i.id, "decline")}
                      >
                        <X />
                        Отклонить
                      </Busy>
                      <Busy
                        busy={busy === i.id}
                        onClick={() => action(i.id, "accept")}
                      >
                        <Check />
                        Принять и открыть контакты
                      </Busy>
                    </>
                  ))}
                {!employer && i.status === "accepted" && !i.contact_revoked && (
                  <Busy
                    busy={busy === i.id}
                    variant="outline"
                    onClick={() => action(i.id, "revoke_contacts")}
                  >
                    Отозвать доступ по приглашению
                  </Busy>
                )}
              </div>
              {!employer && i.contact_revoked && (
                <p className="tiny muted">
                  Самостоятельный отклик или другое принятое приглашение могут
                  сохранять доступ этой компании. Их можно отозвать отдельно.
                </p>
              )}
            </article>
          ))}
        </div>
      )}
    </>
  );
}

export function JobsPage() {
  const { user } = useAuth();
  const employer = user?.role === "employer";
  const qc = useQueryClient();
  const q = useQuery<any[]>({
    queryKey: ["jobs"],
    queryFn: () => api("/jobs"),
  });
  const [edit, setEdit] = useState<any>(null);
  const [applying, setApplying] = useState<any>(null);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [spec, setSpec] = useState("");
  async function save(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const d = new FormData(e.currentTarget);
    setBusy(true);
    setError(null);
    const data = {
      title: d.get("title"),
      description: d.get("description"),
      specialization: d.get("specialization"),
      grade: d.get("grade"),
      work_format: d.get("work_format"),
      salary_min: Number(d.get("salary_min")),
      salary_max: Number(d.get("salary_max")),
      currency: "RUB",
      skills: String(d.get("skills") || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      active: true,
    };
    try {
      if (edit.id) await put(`/jobs/${edit.id}`, data);
      else await post("/jobs", data);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      setEdit(null);
      toast.success("Вакансия сохранена");
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  async function toggle(job: any) {
    setBusy(true);
    setError(null);
    try {
      await put(
        `/jobs/${job.id}`,
        Object.fromEntries(
          [
            "title",
            "description",
            "specialization",
            "grade",
            "work_format",
            "salary_min",
            "salary_max",
            "currency",
            "skills",
            "active",
          ].map((k) => [k, k === "active" ? !job.active : job[k]]),
        ),
      );
      qc.invalidateQueries({ queryKey: ["jobs"] });
      toast.success(
        job.active ? "Вакансия снята с публикации" : "Вакансия опубликована",
      );
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  async function apply(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const d = new FormData(e.currentTarget);
    setBusy(true);
    setError(null);
    try {
      await post(`/jobs/${applying.id}/apply`, {
        message: d.get("message"),
        contact_consent: consent,
      });
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["applications"] });
      setApplying(null);
      setConsent(false);
      toast.success("Отклик отправлен. Контакты доступны этой компании.");
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  const jobs = q.data?.filter((j) => !spec || j.specialization === spec) || [];
  return (
    <>
      <PageTitle
        eyebrow="ОТКРЫТЫЕ ВОЗМОЖНОСТИ"
        title={
          employer ? "Вакансии вашей компании" : "Найдите интересную задачу"
        }
        description={
          employer
            ? "Дополнительный путь знакомства: кандидаты смогут откликнуться сами."
            : "Можно откликаться без ФСП ID и подтверждённой категории."
        }
        action={
          employer ? (
            <Button
              onClick={() => {
                setEdit({});
                setError(null);
              }}
            >
              <Plus />
              Создать вакансию
            </Button>
          ) : (
            <Button variant="outline" asChild>
              <Link to="/applications">
                Мои отклики
                <ArrowRight />
              </Link>
            </Button>
          )
        }
      />
      <div className="mb-6">
        <Select
          aria-label="Направление вакансий"
          value={spec}
          onChange={(e) => setSpec(e.target.value)}
        >
          <option value="">Все направления</option>
          {Object.entries(specs).map(([k, v]) => (
            <option key={k} value={k}>
              {v}
            </option>
          ))}
        </Select>
      </div>
      <ErrorBox error={error || q.error} />
      {q.isPending ? (
        <Loading />
      ) : !jobs.length ? (
        <Empty
          title="Вакансий пока нет"
          text={
            employer
              ? "Создайте первую вакансию или отправьте приглашение напрямую из поиска."
              : "Попробуйте другое направление или вернитесь позже."
          }
        />
      ) : (
        <div className="jobs-grid">
          {jobs.map((j) => (
            <article className="job-card panel" key={j.id}>
              <div className="job-company">
                <div className="company-icon">
                  <BriefcaseBusiness size={22} />
                </div>
                <span>{j.company.name}</span>
                <span className={`status ${j.active ? "status-accepted" : ""}`}>
                  {j.active ? "Открыта" : "Скрыта"}
                </span>
              </div>
              <h2>{j.title}</h2>
              <p className="offer-salary">
                {salary(j.salary_min, j.salary_max)}
              </p>
              <p className="meta-line">
                {j.grade} · {formats[j.work_format]}
              </p>
              <p className="job-description">{j.description}</p>
              <div className="tags">
                {j.skills.map((s: string) => (
                  <span key={s}>{s}</span>
                ))}
              </div>
              {employer ? (
                <div className="flex gap-2 flex-wrap mt-5">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setEdit(j);
                      setError(null);
                    }}
                  >
                    Редактировать
                  </Button>
                  <Busy busy={busy} variant="ghost" onClick={() => toggle(j)}>
                    {j.active ? "Снять с публикации" : "Опубликовать"}
                  </Busy>
                </div>
              ) : (
                <Button
                  className="w-full mt-5"
                  disabled={
                    !!j.application_id && j.application_status !== "withdrawn"
                  }
                  onClick={() => {
                    setApplying(j);
                    setConsent(false);
                    setError(null);
                  }}
                >
                  {j.application_id && j.application_status !== "withdrawn"
                    ? `Отклик: ${statuses[j.application_status]}`
                    : "Откликнуться"}
                  <ArrowRight />
                </Button>
              )}
            </article>
          ))}
        </div>
      )}
      <Dialog
        open={!!edit}
        onOpenChange={(v) => {
          if (!v) setEdit(null);
        }}
      >
        <DialogContent className="modal-scroll">
          <DialogHeader>
            <DialogTitle>
              {edit?.id ? "Редактировать вакансию" : "Новая вакансия"}
            </DialogTitle>
            <DialogDescription>
              Зарплатная вилка обязательна и показывается в рублях.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={save} className="form-stack">
            <Field label="Название">
              <Input
                name="title"
                defaultValue={edit?.title}
                required
                minLength={3}
                maxLength={150}
              />
            </Field>
            <Field label="Команда, задачи и условия">
              <textarea
                name="description"
                defaultValue={edit?.description}
                required
                minLength={10}
                maxLength={5000}
                rows={4}
              />
            </Field>
            <div className="form-grid">
              <Field label="Специализация">
                <Select
                  name="specialization"
                  defaultValue={edit?.specialization || "backend"}
                >
                  {Object.entries(specs).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Грейд">
                <Select name="grade" defaultValue={edit?.grade || "Junior"}>
                  <option>Junior</option>
                  <option>Middle</option>
                </Select>
              </Field>
              <Field label="Зарплата от, ₽">
                <Input
                  name="salary_min"
                  type="number"
                  defaultValue={edit?.salary_min}
                  required
                  min={1}
                  max={10000000}
                />
              </Field>
              <Field label="До, ₽">
                <Input
                  name="salary_max"
                  type="number"
                  defaultValue={edit?.salary_max}
                  required
                  min={1}
                  max={10000000}
                />
              </Field>
              <Field label="Формат работы">
                <Select
                  name="work_format"
                  defaultValue={edit?.work_format || "remote"}
                >
                  {Object.entries(formats).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Навыки через запятую">
                <Input name="skills" defaultValue={edit?.skills?.join(", ")} />
              </Field>
            </div>
            <ErrorBox error={error} />
            <Busy busy={busy} type="submit">
              Сохранить и опубликовать
            </Busy>
          </form>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!applying}
        onOpenChange={(v) => {
          if (!v) setApplying(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Откликнуться на вакансию</DialogTitle>
            <DialogDescription>
              {applying?.title} · {applying?.company.name}
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={apply} className="form-stack">
            <Field label="Сопроводительное сообщение">
              <textarea
                name="message"
                rows={4}
                maxLength={2000}
                placeholder="Что вам интересно в этой задаче?"
              />
            </Field>
            <CheckField checked={consent} onChange={setConsent}>
              Разрешаю компании «{applying?.company.name}» видеть имя, email,
              телефон и мессенджер из моего профиля.
            </CheckField>
            <ErrorBox error={error} />
            <Busy busy={busy} disabled={!consent} type="submit">
              <Send />
              Отправить отклик
            </Busy>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}

export function ApplicationsPage() {
  const { user } = useAuth();
  const employer = user?.role === "employer";
  const qc = useQueryClient();
  const q = useQuery<any[]>({
    queryKey: ["applications"],
    queryFn: () => api("/applications"),
    refetchInterval: 15000,
  });
  const [busy, setBusy] = useState("");
  const [error, setError] = useState<unknown>(null);
  async function change(id: string, status: string) {
    setBusy(id);
    setError(null);
    try {
      await post(`/applications/${id}/status`, { status });
      qc.invalidateQueries({ queryKey: ["applications"] });
      qc.invalidateQueries({ queryKey: ["jobs"] });
      toast.success("Статус отклика обновлён");
    } catch (e) {
      setError(e);
    } finally {
      setBusy("");
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="САМОСТОЯТЕЛЬНЫЕ ЗНАКОМСТВА"
        title={employer ? "Отклики кандидатов" : "Мои отклики"}
        description={
          employer
            ? "Отклик открывает контакты только вашей компании."
            : "Отозвав отклик, вы отзываете разрешение на контакты по нему."
        }
      />
      <ErrorBox error={error || q.error} />
      {q.isPending ? (
        <Loading />
      ) : !q.data?.length ? (
        <Empty
          title="Откликов пока нет"
          text={
            employer
              ? "Они появятся, когда кандидаты ответят на ваши вакансии."
              : "Выберите интересную вакансию и расскажите о себе."
          }
          action={
            <Button asChild variant="outline">
              <Link to="/jobs">
                К вакансиям
                <ArrowRight />
              </Link>
            </Button>
          }
        />
      ) : (
        <div className="offer-list">
          {q.data.map((a) => (
            <article key={a.id} className="panel offer">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">
                    {employer
                      ? a.candidate?.name || "Отклик отозван"
                      : a.job.company.name}
                  </p>
                  <h2>{a.job.title}</h2>
                  <p className="muted">
                    {salary(a.job.salary_min, a.job.salary_max)} ·{" "}
                    {date(a.created_at)}
                  </p>
                </div>
                <span className={`status status-${a.status}`}>
                  {statuses[a.status]}
                </span>
              </div>
              <p className="offer-description">
                {a.message || "Без сопроводительного сообщения"}
              </p>
              {a.candidate?.contacts && (
                <div className="notice">
                  <LockKeyhole size={20} />
                  <div>
                    <b>{a.candidate.contacts.full_name}</b>
                    <p>
                      {a.candidate.contacts.email} ·{" "}
                      {a.candidate.contacts.phone}
                    </p>
                    <p>{a.candidate.contacts.telegram}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        download(
                          `/candidates/${a.candidate_id}/pdf`,
                          "fsp-profile.pdf",
                        ).catch((e) => toast.error(e.message))
                      }
                    >
                      Скачать PDF
                    </Button>
                  </div>
                </div>
              )}
              {!employer && (
                <details>
                  <summary>Связаться с компанией</summary>
                  <p>{a.job.company.contact}</p>
                </details>
              )}
              <div className="offer-actions">
                {a.status !== "withdrawn" &&
                  (employer ? (
                    a.status !== "declined" && (
                      <>
                        {a.status === "sent" && (
                          <Busy
                            busy={busy === a.id}
                            variant="ghost"
                            onClick={() => change(a.id, "viewed")}
                          >
                            Отметить просмотренным
                          </Busy>
                        )}
                        <Busy
                          busy={busy === a.id}
                          variant="outline"
                          onClick={() => change(a.id, "declined")}
                        >
                          Отклонить
                        </Busy>
                        <Busy
                          busy={busy === a.id}
                          disabled={a.status === "interview"}
                          onClick={() => change(a.id, "interview")}
                        >
                          Пригласить на собеседование
                        </Busy>
                      </>
                    )
                  ) : (
                    <Busy
                      busy={busy === a.id}
                      variant="outline"
                      onClick={() => change(a.id, "withdrawn")}
                    >
                      Отозвать отклик и доступ
                    </Busy>
                  ))}
              </div>
            </article>
          ))}
        </div>
      )}
    </>
  );
}

// Composition adapted from shadcn/ui login-03 (MIT), with actual auth states.
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  ArrowUpRight,
  BriefcaseBusiness,
  Code2,
  Check,
  MailCheck,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Busy, CheckField, ErrorBox, Field, Select } from "@/components/common";
import { api, post } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export function Brand({ light = false }: { light?: boolean }) {
  return (
    <Link
      to="/"
      className={`brand ${light ? "brand-light" : ""}`}
      aria-label="ФСП Карьера — главная"
    >
      <span className="brand-mark">
        <span />
        <span />
        <span />
      </span>
      <span>
        <b>
          ФСП<span className="brand-slash">/</span>Карьера
        </b>
        <small>НАВЫКИ РЕШАЮТ</small>
      </span>
    </Link>
  );
}

export function MailNotice({
  receipt,
  message,
}: {
  receipt?: string;
  message?: string;
}) {
  const [show, setShow] = useState(false);
  const mail = useQuery({
    queryKey: ["mail", receipt],
    queryFn: () => api(`/auth/local-mail/${receipt}`),
    enabled: show && !!receipt,
    retry: false,
  });
  const url = mail.data?.body?.match(/https?:\/\/\S+/)?.[0];
  return (
    <div className="notice">
      <MailCheck size={20} />
      <div>
        <strong>{message || "Проверьте почту"}</strong>
        {receipt && (
          <>
            <p>
              На этом стенде письмо доступно в локальной демонстрационной почте.
            </p>
            <Button
              type="button"
              variant="outline"
              onClick={() => setShow(true)}
            >
              Открыть демо-письмо
            </Button>
            {show && (
              <>
                <ErrorBox error={mail.error} />
                {mail.data && (
                  <div className="mail-preview">
                    <small>{mail.data.email}</small>
                    <p>{mail.data.notice}</p>
                    {url && (
                      <a className="text-link" href={url}>
                        Перейти по ссылке из письма <ArrowUpRight size={14} />
                      </a>
                    )}
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export function Welcome() {
  const { user } = useAuth();
  return (
    <div className="welcome">
      <nav className="landing-nav">
        <Brand />
        <div className="flex gap-3">
          {user ? (
            <Button asChild>
              <Link to="/dashboard">
                В кабинет <ArrowRight />
              </Link>
            </Button>
          ) : (
            <>
              <Button asChild variant="ghost">
                <Link to="/login">Войти</Link>
              </Button>
              <Button asChild>
                <Link to="/register">Создать профиль</Link>
              </Button>
            </>
          )}
        </div>
      </nav>
      <main className="landing-grid">
        <section className="landing-copy">
          <p className="eyebrow">
            <span className="live-dot" /> ФЕДЕРАЦИЯ СПОРТИВНОГО ПРОГРАММИРОВАНИЯ
          </p>
          <h1>
            Ваши навыки.
            <br />
            Новые <span>возможности.</span>
          </h1>
          <p className="landing-lead">
            Подтвердите, что умеете, и получайте предложения от команд, которым
            это нужно.
          </p>
          <div className="flex gap-3 flex-wrap">
            <Button size="lg" asChild>
              <Link to="/register">
                Начать с проверки навыков <ArrowRight />
              </Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link to="/login">Я ищу специалистов</Link>
            </Button>
          </div>
          <div className="landing-promises">
            <span>
              <Check />
              Условия и зарплата заранее
            </span>
            <span>
              <Check />
              Контакты только с согласия
            </span>
            <span>
              <Check />
              Можно без ФСП ID
            </span>
          </div>
        </section>
        <section
          className="landing-passport"
          aria-label="Как работает подтверждение"
        >
          <div className="passport-top">
            <ShieldCheck size={32} />
            <span>ПРОФИЛЬ ДОСТИЖЕНИЙ</span>
            <span className="demo-tag">Пример</span>
          </div>
          <h2>
            Меньше догадок.
            <br />
            Больше оснований.
          </h2>
          <div className="passport-row">
            <span className="passport-num">01</span>
            <div>
              <b>Проверяете навыки</b>
              <p>Задания с прозрачными критериями</p>
            </div>
          </div>
          <div className="passport-row">
            <span className="passport-num">02</span>
            <div>
              <b>Попадаете в подборку</b>
              <p>Совпадают навыки, грейд и условия</p>
            </div>
          </div>
          <div className="passport-row">
            <span className="passport-num">03</span>
            <div>
              <b>Выбираете предложение</b>
              <p>Только вы решаете, с кем общаться</p>
            </div>
          </div>
          <p className="passport-footer">
            Резюме рассказывает. Проверка показывает.
          </p>
        </section>
      </main>
      <footer className="landing-footer">
        <span>ФСП / Карьера · Хакатон 2026</span>
        <Link to="/privacy">Приватность и правила демо</Link>
      </footer>
    </div>
  );
}

export function AuthPage({
  mode,
}: {
  mode: "login" | "register" | "forgot" | "verify" | "reset";
}) {
  const [role, setRole] = useState("candidate");
  const [consent, setConsent] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<any>(null);
  const { enter } = useAuth();
  const navigate = useNavigate();
  const title = {
    login: "С возвращением",
    register: "Начнём знакомство",
    forgot: "Восстановить доступ",
    verify: "Подтверждение почты",
    reset: "Новый пароль",
  }[mode];
  const token = new URLSearchParams(window.location.search).get("token") || "";
  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "login") {
        const body = new URLSearchParams({ username: email, password });
        const r = await api("/auth/login", {
          method: "POST",
          body,
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
        });
        enter(r);
        navigate({ to: r.user.role === "employer" ? "/search" : "/dashboard" });
      } else {
        const path = {
          register: "/register",
          forgot: "/forgot-password",
          verify: "/verify",
          reset: "/reset-password",
        }[mode];
        const data =
          mode === "register"
            ? {
                email,
                password,
                full_name: name,
                role,
                processing_consent: consent,
              }
            : mode === "forgot"
              ? { email }
              : mode === "verify"
                ? { token }
                : { token, password };
        setResult(await post(`/auth${path}`, data));
        if (mode === "verify" || mode === "reset")
          window.history.replaceState({}, "", window.location.pathname);
      }
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  async function resend() {
    setBusy(true);
    setError(null);
    try {
      setResult(await post("/auth/resend", { email }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="auth-page">
      <Brand />
      <Card className="auth-card">
        <CardContent>
          <p className="eyebrow">ФСП / КАРЬЕРА</p>
          <h1>{title}</h1>
          <p className="muted">
            {mode === "register"
              ? "Создайте аккаунт и подтвердите email."
              : mode === "verify"
                ? "Подтвердите адрес, чтобы открыть кабинет."
                : "Навыки, достижения и предложения в одном месте."}
          </p>
          <form onSubmit={submit} className="form-stack">
            {mode === "register" && (
              <>
                <div className="role-picker">
                  <button
                    type="button"
                    className={role === "candidate" ? "active" : ""}
                    onClick={() => setRole("candidate")}
                  >
                    <Code2 />Я кандидат
                  </button>
                  <button
                    type="button"
                    className={role === "employer" ? "active" : ""}
                    onClick={() => setRole("employer")}
                  >
                    <BriefcaseBusiness />Я работодатель
                  </button>
                </div>
                <Field label="Имя и фамилия">
                  <Input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                    minLength={2}
                    autoComplete="name"
                  />
                </Field>
              </>
            )}
            {["login", "register", "forgot"].includes(mode) && (
              <Field label="Email">
                <Input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                  placeholder="you@example.ru"
                />
              </Field>
            )}
            {["login", "register", "reset"].includes(mode) && (
              <Field
                label="Пароль"
                hint={mode !== "login" ? "Не менее 10 символов" : undefined}
              >
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  minLength={mode === "login" ? 1 : 10}
                  maxLength={128}
                  required
                  autoComplete={
                    mode === "login" ? "current-password" : "new-password"
                  }
                />
              </Field>
            )}
            {mode === "register" && (
              <CheckField checked={consent} onChange={setConsent}>
                Согласен на обработку данных по{" "}
                <Link to="/privacy" className="text-link">
                  правилам платформы
                </Link>
                . Публикация профиля включается отдельно.
              </CheckField>
            )}
            <ErrorBox error={error} />
            {result && (
              <MailNotice
                receipt={result.local_mail_receipt}
                message={result.message}
              />
            )}
            <Busy
              busy={busy}
              type="submit"
              className="w-full"
              disabled={
                (mode === "register" && !consent) ||
                ((mode === "verify" || mode === "reset") &&
                  (!!result || !token))
              }
            >
              {mode === "login"
                ? "Войти"
                : mode === "register"
                  ? "Создать аккаунт"
                  : mode === "forgot"
                    ? "Отправить письмо"
                    : mode === "verify"
                      ? "Подтвердить адрес"
                      : "Сохранить пароль"}
              <ArrowRight size={16} />
            </Busy>
            {mode === "verify" && !token && !result && (
              <p className="muted">
                В ссылке нет токена. Откройте ссылку из письма.
              </p>
            )}
          </form>
          <div className="auth-links">
            {mode === "login" ? (
              <>
                <Link to="/forgot-password">Забыли пароль?</Link>
                <Link to="/register">Создать аккаунт</Link>
                <button
                  type="button"
                  disabled={!email || busy}
                  onClick={resend}
                >
                  Отправить подтверждение ещё раз
                </button>
              </>
            ) : (
              <Link to="/login">Перейти ко входу</Link>
            )}
          </div>
        </CardContent>
      </Card>
      <Link to="/" className="muted">
        ← На главную
      </Link>
    </main>
  );
}

export function Privacy() {
  return (
    <main className="privacy-page">
      <Brand />
      <h1>Ваш профиль — под вашим контролем</h1>
      <p className="muted">
        Версия согласия 2026-10-v1 · Демонстрационный прототип
      </p>
      <h2>Что хранится</h2>
      <p>
        Данные аккаунта, заполненное резюме, предпочтения, попытки проверок и
        история предложений. Пароль хранится в виде Argon2-хеша. На
        демонстрационном стенде используйте вымышленные данные.
      </p>
      <h2>Кто что видит</h2>
      <p>
        Профиль попадает в поиск после публикации и подтверждения категории. До
        разрешения работодатель видит обезличенный профиль, навыки и достижения.
        Email, имя и закрытые контакты открываются только компании, чьё
        приглашение вы приняли, либо компании, которой вы отправили отклик.
      </p>
      <h2>Как отозвать разрешение</h2>
      <p>
        Снимите публикацию в профиле — он исчезнет из новых и сохранённых
        подборок. Контакты по принятому приглашению можно отозвать в его
        карточке; по отклику — отозвать сам отклик. Если у компании осталось
        другое действующее разрешение, доступ сохраняется. Уже скачанные
        работодателем файлы невозможно отозвать технически.
      </p>
      <h2>О проверках и ФСП</h2>
      <p>
        Категория определяется заданиями демонстрационного банка. Эти результаты
        ещё не калиброваны на людях и не доказывают пригодность к работе.
        Введённый ФСП ID ожидает проверки официальным источником.
        Демо-достижения явно помечены. ФСП ID не обязателен для поиска работы.
      </p>
      <h2>О запуске для реальных пользователей</h2>
      <p>
        Этот стенд предназначен для хакатонной проверки. До промышленного
        использования оператор должен указать свои реквизиты, основание
        обработки, сроки хранения, порядок удаления данных и обращения к
        оператору, подключить рабочую почту и провести правовую и техническую
        проверку.
      </p>
      <Button asChild variant="outline">
        <Link to="/">Вернуться на главную</Link>
      </Button>
    </main>
  );
}

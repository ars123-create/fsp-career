import React, { useState, type ReactNode } from "react";
import ReactDOM from "react-dom/client";
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query";
import {
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
  Outlet,
  Link,
  useRouterState,
  useNavigate,
} from "@tanstack/react-router";
import {
  Bell,
  Bookmark,
  BriefcaseBusiness,
  Building2,
  ChevronRight,
  CircleHelp,
  ClipboardCheck,
  Code2,
  Inbox,
  LayoutDashboard,
  LogOut,
  Menu,
  Search,
  Settings,
  ShieldCheck,
  UserRound,
  X,
} from "lucide-react";
import { Toaster, toast } from "sonner";
import { AuthProvider, useAuth } from "@/lib/auth";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Loading } from "@/components/common";
import { Welcome, AuthPage, Brand, Privacy } from "@/pages/auth";
import {
  Dashboard,
  ProfilePage,
  AssessmentPage,
  CompanyPage,
  AccountPage,
} from "@/pages/candidate";
import { SearchPage } from "@/pages/search";
import { ApplicationsPage, InvitationsPage, JobsPage } from "@/pages/hiring";
import "@fontsource/manrope/400.css";
import "@fontsource/manrope/600.css";
import "@fontsource/manrope/700.css";
import "@fontsource/manrope/800.css";
import "./styles.css";

const qc = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 15000, refetchOnWindowFocus: true },
  },
});
function Root() {
  return (
    <AuthProvider>
      <Outlet />
      <Toaster position="bottom-right" richColors closeButton />
    </AuthProvider>
  );
}

function Shell({ children, role }: { children: ReactNode; role?: string }) {
  const { user, loading, logout } = useAuth();
  const [menu, setMenu] = useState(false);
  const path = useRouterState({ select: (s) => s.location.pathname });
  const navigate = useNavigate();
  const d = useQuery({
    queryKey: ["dashboard"],
    queryFn: () => api("/dashboard"),
    enabled: !!user,
    refetchInterval: 30000,
  });
  if (loading)
    return (
      <main className="auth-page">
        <Loading />
      </main>
    );
  if (!user) return <AuthPage mode="login" />;
  if (role && role !== user.role)
    return (
      <div className="auth-page">
        <h1>Этот раздел доступен другой роли</h1>
        <Button asChild>
          <Link to="/dashboard">В свой кабинет</Link>
        </Button>
      </div>
    );
  const employer = user.role === "employer";
  const links = employer
    ? ([
        ["/dashboard", "Обзор", LayoutDashboard],
        ["/search", "Поиск кандидатов", Search],
        ["/invitations", "Приглашения", SendIcon],
        ["/jobs", "Вакансии", BriefcaseBusiness],
        ["/applications", "Отклики", Inbox],
        ["/company", "Компания", Building2],
      ] as const)
    : ([
        ["/dashboard", "Обзор", LayoutDashboard],
        ["/profile", "Мой профиль", UserRound],
        ["/assessments", "Проверка навыков", ClipboardCheck],
        ["/invitations", "Приглашения", SendIcon],
        ["/jobs", "Вакансии", BriefcaseBusiness],
        ["/applications", "Мои отклики", Inbox],
      ] as const);
  async function signout() {
    try {
      await logout();
    } catch {
      toast.info("Локальная сессия завершена");
    }
    navigate({ to: "/" });
  }
  return (
    <div className="app-shell">
      {menu && (
        <button
          className="mobile-backdrop"
          aria-label="Закрыть меню"
          onClick={() => setMenu(false)}
        />
      )}
      <aside className={`sidebar ${menu ? "open" : ""}`}>
        <Brand light />
        <p className="sidebar-label">
          {employer ? "КАБИНЕТ РАБОТОДАТЕЛЯ" : "КАБИНЕТ КАНДИДАТА"}
        </p>
        <nav>
          {links.map(([to, label, Icon]) => (
            <Link
              to={to}
              key={to}
              onClick={() => setMenu(false)}
              className={path === to ? "active" : ""}
            >
              <Icon size={19} />
              <span>{label}</span>
              {to === "/invitations" && d.data?.pending > 0 && (
                <b className="nav-count">{d.data.pending}</b>
              )}
            </Link>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <ShieldCheck size={23} />
            <b>Навыки говорят за вас</b>
            <p>Подтверждения и возможности в одном профиле.</p>
          </div>
          <Link to="/account" onClick={() => setMenu(false)}>
            <Settings size={17} />
            Настройки аккаунта
          </Link>
          <Link to="/privacy">
            <CircleHelp size={17} />
            Приватность
          </Link>
          <div className="sidebar-user">
            <div className="user-circle">{user.full_name[0]}</div>
            <div>
              <b>{user.full_name}</b>
              <small>{employer ? "Работодатель" : "Кандидат"}</small>
            </div>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Выйти"
              onClick={signout}
            >
              <LogOut size={17} />
            </Button>
          </div>
        </div>
      </aside>
      <div className="app-main">
        <header className="topbar">
          <div>
            <Button
              className="mobile-menu"
              variant="ghost"
              size="icon"
              aria-label="Открыть меню"
              aria-expanded={menu}
              onClick={() => setMenu(true)}
            >
              <Menu />
            </Button>
            <span className="topbar-section">Рабочее пространство</span>
            <ChevronRight size={14} />
            <b>{links.find((l) => l[0] === path)?.[1] || "Настройки"}</b>
          </div>
          <div>
            {user.is_demo && (
              <span className="demo-badge">
                <span />
                Демо-режим
              </span>
            )}
            <Link
              to="/invitations"
              className="notification"
              aria-label="Открыть приглашения"
            >
              <Bell size={19} />
              {d.data?.pending > 0 && <i />}
            </Link>
            <div className="user-circle small">{user.full_name[0]}</div>
          </div>
        </header>
        <main className="workspace">
          {user.is_demo && (
            <p className="demo-banner">
              Демонстрационный стенд · профили, компании и достижения вымышлены
            </p>
          )}
          {children}
        </main>
        <footer className="workspace-footer">
          <span>ФСП / Карьера · 2026</span>
          <span>Подтверждённые навыки. Осознанный выбор.</span>
        </footer>
      </div>
    </div>
  );
}
function SendIcon({ size = 19 }: { size?: number }) {
  return <Inbox size={size} />;
}
const root = createRootRoute({
  component: Root,
  notFoundComponent: () => (
    <main className="auth-page">
      <h1>Страница не найдена</h1>
      <Link to="/">На главную</Link>
    </main>
  ),
  errorComponent: ({ error }) => (
    <main className="auth-page">
      <h1>Не удалось открыть страницу</h1>
      <p>
        {error instanceof Error
          ? error.message
          : "Обновите страницу и попробуйте снова."}
      </p>
      <Button onClick={() => window.location.reload()}>Повторить</Button>
    </main>
  ),
});
const routes = [
  createRoute({ getParentRoute: () => root, path: "/", component: Welcome }),
  createRoute({
    getParentRoute: () => root,
    path: "/login",
    component: () => <AuthPage mode="login" />,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/register",
    component: () => <AuthPage mode="register" />,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/forgot-password",
    component: () => <AuthPage mode="forgot" />,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/verify",
    component: () => <AuthPage mode="verify" />,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/reset-password",
    component: () => <AuthPage mode="reset" />,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/privacy",
    component: Privacy,
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/dashboard",
    component: () => (
      <Shell>
        <Dashboard />
      </Shell>
    ),
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/search",
    component: () => (
      <Shell role="employer">
        <SearchPage />
      </Shell>
    ),
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/profile",
    component: () => (
      <Shell role="candidate">
        <ProfilePage />
      </Shell>
    ),
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/assessments",
    component: () => (
      <Shell role="candidate">
        <AssessmentPage />
      </Shell>
    ),
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/company",
    component: () => (
      <Shell role="employer">
        <CompanyPage />
      </Shell>
    ),
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/invitations",
    component: () => (
      <Shell>
        <InvitationsPage />
      </Shell>
    ),
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/jobs",
    component: () => (
      <Shell>
        <JobsPage />
      </Shell>
    ),
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/applications",
    component: () => (
      <Shell>
        <ApplicationsPage />
      </Shell>
    ),
  }),
  createRoute({
    getParentRoute: () => root,
    path: "/account",
    component: () => (
      <Shell>
        <AccountPage />
      </Shell>
    ),
  }),
];
const router = createRouter({ routeTree: root.addChildren(routes) });
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>,
);

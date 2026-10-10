import { useId, type ReactNode, type SelectHTMLAttributes } from "react";
import {
  AlertCircle,
  ArrowRight,
  Check,
  Loader2,
  SearchX,
  ShieldCheck,
} from "lucide-react";
import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

export function PageTitle({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-title">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p className="muted page-description">{description}</p>}
      </div>
      {action}
    </header>
  );
}
export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}
export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`select ${props.className || ""}`} />;
}
export function ErrorBox({ error }: { error: unknown }) {
  return error ? (
    <div className="error-box" role="alert">
      <AlertCircle size={18} />
      <span>{error instanceof Error ? error.message : String(error)}</span>
    </div>
  ) : null;
}
export function Loading() {
  return (
    <div aria-label="Загрузка" className="space-y-4">
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-40 w-full" />
      <Skeleton className="h-32 w-full" />
    </div>
  );
}
export function Empty({
  title,
  text,
  action,
}: {
  title: string;
  text: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <SearchX size={32} />
      <h3>{title}</h3>
      <p>{text}</p>
      {action}
    </div>
  );
}
export function Busy({
  children,
  busy,
  ...props
}: React.ComponentProps<typeof Button> & { busy?: boolean }) {
  return (
    <Button {...props} disabled={busy || props.disabled}>
      {busy && <Loader2 size={16} className="animate-spin" />}
      {children}
    </Button>
  );
}
export function CheckField({
  children,
  checked,
  onChange,
}: {
  children: ReactNode;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  const id = useId();
  return (
    <label className="check-field" htmlFor={id}>
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{children}</span>
    </label>
  );
}
export function Verified({
  children = "Навыки проверены",
}: {
  children?: ReactNode;
}) {
  return (
    <span className="verified">
      <ShieldCheck size={14} />
      {children}
    </span>
  );
}
export function StepLink({
  to,
  n,
  title,
  text,
  done,
}: {
  to: string;
  n: string;
  title: string;
  text: string;
  done?: boolean;
}) {
  return (
    <Link to={to} className="step-link">
      <span className={done ? "step-number done" : "step-number"}>
        {done ? <Check size={20} /> : n}
      </span>
      <div>
        <h3>{title}</h3>
        <p>{text}</p>
      </div>
      <ArrowRight size={20} />
    </Link>
  );
}

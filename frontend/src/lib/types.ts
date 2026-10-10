export interface User {
  id: string;
  email: string;
  full_name: string;
  role: "candidate" | "employer";
  is_demo: boolean;
}
export interface Skill {
  score: number;
  families: number;
  checked_at: number;
  source: string;
  latest_checked_at?: number;
  rubric_version?: string;
}
export interface Candidate {
  id: string;
  name: string;
  headline: string;
  about: string;
  city: string;
  grade: string | null;
  specialization: string | null;
  category: string;
  work_format: string;
  employment: string;
  salary_min: number;
  salary_max: number;
  experience_years: number;
  stack: string[];
  skills: Record<string, Skill>;
  achievements: {
    title: string;
    year: number;
    source: string;
    is_demo: boolean;
  }[];
  evidence: {
    skill: string;
    family: string;
    score: number;
    checked_at: number;
    is_demo: boolean;
  }[];
  contacts: {
    full_name: string;
    email: string;
    phone: string;
    telegram: string;
  } | null;
  match_score: number;
  reasons: string[];
  limitations: string[];
  is_demo: boolean;
}
export interface PreferenceAnswer {
  left: Record<string, number>;
  right: Record<string, number>;
  choice: "left" | "right" | "skip";
}
export interface SearchInput {
  query: string;
  specialization: string;
  grade: string;
  work_format: string;
  skills: string[];
  salary_min: number;
  salary_max: number;
  fsp_only: boolean;
  title: string;
  ranking: {
    mode: "evidence" | "robust" | "preference" | "weighted" | "mixed";
    priorities: [string, string][];
    comparisons: PreferenceAnswer[];
    weights?: Record<string, number>;
  };
}
export interface SearchResult {
  candidates: Candidate[];
  total: number;
  normalized: SearchInput;
  notices: string[];
  algorithm: string;
  weighted: {
    considered: number;
    selected_count: number;
    weights: Record<string, number>;
    changed_count: number;
    model: string;
  } | null;
  robust: {
    status: "optimal" | "bounded" | "fallback";
    considered: number;
    selected_count: number;
    worst_loss_points: number;
    baseline_worst_loss_points: number;
    nominal_loss_points: number;
    optimization_gap_points: number;
    worst_weights: Record<string, number>;
    challenger_ids: string[];
    question: {
      skills: [string, string];
      worst_after_points: number;
      exact_choice: boolean;
    } | null;
    model: string;
  } | null;
  preference: {
    engine: string;
    status: "question" | "complete" | "not_needed" | "empty";
    considered: number;
    selected_count: number;
    answered: number;
    skipped: number;
    remaining: number;
    weights: Record<string, number>;
    changed_count: number;
    question: {
      left: Record<string, number>;
      right: Record<string, number>;
      expected_gain_points: number;
    } | null;
    model: string;
  } | null;
}
export type AssessmentAnswer = number | Record<string, string>;
export interface Rubric {
  version: string;
  pass_score: number;
  min_skill_score: number;
  min_practical_skill_score?: number;
  notice: string;
}
export interface AssessmentCatalog {
  question_count: number;
  practical_count: number;
  duration_minutes: number;
  challenge_count: number;
  challenge_duration_minutes: number;
  cooldown_days: number;
  rubric: Rubric;
}
export interface Question {
  id: string;
  title: string;
  skill: string;
  family: string;
  prompt: string;
  code?: string;
  choices: string[];
  answer_type: "choice" | "fields";
  fields?: { name: string; label: string; places: number }[];
  tables?: {
    name: string;
    columns: string[];
    rows: (string | number | null)[][];
  }[];
  correct_choice?: number;
  correct_values?: Record<string, string>;
  is_correct?: boolean;
  explanation?: string;
}
export interface Attempt {
  id: string;
  specialization: string;
  grade: string;
  kind: string;
  status: string;
  answers: Record<string, AssessmentAnswer>;
  rubric: Rubric;
  rubric_version: string;
  questions: Question[];
  result: {
    passed: boolean;
    score: number;
    skills: Record<string, number>;
    practical_skills?: Record<string, number>;
    grade_outcome: string;
    evidence_outcome?: string;
    current_grade: string;
    next_grade_change_at: number;
  };
  created_at: number;
  expires_at: number;
  is_demo: boolean;
}
export const specs: Record<string, string> = {
  backend: "Python backend",
  data: "Аналитика данных / SQL",
};
export const formats: Record<string, string> = {
  remote: "Удалённо",
  hybrid: "Гибрид",
  office: "В офисе",
};
export const statuses: Record<string, string> = {
  sent: "Отправлено",
  viewed: "Просмотрено",
  accepted: "Принято",
  declined: "Отклонено",
  withdrawn: "Отозвано",
  expired: "Срок истёк",
  interview: "Собеседование",
  in_progress: "В процессе",
  completed: "Завершено",
  cancelled: "Отменено",
};
export const rub = (n: number) => new Intl.NumberFormat("ru-RU").format(n);
export const salary = (lo: number, hi: number) =>
  hi ? `${rub(lo)} – ${rub(hi)} ₽` : "Ожидания не указаны";
export const date = (ts: number) =>
  ts ? new Date(ts * 1000).toLocaleDateString("ru-RU") : "—";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
export const session = {
  token: () => sessionStorage.getItem("fsp.access"),
  set: (token: string) => sessionStorage.setItem("fsp.access", token),
  clear: () => sessionStorage.removeItem("fsp.access"),
};
export async function api<T = any>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  if (
    !(options.body instanceof FormData) &&
    options.body &&
    !headers.has("Content-Type")
  )
    headers.set("Content-Type", "application/json");
  if (session.token())
    headers.set("Authorization", `Bearer ${session.token()}`);
  let response: Response;
  try {
    response = await fetch(`/api${path}`, { ...options, headers });
  } catch {
    throw new ApiError(
      "Сервер недоступен. Проверьте соединение и попробуйте снова.",
      0,
    );
  }
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    if (response.status === 401 && !path.startsWith("/auth/"))
      window.dispatchEvent(new Event("fsp:session-expired"));
    throw new ApiError(
      Array.isArray(error.detail)
        ? error.detail
            .map((e: any) => e.msg.replace("Value error, ", ""))
            .join("; ")
        : error.detail || "Не удалось выполнить действие.",
      response.status,
    );
  }
  return response.json();
}
export const post = <T = any>(path: string, data?: unknown) =>
  api<T>(path, { method: "POST", body: JSON.stringify(data ?? {}) });
export const put = <T = any>(path: string, data: unknown) =>
  api<T>(path, { method: "PUT", body: JSON.stringify(data) });
export async function download(path: string, filename: string) {
  const r = await fetch(`/api${path}`, {
    headers: { Authorization: `Bearer ${session.token()}` },
  });
  if (!r.ok)
    throw new Error(
      "Файл недоступен. Проверьте разрешение на просмотр профиля.",
    );
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

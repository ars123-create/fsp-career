import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, post, session } from "./api";
import type { User } from "./types";

const Auth = createContext<{
  user: User | null;
  loading: boolean;
  enter: (result: { access_token: string; user: User }) => void;
  logout: () => Promise<void>;
}>({ user: null, loading: true, enter: () => {}, logout: async () => {} });
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(!!session.token());
  const qc = useQueryClient();
  useEffect(() => {
    if (session.token())
      api<User>("/auth/me")
        .then(setUser)
        .catch(() => session.clear())
        .finally(() => setLoading(false));
    const expired = () => {
      session.clear();
      setUser(null);
      qc.clear();
    };
    window.addEventListener("fsp:session-expired", expired);
    return () => window.removeEventListener("fsp:session-expired", expired);
  }, [qc]);
  const enter = (result: { access_token: string; user: User }) => {
    qc.clear();
    session.set(result.access_token);
    setUser(result.user);
    setLoading(false);
  };
  const logout = async () => {
    try {
      await post("/auth/logout");
    } finally {
      session.clear();
      setUser(null);
      qc.clear();
    }
  };
  return (
    <Auth.Provider value={{ user, loading, enter, logout }}>
      {children}
    </Auth.Provider>
  );
}
export const useAuth = () => useContext(Auth);

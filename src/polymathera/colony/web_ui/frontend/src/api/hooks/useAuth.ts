import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../client";

export interface AuthResponse {
  user_id: string;
  username: string;
  tenant_id: string;
  default_colony_id: string | null;
  message: string;
}

export interface UserInfo {
  user_id: string;
  username: string;
  tenant_id: string;
  created_at: string | null;
  colonies: ColonyInfo[];
}

export interface ColonyInfo {
  colony_id: string;
  name: string;
  tenant_id: string;
  description: string;
  is_default: boolean;
  created_at: string | null;
}

export function useCurrentUser() {
  return useQuery({
    queryKey: ["auth", "me"],
    queryFn: () => apiFetch<UserInfo>("/auth/me"),
    retry: false,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchInterval: false,
  });
}

export function useSignup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { username: string; password: string }) =>
      apiFetch<AuthResponse>("/auth/signup", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["auth"] }),
  });
}

export function useLogin() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { username: string; password: string }) =>
      apiFetch<AuthResponse>("/auth/login", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["auth"] }),
  });
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<{ message: string }>("/auth/logout", { method: "POST" }),
    onSuccess: () => {
      qc.clear(); // Clear all cached data on logout
    },
  });
}

export function useColonies(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ["colonies"],
    queryFn: () => apiFetch<ColonyInfo[]>("/colonies/"),
    enabled: options?.enabled ?? true,
    retry: false,
  });
}

export function useCreateColony() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; description?: string }) =>
      apiFetch<ColonyInfo>("/colonies/", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["colonies"] });
      qc.invalidateQueries({ queryKey: ["auth", "me"] });
    },
  });
}

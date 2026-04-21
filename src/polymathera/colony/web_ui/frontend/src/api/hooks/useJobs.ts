import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type {
  JobSubmitRequest,
  JobSubmitResponse,
  JobStatusResponse,
} from "../types";

export function useSubmitJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: JobSubmitRequest) =>
      apiFetch<JobSubmitResponse>("/jobs/submit", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sessions"] });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["jobs", jobId],
    queryFn: () => apiFetch<JobStatusResponse>(`/jobs/${jobId}`),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed" || status === "cancelled") {
        return false; // Stop polling on terminal status
      }
      return 3000; // Poll every 3 seconds while active
    },
  });
}

export function useJobs() {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => apiFetch<JobStatusResponse[]>("/jobs/"),
    refetchInterval: 10000,
  });
}

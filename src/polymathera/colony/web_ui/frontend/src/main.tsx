import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider, MutationCache, QueryCache } from "@tanstack/react-query";
import App from "./App";
import "./index.css";
import { showErrorToast } from "./components/shared/ErrorToast";

const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error) => {
      showErrorToast(error.message);
    },
  }),
  mutationCache: new MutationCache({
    onError: (error) => {
      showErrorToast(error.message);
    },
  }),
  defaultOptions: {
    queries: {
      refetchInterval: 5000,
      staleTime: 3000,
      retry: 1,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>
);

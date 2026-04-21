import { AppShell } from "./components/layout/AppShell";
import { ErrorToastContainer } from "./components/shared/ErrorToast";

export default function App() {
  return (
    <>
      <AppShell />
      <ErrorToastContainer />
    </>
  );
}

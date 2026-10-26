import { useState, type ReactNode } from "react";
import { AlertCircle, LoaderCircle, CheckCircle2 } from "lucide-react";
export function ErrorBox({ message }: { message: string }) {
  return message ? (
    <div className="notice error" role="alert">
      <AlertCircle size={17} />
      <span>{message}</span>
    </div>
  ) : null;
}
export function Loading() {
  return (
    <div className="empty" role="status">
      <LoaderCircle className="spin" size={20} /> Loading workspace…
    </div>
  );
}
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
export function Badge({
  children,
  good = false,
}: {
  children: ReactNode;
  good?: boolean;
}) {
  return <span className={`badge ${good ? "good" : ""}`}>{children}</span>;
}
export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
export function useAction() {
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [success, setSuccess] = useState("");
  return {
    busy,
    error,
    success,
    run: async (fn: () => Promise<void>, message = "Done.") => {
      setBusy(true);
      setError("");
      setSuccess("");
      try {
        await fn();
        setSuccess(message);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Operation failed.");
      } finally {
        setBusy(false);
      }
    },
  };
}
export function Feedback({ action }: { action: ReturnType<typeof useAction> }) {
  return (
    <>
      <ErrorBox message={action.error} />
      {action.success && (
        <div className="notice success" role="status">
          <CheckCircle2 size={16} />
          {action.success}
        </div>
      )}
    </>
  );
}
export const engineName = (name: string) =>
  ({
    vectorforge_exact: "VectorForge Exact",
    vectorforge_ivf: "VectorForge IVF",
    faiss_flat: "FAISS Flat",
    faiss_ivf: "FAISS IVF",
  })[name] || name;

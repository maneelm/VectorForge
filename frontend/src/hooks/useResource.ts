import { useEffect, useRef, useState } from "react";
import { request } from "../api/client";
export function useResource<T>(path: string | null) {
  const previous = useRef<string | null>(null);
  const [data, setData] = useState<T | null>(null),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false),
    [version, setVersion] = useState(0);
  useEffect(() => {
    if (!path) {
      setData(null);
      setLoading(false);
      setError("");
      previous.current = null;
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    if (previous.current !== path) setData(null);
    previous.current = path;
    setError("");
    request<T>(path, "GET", undefined, controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setData(value);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [path, version]);
  return { data, error, loading, reload: () => setVersion((v) => v + 1) };
}

const base = import.meta.env.VITE_API_BASE || "/api/v1";
export async function request<T>(
  path: string,
  method = "GET",
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(base + path, {
      method,
      ...(body === undefined
        ? {}
        : {
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new Error(
      "Cannot reach the API. Start the backend or check the server connection, then try again.",
    );
  }
  if (response.status === 204) return undefined as T;
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      `The server returned an unreadable response (${response.status}).`,
    );
  }
  if (!response.ok) {
    const detail = data.error?.details
      ?.map(
        (d: { location: string[]; message: string }) =>
          `${d.location.join(".")}: ${d.message}`,
      )
      .join("; ");
    throw new Error(
      detail || data.error?.message || `Request failed (${response.status}).`,
    );
  }
  return data as T;
}
export function parseVectorInput(text: string) {
  const body = JSON.parse(text);
  if (
    Array.isArray(body.ids) &&
    body.ids.some(
      (id: unknown) => typeof id === "number" && !Number.isSafeInteger(id),
    )
  )
    throw new Error(
      "Use quoted decimal strings for IDs outside JavaScript’s safe integer range.",
    );
  return body;
}

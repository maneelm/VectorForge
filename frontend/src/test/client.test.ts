import { describe, it, expect, vi } from "vitest";
import { parseVectorInput, request } from "../api/client";
describe("API client", () => {
  it("preserves quoted int64 IDs and rejects rounded numeric IDs", () => {
    expect(parseVectorInput('{"ids":["9223372036854775807"]}').ids[0]).toBe(
      "9223372036854775807",
    );
    expect(() => parseVectorInput('{"ids":[9223372036854775807]}')).toThrow(
      "quoted",
    );
  });
  it("shows useful API validation errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({ error: { message: "Duplicate IDs" } }),
            { status: 409 },
          ),
        ),
    );
    await expect(request("/indexes", "POST", {})).rejects.toThrow(
      "Duplicate IDs",
    );
  });
  it("handles unreachable backend", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network")));
    await expect(request("/health")).rejects.toThrow("Start the backend");
  });
  it("handles empty DELETE success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 204 })),
    );
    await expect(request("/indexes/test", "DELETE")).resolves.toBeUndefined();
  });
});

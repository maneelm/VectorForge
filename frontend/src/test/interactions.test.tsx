import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect, vi } from "vitest";
import { Indexes } from "../pages/Indexes";
import { Workspace } from "../pages/Workspace";
import { BenchmarkCharts } from "../components/BenchmarkCharts";
import * as client from "../api/client";
const index = {
  id: "test",
  name: "Research",
  type: "exact",
  dimension: 3,
  metric: "cosine",
  size: 0,
  is_trained: true,
  status: "ready",
  configuration: {},
};
describe("real API interactions", () => {
  it("submits a validated create-index payload", async () => {
    const call = vi
      .spyOn(client, "request")
      .mockImplementation(async (path, method) =>
        method === "POST" ? index : ([] as never),
      );
    render(
      <MemoryRouter>
        <Indexes />
      </MemoryRouter>,
    );
    await userEvent.type(screen.getByLabelText("Index name"), "Research");
    await userEvent.click(screen.getByRole("button", { name: "Create index" }));
    await waitFor(() =>
      expect(call).toHaveBeenCalledWith("/indexes", "POST", {
        name: "Research",
        type: "exact",
        dimension: 3,
        metric: "cosine",
      }),
    );
    expect(await screen.findByText("Index created.")).toBeInTheDocument();
  });
  it("inserts vectors and displays a real search response", async () => {
    const call = vi
      .spyOn(client, "request")
      .mockImplementation(async (path, method) => {
        if (path === "/indexes") return [index] as never;
        if (path.endsWith("/search"))
          return {
            matches: [{ id: "101", score: 1, rank: 1 }],
            latency_ms: 0.25,
            metric: "cosine",
            nprobe: null,
            latency_scope: "API-measured engine-call latency",
          } as never;
        return index as never;
      });
    render(
      <MemoryRouter>
        <Workspace />
      </MemoryRouter>,
    );
    await screen.findByText("0 vectors");
    await userEvent.click(
      screen.getByRole("button", { name: "Insert vectors" }),
    );
    await waitFor(() =>
      expect(call).toHaveBeenCalledWith(
        "/indexes/test/vectors",
        "POST",
        expect.objectContaining({ ids: ["101", "102", "103"] }),
      ),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Search vectors" }),
    );
    expect(await screen.findByText("101")).toBeInTheDocument();
    expect(screen.getByText("1.00000000")).toBeInTheDocument();
    expect(screen.getByText(/API-measured engine-call/)).toBeInTheDocument();
  });
  it("reveals IVF-only configuration", () => {
    vi.spyOn(client, "request").mockResolvedValue([] as never);
    render(
      <MemoryRouter>
        <Indexes />
      </MemoryRouter>,
    );
    fireEvent.change(screen.getByLabelText("Index type"), {
      target: { value: "ivf" },
    });
    expect(screen.getByLabelText("Clusters (nlist)")).toBeInTheDocument();
    expect(screen.getByLabelText("Probes (nprobe)")).toBeInTheDocument();
  });
  it("renders all six benchmark visualizations", () => {
    render(<BenchmarkCharts rows={[]} />);
    for (const text of [
      "Search latency · mean ms",
      "Throughput · queries / second",
      "Recall versus latency",
      "Recall versus nprobe",
      "Index build · milliseconds",
      "Memory · process RSS MiB",
    ])
      expect(screen.getByRole("heading", { name: text })).toBeInTheDocument();
  });
});

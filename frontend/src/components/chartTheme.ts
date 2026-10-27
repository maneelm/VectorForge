// Stable engine identities across charts, legends, and the comparison table.
export const engineColors: Record<string, string> = {
  vectorforge_exact: "#8bbcf7",
  vectorforge_ivf: "#91d5aa",
  faiss_flat: "#c5a4ed",
  faiss_ivf: "#efc47e",
};
export const engineColor = (engine: string) =>
  engineColors[engine] ?? "#a3b0bf";
export const shortEngineName = (engine: string) =>
  ({
    vectorforge_exact: "VF Exact",
    vectorforge_ivf: "VF IVF",
    faiss_flat: "FAISS Flat",
    faiss_ivf: "FAISS IVF",
  })[engine] ?? engine;
export const chartNumber = (value: unknown) =>
  typeof value === "number"
    ? value.toLocaleString(undefined, { maximumSignificantDigits: 4 })
    : String(value ?? "N/A");

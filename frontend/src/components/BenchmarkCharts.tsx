import type { CSSProperties, ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Result } from "../types";
import { Empty, engineName } from "./UI";
import { chartNumber, engineColor, shortEngineName } from "./chartTheme";

const tick = {
  fill: "#a3b0bf",
  fontSize: 12,
  fontFamily: "ui-sans-serif, system-ui, sans-serif",
};
const tooltipProps = {
  contentStyle: {
    background: "#202731",
    border: "1px solid #526173",
    borderRadius: 5,
    color: "#e6ebf1",
  },
  labelStyle: { color: "#e6ebf1", marginBottom: 5 },
  itemStyle: { color: "#e6ebf1" },
  cursor: { fill: "#ffffff06" },
  formatter: (value: unknown) => chartNumber(value),
};
function Chart({
  title,
  description,
  children,
  height,
}: {
  title: string;
  description: string;
  children: ReactNode;
  height?: number;
}) {
  return (
    <section className="panel chart-panel" aria-label={title}>
      <h2>{title}</h2>
      <p className="chart-description">{description}</p>
      <div className="chart" style={height ? { height } : undefined}>
        {children}
      </div>
    </section>
  );
}

export function BenchmarkCharts({ rows }: { rows: Result[] }) {
  const good = rows.filter((row) => row.status === "ok");
  const data = good.map((row) => ({
    ...row,
    label:
      shortEngineName(row.engine) +
      (row.nprobe !== null
        ? ` / ${row.nprobe}${row.full_probe_direct_path ? "*" : ""}`
        : ""),
    fullLabel:
      engineName(row.engine) +
      (row.nprobe !== null
        ? ` / nprobe=${row.nprobe}${row.full_probe_direct_path ? "*" : ""}`
        : ""),
  }));
  // Build/RSS values repeat for each probe setting; show one record per engine.
  const build = Object.values(
    Object.fromEntries(
      data.map((row) => [
        row.engine,
        {
          ...row,
          label: shortEngineName(row.engine),
          fullLabel: engineName(row.engine),
          baseline:
            row.rss_baseline_bytes === null
              ? null
              : row.rss_baseline_bytes / 2 ** 20,
          retained:
            row.retained_rss_delta_bytes === null
              ? null
              : row.retained_rss_delta_bytes / 2 ** 20,
        },
      ]),
    ),
  );
  const engines = [...new Set(good.map((row) => row.engine))];
  const barHeight = Math.max(300, data.length * 26 + 45);
  const bars = (field: "latency_mean_ms" | "qps", name: string) => (
    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 0, right: 24, bottom: 15, left: 0 }}
      >
        <CartesianGrid stroke="#303944" horizontal={false} />
        <XAxis
          type="number"
          domain={[0, "auto"]}
          tick={tick}
          tickFormatter={chartNumber}
        />
        <YAxis
          type="category"
          dataKey="label"
          width={128}
          tick={tick}
          interval={0}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip
          {...tooltipProps}
          labelFormatter={(_, payload) => payload[0]?.payload.fullLabel ?? ""}
        />
        <Bar
          dataKey={field}
          name={name}
          maxBarSize={16}
          isAnimationActive={false}
        >
          {data.map((row) => (
            <Cell
              key={row.engine + ":" + row.nprobe}
              fill={engineColor(row.engine)}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
  return (
    <>
      <div className="chart-legend" aria-label="Engine colors">
        {engines.map((engine) => (
          <span
            key={engine}
            style={{ "--engine-color": engineColor(engine) } as CSSProperties}
          >
            <i aria-hidden="true" />
            {engineName(engine)}
          </span>
        ))}
      </div>
      <div className="charts-grid">
        <Chart
          title="Search latency · mean ms"
          description="Lower is better. Row suffix indicates nprobe; * marks direct full probing."
          height={barHeight}
        >
          {bars("latency_mean_ms", "Mean latency (ms)")}
        </Chart>
        <Chart
          title="Throughput · queries / second"
          description="Higher is better. Measured workload throughput, not inverse latency."
          height={barHeight}
        >
          {bars("qps", "Throughput (queries/s)")}
        </Chart>
        <Chart
          title="Recall versus latency"
          description="Top left indicates higher recall with lower latency. Each point is a configuration."
        >
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <ScatterChart margin={{ top: 10, bottom: 28, left: 0, right: 24 }}>
              <CartesianGrid stroke="#303944" />
              <XAxis
                type="number"
                dataKey="latency_mean_ms"
                domain={[0, "auto"]}
                tick={tick}
                tickFormatter={chartNumber}
                name="Mean latency"
                unit=" ms"
                label={{
                  value: "Mean latency (ms)",
                  position: "bottom",
                  offset: 10,
                  ...tick,
                }}
              />
              <YAxis
                type="number"
                dataKey="recall_at_k"
                domain={[0, 1]}
                tick={tick}
                width={48}
                name="Recall@k"
              />
              <Tooltip
                content={({ active, payload }) => {
                  const point = payload?.[0]?.payload;
                  return active && point ? (
                    <div className="chart-tooltip">
                      <strong>{point.fullLabel}</strong>
                      <p>
                        Mean latency: {chartNumber(point.latency_mean_ms)} ms
                      </p>
                      <p>Recall@k: {chartNumber(point.recall_at_k)}</p>
                    </div>
                  ) : null;
                }}
              />
              {engines.map((engine) => (
                <Scatter
                  key={engine}
                  name={engineName(engine)}
                  data={data.filter((row) => row.engine === engine)}
                  fill={engineColor(engine)}
                  isAnimationActive={false}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </Chart>
        <Chart
          title="Recall versus nprobe"
          description="Recall@k on a fixed 0–1 scale. Increasing probes searches more clusters."
        >
          {data.some((row) => row.nprobe !== null) ? (
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <LineChart margin={{ top: 10, bottom: 28, left: 0, right: 24 }}>
                <CartesianGrid stroke="#303944" />
                <XAxis
                  type="number"
                  dataKey="nprobe"
                  allowDuplicatedCategory={false}
                  tick={tick}
                  name="nprobe"
                  label={{
                    value: "nprobe (clusters searched)",
                    position: "bottom",
                    offset: 10,
                    ...tick,
                  }}
                />
                <YAxis domain={[0, 1]} tick={tick} width={48} />
                <Tooltip
                  {...tooltipProps}
                  labelFormatter={(label) => `nprobe=${label}`}
                />
                {engines
                  .filter((engine) => engine.endsWith("ivf"))
                  .map((engine) => (
                    <Line
                      key={engine}
                      data={data
                        .filter((row) => row.engine === engine)
                        .sort((a, b) => a.nprobe! - b.nprobe!)}
                      dataKey="recall_at_k"
                      name={engineName(engine)}
                      stroke={engineColor(engine)}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      connectNulls={false}
                      isAnimationActive={false}
                    />
                  ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <Empty>Probe comparisons apply to IVF indexes.</Empty>
          )}
        </Chart>
        <Chart
          title="Index build · milliseconds"
          description="Stacked construction, training, and insertion times. Lower total is better."
        >
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <BarChart
              data={build}
              layout="vertical"
              margin={{ right: 24, bottom: 10 }}
            >
              <CartesianGrid stroke="#303944" horizontal={false} />
              <XAxis
                type="number"
                domain={[0, "auto"]}
                tick={tick}
                tickFormatter={chartNumber}
              />
              <YAxis
                type="category"
                dataKey="label"
                width={88}
                tick={tick}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                {...tooltipProps}
                labelFormatter={(_, payload) =>
                  payload[0]?.payload.fullLabel ?? ""
                }
              />
              <Legend iconSize={9} />
              {[
                ["construction_ms", "Construction", "#d1d8e1"],
                ["training_ms", "Training", "#8e9fb3"],
                ["insertion_ms", "Insertion", "#52667d"],
              ].map(([field, name, fill]) => (
                <Bar
                  key={field}
                  dataKey={field}
                  name={name + " (ms)"}
                  stackId="build"
                  fill={fill}
                  maxBarSize={24}
                  isAnimationActive={false}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Chart>
        <Chart
          title="Memory · process RSS MiB"
          description="Baseline and retained delta shown separately. Delta is an estimate, not peak memory."
        >
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <BarChart
              data={build}
              layout="vertical"
              margin={{ right: 24, bottom: 10 }}
            >
              <CartesianGrid stroke="#303944" horizontal={false} />
              <XAxis
                type="number"
                domain={[(min: number) => Math.min(0, min), "auto"]}
                tick={tick}
                tickFormatter={chartNumber}
              />
              <YAxis
                type="category"
                dataKey="label"
                width={88}
                tick={tick}
                tickLine={false}
                axisLine={false}
              />
              <ReferenceLine x={0} stroke="#526173" />
              <Tooltip
                {...tooltipProps}
                labelFormatter={(_, payload) =>
                  payload[0]?.payload.fullLabel ?? ""
                }
              />
              <Legend iconSize={9} />
              <Bar
                dataKey="baseline"
                name="Baseline (MiB)"
                fill="#8e9fb3"
                maxBarSize={16}
                isAnimationActive={false}
              />
              <Bar
                dataKey="retained"
                name="Retained delta (MiB)"
                fill="#d1d8e1"
                maxBarSize={16}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </Chart>
      </div>
    </>
  );
}

import { useState } from "react";
import { LineChart, type Series } from "./Chart";
import { type MetricHistory, type NodeDetailInfo, api } from "./api";
import { Badge, Icon, Meter, STATUS_LABEL, Stat, duration, gb, timeAgo, usePolling } from "./ui";

const RANGES = [
  { id: "1h", label: "1시간" },
  { id: "6h", label: "6시간" },
  { id: "24h", label: "24시간" },
  { id: "7d", label: "7일" },
  { id: "30d", label: "30일" },
];
const STEP_LABEL: Record<string, string> = { "1m": "1분", "10m": "10분", "1h": "1시간" };

function uptime(seconds?: number): string {
  if (!seconds) return "–";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  return d > 0 ? `${d}일 ${h}시간` : `${h}시간 ${Math.floor((seconds % 3600) / 60)}분`;
}

function size(gbValue?: number | null): string {
  if (gbValue == null) return "–";
  return gbValue >= 1000 ? `${(gbValue / 1024).toFixed(1)} TB` : `${Math.round(gbValue)} GB`;
}

function occupant(state: NodeDetailInfo["gpu_states"][number] | undefined): string {
  if (!state || state.state === "free") return "비어 있음";
  if (state.state === "job") return "BADAMOA 작업";
  return state.users.length ? state.users.join(", ") : "다른 프로세스";
}

export function NodeDetail({ nodeId }: { nodeId: string }) {
  const [range, setRange] = useState("1h");
  const [mode, setMode] = useState<"avg" | "max">("avg");
  const { data: node, error } = usePolling(() => api<NodeDetailInfo>(`/nodes/${nodeId}`), 5000, [nodeId]);
  const { data: history } = usePolling(
    () => api<MetricHistory>(`/nodes/${nodeId}/metrics?range=${range}`),
    range === "1h" || range === "6h" ? 15000 : 60000,
    [nodeId, range],
  );

  if (error && !node) return <div className="error">{error}</div>;
  if (!node) return <div className="muted">불러오는 중…</div>;

  const m = node.metrics;
  const gpus = m.gpus ?? [];
  const gpuUtil = gpus.length ? gpus.reduce((s, g) => s + (g.utilization_pct ?? 0), 0) / gpus.length : null;
  const gpuUsed = gpus.reduce((s, g) => s + (g.memory_used_mb ?? 0), 0);
  const gpuTotal = gpus.reduce((s, g) => s + (g.memory_total_mb ?? 0), 0);
  const gpuPower = gpus.reduce((s, g) => s + (g.power_w ?? 0), 0);
  const hasGpuPower = gpus.some((g) => g.power_w != null);
  const totalPower = hasGpuPower || m.cpu_power_w != null ? gpuPower + (m.cpu_power_w ?? 0) : null;
  const memPct = m.memory_total_mb ? Math.round(((m.memory_used_mb ?? 0) / m.memory_total_mb) * 100) : null;

  const pick = (key: string, label: string): Series | null => {
    const values = history?.series[key]?.[mode];
    return values ? { key, label, values } : null;
  };
  const group = (prefix: string, label: (name: string) => string): Series[] =>
    Object.keys(history?.series ?? {})
      .filter((k) => k.startsWith(prefix))
      .sort()
      .map((k) => ({ key: k, label: label(k.slice(prefix.length)), values: history!.series[k][mode] }));
  const gpuSeries = (suffix: string): Series[] =>
    node.info.gpus
      .map((g) => pick(`gpu${g.index}_${suffix}`, `GPU ${g.index}`))
      .filter((s): s is Series => s !== null);
  const only = (...items: (Series | null)[]) => items.filter((s): s is Series => s !== null);

  const sub = history ? `${RANGES.find((r) => r.id === range)?.label} · ${STEP_LABEL[history.resolution]} ${mode === "avg" ? "평균" : "최대"}` : "";
  const charts: { title: string; unit: string; yMax?: number; series: Series[] }[] = [
    { title: "CPU · 메모리 사용률", unit: "%", yMax: 100, series: only(pick("cpu", "CPU"), pick("mem", "메모리")) },
    { title: "GPU 사용률", unit: "%", yMax: 100, series: gpuSeries("util") },
    { title: "GPU 메모리 사용률", unit: "%", yMax: 100, series: gpuSeries("mem") },
    { title: "GPU 온도", unit: "°C", series: gpuSeries("temp") },
    { title: "전력", unit: "W", series: only(pick("power", "합계"), pick("cpu_power", "CPU"), pick("gpu_power", "GPU")) },
    { title: "디스크 사용률", unit: "%", yMax: 100, series: group("disk_pct:", (n) => n) },
    { title: "디스크 I/O 사용률", unit: "%", yMax: 100, series: group("io_busy:", (n) => n) },
    { title: "디스크 읽기", unit: "MB/s", series: group("io_read:", (n) => n) },
    { title: "디스크 쓰기", unit: "MB/s", series: group("io_write:", (n) => n) },
  ].filter((c) => c.series.length > 0);

  const forecast = Object.fromEntries(node.disk_forecast.map((f) => [f.mount, f]));
  const states = Object.fromEntries(node.gpu_states.map((s) => [s.index, s]));

  return (
    <>
      <a className="link back" href="#/nodes">
        ← 노드 목록
      </a>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            {node.name} <Badge kind={node.status}>{STATUS_LABEL[node.status]}</Badge>
          </h1>
          <p className="muted">
            {node.info.hostname} · {node.info.os_release ?? node.os} · 에이전트 v{node.agent_version} · 업타임{" "}
            {uptime(m.uptime_s)} · 수집 {timeAgo(node.last_seen_at)}
          </p>
        </div>
      </div>

      <div className="stats five">
        <Stat
          icon="chip"
          label="CPU"
          value={<>{Math.round(m.cpu_pct ?? 0)}<small>%</small></>}
          sub={[
            `${node.info.cpu_count ?? "?"}코어`,
            m.load ? `load ${m.load[0]}` : null,
            m.cpu_temp_c != null ? `${m.cpu_temp_c}°C` : null,
            m.cpu_power_w != null ? `${m.cpu_power_w}W` : null,
          ]
            .filter(Boolean)
            .join(" · ")}
        />
        <Stat
          icon="server"
          label="메모리"
          value={<>{memPct ?? "–"}<small>%</small></>}
          sub={`${gb(m.memory_used_mb)} / ${gb(m.memory_total_mb)}`}
        />
        <Stat
          icon="activity"
          label="GPU 사용률"
          value={gpuUtil == null ? "–" : <>{Math.round(gpuUtil)}<small>%</small></>}
          sub={gpus.length ? `${gpus.length}개 평균` : "GPU 없음"}
        />
        <Stat
          icon="database"
          label="GPU 메모리"
          value={gpuTotal ? <>{Math.round((gpuUsed / gpuTotal) * 100)}<small>%</small></> : "–"}
          sub={gpuTotal ? `${(gpuUsed / 1024).toFixed(1)} / ${Math.round(gpuTotal / 1024)} GB` : " "}
        />
        <Stat
          icon="alert"
          label="전력"
          value={totalPower == null ? "–" : <>{Math.round(totalPower)}<small>W</small></>}
          sub={totalPower == null ? "수집되지 않음" : `CPU ${m.cpu_power_w ?? "–"}W · GPU ${hasGpuPower ? gpuPower : "–"}W`}
        />
      </div>

      <div className="toolbar">
        <div className="segmented" role="group" aria-label="값">
          <span className="eyebrow">값</span>
          {(["avg", "max"] as const).map((v) => (
            <button key={v} className={mode === v ? "active" : ""} onClick={() => setMode(v)} aria-pressed={mode === v}>
              {v === "avg" ? "평균" : "최대"}
            </button>
          ))}
        </div>
        <div className="segmented" role="group" aria-label="기간">
          <span className="eyebrow">기간</span>
          {RANGES.map((r) => (
            <button key={r.id} className={range === r.id ? "active" : ""} onClick={() => setRange(r.id)} aria-pressed={range === r.id}>
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {history && (
        <div className="chart-grid">
          {charts.map((c) => (
            <LineChart
              key={c.title}
              title={c.title}
              subtitle={sub}
              unit={c.unit}
              yMax={c.yMax}
              timestamps={history.timestamps}
              from={history.from}
              to={history.to}
              stepSeconds={history.step_s}
              series={c.series}
            />
          ))}
        </div>
      )}

      <section className="card table-wrap">
        <div className="card-head">
          <h3>사용자별 사용량</h3>
          <span className="muted small">직접 접속해 실행 중인 프로세스 포함</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>사용자</th>
              <th>상태</th>
              <th className="right">CPU</th>
              <th className="right">메모리</th>
              <th className="right">GPU</th>
              <th className="right">GPU 메모리</th>
              <th className="right">프로세스</th>
            </tr>
          </thead>
          <tbody>
            {(m.users ?? []).length === 0 && (
              <tr>
                <td colSpan={7} className="empty">
                  수집된 사용자 정보가 없습니다.
                </td>
              </tr>
            )}
            {(m.users ?? []).slice(0, 12).map((u) => (
              <tr key={u.user}>
                <td>
                  <strong>{u.user}</strong>
                </td>
                <td>
                  <Badge kind={u.active ? "running" : "idle"}>{u.active ? "작업 중" : "대기"}</Badge>
                </td>
                <td className="right num">{u.cpu_cores.toFixed(1)}코어</td>
                <td className="right num">{(u.memory_mb / 1024).toFixed(1)} GB</td>
                <td className="right num">{u.gpus.length ? u.gpus.map((g) => `#${g}`).join(", ") : "–"}</td>
                <td className="right num">{u.gpu_memory_mb ? `${(u.gpu_memory_mb / 1024).toFixed(1)} GB` : "–"}</td>
                <td className="right num">{u.processes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {node.info.gpus.length > 0 && (
        <section className="card table-wrap">
          <div className="card-head">
            <h3>GPU 장치</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>GPU</th>
                <th>모델</th>
                <th>사용률</th>
                <th className="right">메모리</th>
                <th className="right">온도</th>
                <th className="right">전력</th>
                <th>사용 중</th>
              </tr>
            </thead>
            <tbody>
              {node.info.gpus.map((gpu) => {
                const live = gpus.find((g) => g.index === gpu.index);
                const state = states[gpu.index];
                return (
                  <tr key={gpu.index}>
                    <td className="num">#{gpu.index}</td>
                    <td>{gpu.name}</td>
                    <td className="meter-cell">
                      <Meter value={live?.utilization_pct} max={100} />
                    </td>
                    <td className="right num">
                      {live ? `${((live.memory_used_mb ?? 0) / 1024).toFixed(1)} / ${gb(live.memory_total_mb)}` : "–"}
                    </td>
                    <td className="right num">{live?.temperature_c != null ? `${live.temperature_c}°C` : "–"}</td>
                    <td className="right num">
                      {live?.power_w != null ? `${live.power_w}${live.power_limit_w ? ` / ${live.power_limit_w}` : ""}W` : "–"}
                    </td>
                    <td>
                      <span className={`occupant ${state?.state ?? "free"}`}>
                        <i />
                        {state?.state === "job" && state.job_id ? "BADAMOA 작업" : occupant(state)}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      <section className="card table-wrap">
        <div className="card-head">
          <h3>디스크</h3>
          <span className="muted small">증감과 예상은 기록이 반나절 이상 쌓이면 표시됩니다</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>마운트</th>
              <th>장치</th>
              <th>파일시스템</th>
              <th>사용률</th>
              <th className="right">사용</th>
              <th className="right">남음</th>
              <th className="right">전체</th>
              <th className="right">7일 증감</th>
              <th className="right">가득 참 예상</th>
            </tr>
          </thead>
          <tbody>
            {(m.disks ?? []).map((d) => {
              const f = forecast[d.mount];
              return (
                <tr key={d.mount}>
                  <td className="mono">{d.mount}</td>
                  <td className="mono muted">{d.device}</td>
                  <td className="muted">{d.fstype}</td>
                  <td className="meter-cell">
                    <Meter value={d.pct} max={100} />
                  </td>
                  <td className="right num">{size(d.used_gb)}</td>
                  <td className="right num">{size(d.free_gb)}</td>
                  <td className="right num">{size(d.total_gb)}</td>
                  <td className="right num">
                    {f?.delta_7d_gb == null ? "–" : `${f.delta_7d_gb > 0 ? "+" : ""}${f.delta_7d_gb} GB`}
                  </td>
                  <td className="right num">
                    {f?.delta_7d_gb == null ? "–" : f.days_until_full == null ? "늘지 않음" : `약 ${f.days_until_full}일`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <div className="two-col">
        <section className="card table-wrap">
          <div className="card-head">
            <h3>물리 디스크</h3>
            <span className="muted small">현재 I/O</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>디스크</th>
                <th>모델</th>
                <th>종류</th>
                <th className="right">읽기</th>
                <th className="right">쓰기</th>
              </tr>
            </thead>
            <tbody>
              {(m.physical_disks ?? []).map((d) => (
                <tr key={d.name}>
                  <td className="mono">{d.name}</td>
                  <td>{d.model ?? <span className="muted">–</span>}</td>
                  <td>{d.kind ?? <span className="muted">–</span>}</td>
                  <td className="right num">{d.read_mbps == null ? "–" : `${d.read_mbps.toFixed(1)} MB/s`}</td>
                  <td className="right num">{d.write_mbps == null ? "–" : `${d.write_mbps.toFixed(1)} MB/s`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="card table-wrap">
          <div className="card-head">
            <h3>로그인 세션</h3>
          </div>
          <table>
            <thead>
              <tr>
                <th>사용자</th>
                <th>터미널</th>
                <th>접속 위치</th>
                <th className="right">접속 시각</th>
              </tr>
            </thead>
            <tbody>
              {(m.sessions ?? []).length === 0 && (
                <tr>
                  <td colSpan={4} className="empty">
                    접속한 사용자가 없습니다.
                  </td>
                </tr>
              )}
              {(m.sessions ?? []).map((s, i) => (
                <tr key={i}>
                  <td>
                    <strong>{s.user ?? "–"}</strong>
                  </td>
                  <td className="mono muted">{s.terminal ?? "–"}</td>
                  <td className="mono muted">{s.host ?? "–"}</td>
                  <td className="right muted">{new Date(s.started * 1000).toLocaleString("ko-KR", { hour12: false })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>

      <section className="card table-wrap">
        <div className="card-head">
          <h3>이 노드에서 실행한 작업</h3>
          <a className="link" href="#/jobs">
            전체 보기 <Icon name="arrow" size={14} />
          </a>
        </div>
        <table>
          <tbody>
            {node.recent_jobs.length === 0 && (
              <tr>
                <td className="empty">아직 실행한 작업이 없습니다.</td>
              </tr>
            )}
            {node.recent_jobs.map((job) => (
              <tr key={job.id}>
                <td>
                  <strong>{job.title}</strong> <span className="mono muted small">{job.type}</span>
                </td>
                <td>
                  <Badge kind={job.status}>{STATUS_LABEL[job.status]}</Badge>
                </td>
                <td className="muted">{job.gpu_indices.length ? `GPU ${job.gpu_indices.join(", ")}` : "GPU 없음"}</td>
                <td className="muted">{duration(job.duration_s)}</td>
                <td className="muted right">{timeAgo(job.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}

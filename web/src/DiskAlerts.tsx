import { useState } from "react";
import { LogTail } from "./Training";
import { type Job, api } from "./api";
import { Badge, CopyBlock, Modal, STATUS_LABEL, usePolling } from "./ui";

/** A disk BADAMOA writes to on a server that is nearly full (the Hub's ``disk_space``). */
export interface DiskAlert {
  mount: string;
  free_gb: number | null;
  total_gb: number | null;
  level: "low" | "full";
  uses: ("home" | "work" | "images" | "docker")[];
  label: string; // what is on it, in words
  paths: Record<string, string>;
  docker?: string; // Docker's data-root, when Docker's images are on this disk
}

function room(gbValue: number | null): string {
  if (gbValue == null) return "?";
  return gbValue >= 1000 ? `${(gbValue / 1024).toFixed(1)} TB` : `${gbValue.toFixed(gbValue < 10 ? 1 : 0)} GB`;
}

function line(a: DiskAlert): string {
  return `디스크 ${a.mount} ${a.level === "full" ? "가득 참" : "거의 참"} · ${room(a.free_gb)} 남음`;
}

/** Under a server's name on the board: the fullest disk BADAMOA writes to, the rest on hover. */
export function DiskAlertLine({ alerts }: { alerts?: DiskAlert[] }) {
  if (!alerts || alerts.length === 0) return null;
  const first = alerts[0];
  const title = alerts.map((a) => `${line(a)} / ${room(a.total_gb)} — ${a.label}`).join("\n");
  return (
    <div className={`small ${first.level === "full" ? "bad-text" : "warn-text"}`} title={title}>
      {line(first)}
      {alerts.length > 1 ? ` 외 ${alerts.length - 1}개` : ""}
    </div>
  );
}

const FIX: Record<string, string> = {
  home: "MOA의 설정 · 기록이 있는 곳입니다. 다른 파일을 정리해 공간을 비우세요",
  work: "가상환경 · 작업 폴더 · 받아 둔 가중치를 두는 곳입니다. 아래 저장 공간에서 지우거나 노드 설정에서 작업 폴더를 여유 있는 디스크로 바꾸세요",
  images: "다른 서버에서 받은 이미지를 두는 곳입니다. 아래 '받아 둔 버전'에서 지우거나 노드 설정에서 보관 위치를 바꾸세요",
  docker: "Docker 이미지가 있는 곳입니다. 아래 'Docker 이미지 위치 옮기기'로 다른 디스크로 옮기면 비워집니다",
};

/** How an admin moves Docker's images to a roomier disk (``data-root``); BADAMOA never does it itself. */
function dockerMove(current: string, target: string): string {
  const to = target || "<여유 있는 디스크>/docker"; // a placeholder that cannot run as it is
  if (current.includes("/.local/share/docker")) {
    // rootless Docker: the account's own daemon and settings
    return [
      "systemctl --user stop docker",
      `mkdir -p ${to}`,
      `rsync -aHAX ${current}/ ${to}/`,
      `python3 -c 'import json,pathlib;p=pathlib.Path.home()/".config/docker/daemon.json";p.parent.mkdir(parents=True,exist_ok=True);c=json.loads(p.read_text()) if p.exists() else {};c["data-root"]="${to}";p.write_text(json.dumps(c,indent=2))'`,
      "systemctl --user start docker",
      "docker info --format '{{.DockerRootDir}}'",
    ].join("\n");
  }
  return [
    "sudo systemctl stop docker docker.socket",
    `sudo mkdir -p ${to}`,
    `sudo rsync -aHAX ${current}/ ${to}/`,
    `sudo python3 -c 'import json,pathlib;p=pathlib.Path("/etc/docker/daemon.json");c=json.loads(p.read_text()) if p.exists() else {};c["data-root"]="${to}";p.write_text(json.dumps(c,indent=2))'`,
    "sudo systemctl start docker",
    "docker info --format '{{.DockerRootDir}}'",
  ].join("\n");
}

/** On the server's page: each nearly full disk, what BADAMOA keeps there and what to do; for Docker, how to move it. */
export function DiskAlertBox({
  alerts,
  dockerTarget,
  isAdmin,
  nodeId,
}: {
  alerts?: DiskAlert[];
  dockerTarget?: string;
  isAdmin: boolean;
  nodeId: string;
}) {
  const [moving, setMoving] = useState(false);
  if (!alerts || alerts.length === 0) return null;
  return (
    <section className={`card disk-alerts ${alerts.some((a) => a.level === "full") ? "full" : ""}`}>
      <h3>디스크 공간 부족</h3>
      {alerts.map((a) => (
        <div key={a.mount} className="disk-alert">
          <div>
            <strong className={a.level === "full" ? "bad-text" : "warn-text"}>
              디스크 {a.mount} · {room(a.free_gb)} 남음
            </strong>{" "}
            <span className="muted">/ {room(a.total_gb)} · {a.label}</span>
          </div>
          <ul className="muted small">
            {a.uses.map((u) => (
              <li key={u}>
                <span className="mono">{a.paths[u]}</span>: {FIX[u]}
              </li>
            ))}
          </ul>
          {a.docker && isAdmin && (
            <div className="docker-move-actions">
              <button onClick={() => setMoving(true)}>Docker 이미지 위치 옮기기</button>
              <span className="muted small">MOA가 이 서버에서 멈춤 → 복사 → 설정 → 시작 → 확인까지 하고, 실패하면 되돌립니다.</span>
            </div>
          )}
          {a.docker && isAdmin && (
            <details className="docker-move">
              <summary>Docker 이미지 폴더를 여유 있는 디스크로 옮기는 법</summary>
              <p className="muted small">
                Docker의 <code>data-root</code>를 바꿉니다(<code>/etc/docker/daemon.json</code>). 멈추는 동안 이 서버의 컨테이너가 모두
                멈추니 학습이 없을 때 하세요. 새 위치가 나오고 학습이 잘 되면 예전 폴더 <code>{a.docker}</code>를 지워 공간을 비웁니다.
              </p>
              <CopyBlock label="그 서버에서" text={dockerMove(a.docker, dockerTarget ?? "")} />
            </details>
          )}
        </div>
      ))}
      {moving && <DockerMoveDialog nodeId={nodeId} suggest={dockerTarget ?? ""} onClose={() => setMoving(false)} />}
    </section>
  );
}

interface DockerMoveChoice {
  folder: string; // <mount>/docker
  mount: string;
  free_gb: number | null;
  total_gb: number | null;
}

interface DockerMoveState {
  path: string; // Docker's data-root now
  suggest: string; // a folder on the disk with the most room
  choices?: DockerMoveChoice[]; // each other disk's docker folder, the roomiest first
  here?: { mount: string; free_gb: number | null; total_gb: number | null } | null; // the disk it is on now
  possible: boolean;
  why: string;
}

const TYPED = "\u0000typed"; // the "직접 입력" choice

/**
 * Move a server's Docker images to another disk (admins): the node stops Docker, copies its folder, points Docker
 * at the new one, checks every image is there, and only then removes the old folder; any failure puts it back.
 */
export function DockerMoveDialog({ nodeId, suggest, onClose }: { nodeId: string; suggest: string; onClose: () => void }) {
  const { data: state, error: loadError } = usePolling(() => api<DockerMoveState>(`/nodes/${nodeId}/docker-move`), 0, [nodeId]);
  const [picked, setPicked] = useState<string | null>(null); // a choice's folder, or TYPED
  const [typed, setTyped] = useState("");
  const [removeOld, setRemoveOld] = useState(true);
  const [stopContainers, setStopContainers] = useState(false);
  const [jobId, setJobId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { data: job } = usePolling(() => (jobId ? api<Job>(`/jobs/${jobId}`) : Promise.resolve(null)), jobId ? 2000 : 0, [jobId]);
  const options = state?.choices ?? [];
  const chosen = picked ?? (options[0]?.folder || TYPED);
  const to = chosen === TYPED ? typed : chosen;
  const running = !!job && ["queued", "running"].includes(job.status);

  const start = async () => {
    if (!window.confirm(`이 서버의 Docker를 멈추고 이미지를 ${to}(으)로 옮길까요?\n멈춘 동안 이 서버의 컨테이너가 모두 멈춥니다.`)) return;
    setBusy(true);
    setError(null);
    try {
      const made = await api<{ job_id: string }>(`/nodes/${nodeId}/docker-move`, {
        body: { target: to.trim(), remove_old: removeOld, stop_containers: stopContainers },
      });
      setJobId(made.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title="Docker 이미지 위치 옮기기" onClose={onClose}>
      {loadError && <div className="error">{loadError}</div>}
      {!state && !loadError && <p className="muted">확인하는 중…</p>}
      {state && !jobId && (
        <>
          <p className="muted">
            Docker는 이 서버의 이미지를 모두 한 폴더(<code>data-root</code>)에 둡니다. 지금 <code>{state.path || "?"}</code>입니다. 옮기려면
            Docker를 멈춰야 해서, <strong>그동안 이 서버의 컨테이너가 모두 멈춥니다</strong>(MOA 것이든 다른 사람 것이든). 이미지 양에
            따라 몇 분에서 수십 분 걸립니다. 하나라도 잘못되면 설정을 되돌리고 원래 폴더로 Docker를 다시 켭니다.
          </p>
          {!state.possible && <div className="notice">{state.why}</div>}
          <fieldset className="move-choices" disabled={!state.possible}>
            <legend>
              옮길 폴더 (다른 디스크의 빈 폴더)
              {state.here && (
                <span className="muted small">
                  {" "}
                  · 지금 디스크 {state.here.mount}: {room(state.here.free_gb)} 남음
                </span>
              )}
            </legend>
            {options.map((c, n) => (
              <label key={c.folder} className="check move-choice">
                <input type="radio" name="docker-target" checked={chosen === c.folder} onChange={() => setPicked(c.folder)} />
                <code>{c.folder}</code>
                <span className="muted small">
                  디스크 {c.mount} · {room(c.free_gb)} 남음{c.total_gb != null ? ` / ${room(c.total_gb)}` : ""}
                  {n === 0 ? " · 가장 여유" : ""}
                </span>
              </label>
            ))}
            <label className="check move-choice">
              <input type="radio" name="docker-target" checked={chosen === TYPED} onChange={() => setPicked(TYPED)} />
              직접 입력
              {chosen === TYPED && (
                <input
                  autoFocus
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                  placeholder={state.suggest || suggest || "/home4/docker"}
                />
              )}
            </label>
          </fieldset>
          <label className="check">
            <input type="checkbox" checked={removeOld} onChange={(e) => setRemoveOld(e.target.checked)} disabled={!state.possible} />
            새 위치로 잘 켜지고 이미지가 모두 보이면 예전 폴더를 지워 공간을 비움
          </label>
          <label className="check">
            <input type="checkbox" checked={stopContainers} onChange={(e) => setStopContainers(e.target.checked)} disabled={!state.possible} />
            돌고 있는 컨테이너가 있어도 멈추고 옮김 (끄면 컨테이너가 있을 때 아무것도 하지 않고 그 목록을 알려 줌)
          </label>
          {error && <div className="error">{error}</div>}
          <div className="modal-actions">
            <button type="button" onClick={onClose}>
              취소
            </button>
            <button className="danger" disabled={busy || !state.possible || !to.trim().startsWith("/")} onClick={() => void start()}>
              옮기기
            </button>
          </div>
        </>
      )}
      {jobId && (
        <>
          <p>
            {job ? <Badge kind={job.status}>{STATUS_LABEL[job.status] ?? job.status}</Badge> : null}{" "}
            {running ? "옮기는 중입니다. 이 창을 닫아도 서버에서 계속합니다." : job?.status === "succeeded" ? "옮겼습니다." : job?.error ?? ""}
          </p>
          <LogTail jobId={jobId} live={running || job?.status === "failed"} />
          <div className="modal-actions">
            <button type="button" onClick={onClose}>
              닫기
            </button>
          </div>
        </>
      )}
    </Modal>
  );
}

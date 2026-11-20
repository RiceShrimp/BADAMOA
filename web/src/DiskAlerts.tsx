import { CopyBlock } from "./ui";

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
  home: "BADAMOA의 설정 · 기록이 있는 곳입니다. 다른 파일을 정리해 공간을 비우세요",
  work: "가상환경 · 작업 폴더 · 받아 둔 가중치를 두는 곳입니다. 아래 저장 공간에서 지우거나 노드 설정에서 작업 폴더를 여유 있는 디스크로 바꾸세요",
  images: "다른 서버에서 받은 이미지를 두는 곳입니다. 아래 '받아 둔 버전'에서 지우거나 노드 설정에서 보관 위치를 바꾸세요",
  docker: "Docker 이미지가 있는 곳입니다. BADAMOA는 Docker의 설정을 바꾸지 않습니다",
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
export function DiskAlertBox({ alerts, dockerTarget, isAdmin }: { alerts?: DiskAlert[]; dockerTarget?: string; isAdmin: boolean }) {
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
    </section>
  );
}

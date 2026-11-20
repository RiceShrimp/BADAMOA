import { useState } from "react";
import { api } from "./api";
import { CopyBlock, timeAgo, usePolling } from "./ui";

/** What the Hub reports about moving it to another PC (``hub/migration.py``). */
interface MigrationView {
  phase: "" | "exporting" | "exported" | "moved" | "failed";
  read_only: boolean;
  url: string;
  imported: { exported_at: string; version: string; file: string; at: string } | null;
  bytes: number;
  files: number;
  parts: { database: number; storage: number; wheels: number };
  running: { id: string; type: string }[];
  disks: { mount: string; free_gb: number; total_gb: number }[];
  stays: { originals: string[]; registry: string; local_node: string };
  file: string | null;
  error: string | null;
  progress: { files: number; bytes: number; total_bytes: number; step: string } | null;
  exported_at: string | null;
  download: string;
  file_bytes: number;
  nodes: { id: string; name: string; here: boolean; version: string }[];
}

const TYPED = "\u0000typed";

function size(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

function folderOn(mount: string): string {
  return mount.includes("\\") ? `${mount.replace(/\\$/, "")}\\badamoa-export` : `${mount.replace(/\/$/, "")}/badamoa-export`;
}

/**
 * Settings → Hub 옮기기 (admins): export this Hub into one file, import it on the new PC, then send the nodes
 * there. While exported this Hub only reads (what changed here would not reach the new one).
 */
export function MigrationPanel() {
  const { data: view, error, refresh } = usePolling(() => api<MigrationView>("/migration"), 4000);
  const [picked, setPicked] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  if (error && !view) return <div className="error">{error}</div>;
  if (!view) return <p className="muted">불러오는 중…</p>;

  const options = view.disks.map((d) => folderOn(d.mount));
  const chosen = picked ?? (options[0] || TYPED);
  const folder = chosen === TYPED ? typed : chosen;
  const fileName = (view.file ?? "").split(/[\\/]/).pop() ?? "";

  const act = async (path: string, body?: unknown, ask?: string) => {
    if (ask && !window.confirm(ask)) return;
    setBusy(true);
    setFailed(null);
    try {
      await api(path, { method: "POST", body });
      await refresh();
    } catch (err) {
      setFailed(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card migration">
      <h3>Hub 옮기기</h3>
      <p className="muted">
        이 Hub를 다른 PC로 옮길 때 씁니다. 파일 하나로 내보내고 새 PC에서 가져온 뒤, 아래에 새 Hub 주소를 적으면 노드와 접속하는 사람이 새
        Hub로 넘어갑니다. 계정 · 프로젝트 · 데이터셋 기록 · 학습 · 모델 · 평가 · 썸네일 · 결과 파일이 모두 갑니다. 데이터셋 원본과 받아 둔 이미지는
        노드에 있어 옮기지 않아도 됩니다.
      </p>
      {view.imported && (
        <div className="notice">
          이 Hub는 {view.imported.exported_at.slice(0, 16).replace("T", " ")}에 내보낸 파일({view.imported.file})에서 가져왔습니다.
        </div>
      )}

      {(view.phase === "" || view.phase === "failed") && (
        <>
          <h4>1. 내보내기</h4>
          {view.phase === "failed" && <div className="error">지난 내보내기가 실패했습니다: {view.error}</div>}
          <p>
            파일 크기 약 <strong>{size(view.bytes)}</strong>{" "}
            <span className="muted small">
              (DB {size(view.parts.database)} · Hub 저장 파일 {view.files.toLocaleString()}개 {size(view.parts.storage)} · 노드 패키지{" "}
              {size(view.parts.wheels)})
            </span>
          </p>
          <ul className="muted small">
            {view.stays.originals.length > 0 && (
              <li>
                이 PC의 노드({view.stays.local_node})에 원본이 있는 데이터셋 폴더 {view.stays.originals.length}개(
                {view.stays.originals.slice(0, 3).join(", ")}
                {view.stays.originals.length > 3 ? " …" : ""})는 이 PC에 남습니다. 이 PC를 노드로 남겨 두면 새 Hub에서도 그대로 씁니다.
              </li>
            )}
            {view.stays.registry && (
              <li>
                Docker 이미지는 이 PC의 레지스트리({view.stays.registry})에 그대로 있고, 새 Hub도 이 주소에서 받습니다. 이 PC를 끄려면 그 전에
                이미지를 새 레지스트리로 옮기세요.
              </li>
            )}
            <li>내보내기를 누른 때부터 이 Hub는 읽기만 됩니다(그 뒤에 바꾼 것은 새 Hub에 가지 않음). 옮기기를 취소하면 다시 씁니다.</li>
          </ul>
          {view.running.length > 0 && (
            <div className="notice">
              돌고 있는 작업 {view.running.length}개가 끝나야 내보낼 수 있습니다. <a href="#/jobs">작업 보기</a>
            </div>
          )}
          <fieldset className="move-choices">
            <legend>파일을 둘 폴더 (이 Hub PC)</legend>
            {view.disks.map((d, n) => (
              <label key={d.mount} className="check move-choice">
                <input type="radio" name="export-folder" checked={chosen === options[n]} onChange={() => setPicked(options[n])} />
                <code>{options[n]}</code>
                <span className="muted small">
                  {d.free_gb.toFixed(0)} GB 남음 / {d.total_gb.toFixed(0)} GB{n === 0 ? " · 가장 여유" : ""}
                  {d.free_gb * 1024 ** 3 < view.bytes * 1.05 ? " · 모자람" : ""}
                </span>
              </label>
            ))}
            <label className="check move-choice">
              <input type="radio" name="export-folder" checked={chosen === TYPED} onChange={() => setPicked(TYPED)} />
              직접 입력
              {chosen === TYPED && <input autoFocus value={typed} onChange={(e) => setTyped(e.target.value)} placeholder={options[0] ?? "D:\\badamoa-export"} />}
            </label>
          </fieldset>
          {failed && <div className="error">{failed}</div>}
          <button
            className="primary"
            disabled={busy || view.running.length > 0 || !folder.trim()}
            onClick={() =>
              void act(
                "/migration/export",
                { folder: folder.trim() },
                `${folder}에 이 Hub를 내보낼까요?\n지금부터 이 Hub는 읽기만 됩니다(옮기기를 취소하면 다시 씀).`,
              )
            }
          >
            내보내기
          </button>
        </>
      )}

      {view.phase === "exporting" && (
        <>
          <h4>1. 내보내는 중</h4>
          <p className="muted">끝날 때까지 이 Hub는 읽기만 됩니다. 이 창을 닫아도 계속합니다.</p>
          {view.progress && (
            <>
              <progress max={Math.max(1, view.progress.total_bytes)} value={view.progress.bytes} />
              <div className="muted small">
                {view.progress.step === "db"
                  ? "DB를 떠 두는 중…"
                  : `${view.progress.files.toLocaleString()}개 · ${size(view.progress.bytes)} / 약 ${size(view.progress.total_bytes)}`}
              </div>
            </>
          )}
        </>
      )}

      {(view.phase === "exported" || view.phase === "moved") && (
        <>
          <h4>1. 내보낸 파일</h4>
          <p>
            <code>{view.file}</code> <span className="muted small">({size(view.file_bytes)})</span>
            {view.download && (
              <>
                {" "}
                <a className="btn small" href={view.download} download>
                  내려받기
                </a>
              </>
            )}
          </p>
          <div className="muted small">새 PC의 브라우저에서 이 화면을 열어 내려받거나, 파일을 복사해 새 PC로 옮기세요.</div>

          <h4>2. 새 PC에서 가져오기</h4>
          <ol className="migration-steps">
            <li>새 PC에 MOA를 받고 Hub를 띄울 준비를 합니다(README의 Hub 띄우기: uv sync · 웹 빌드). 아직 Hub는 켜지 않습니다.</li>
            <li>
              그 폴더에서 파일을 가져옵니다. <code>--home</code>은 Hub를 띄울 때와 같게 적습니다.
              <CopyBlock text={`uv run badamoa hub import ${fileName || "badamoa-hub-….tar"} --home .badamoa/hub`} />
            </li>
            <li>
              Hub를 켭니다(<code>uv run badamoa hub run --home .badamoa/hub</code>). 예전과 같은 계정으로 로그인됩니다.
            </li>
          </ol>

          <h4>3. 노드와 접속을 새 Hub로 넘기기</h4>
          {view.phase === "moved" ? (
            <p>
              <strong>{view.url}</strong>(으)로 넘겼습니다. 노드는 다음 신호 때 새 Hub가 자기를 아는지 확인하고 넘어가며, 이 Hub에 접속하는 사람은 새
              Hub로 안내됩니다.
            </p>
          ) : (
            <p className="muted">새 Hub가 켜지면 그 주소를 적습니다. 새 Hub가 이 Hub를 가져온 것이 맞는지 확인한 뒤에 넘깁니다.</p>
          )}
          <div className="row-inline">
            <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder={view.url || "http://새 PC 주소:8470"} />
            <button
              className="primary"
              disabled={busy || !url.trim()}
              onClick={() => void act("/migration/hand-over", { url: url.trim() }, `노드와 접속을 ${url.trim()}(으)로 넘길까요?`)}
            >
              넘기기
            </button>
          </div>
          {failed && <div className="error">{failed}</div>}
          <table className="plain compact">
            <thead>
              <tr>
                <th>노드</th>
                <th>이 Hub에</th>
              </tr>
            </thead>
            <tbody>
              {view.nodes.map((n) => (
                <tr key={n.id}>
                  <td>{n.name}</td>
                  <td className="small">
                    {n.here
                      ? view.phase === "moved"
                        ? "아직 여기 (곧 넘어감)"
                        : "여기에 붙어 있음"
                      : view.phase === "moved"
                        ? "넘어갔거나 꺼짐"
                        : "꺼져 있음"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            {view.exported_at ? `내보낸 때: ${timeAgo(view.exported_at)}. ` : ""}새 PC로 옮기지 않기로 했으면 취소하세요. 이 Hub를 다시 쓰게 되고 파일은 그대로
            남습니다.
          </p>
          <button
            disabled={busy}
            onClick={() =>
              void act(
                "/migration/cancel",
                undefined,
                view.phase === "moved"
                  ? "옮기기를 취소할까요? 이미 새 Hub로 넘어간 노드는 그대로 새 Hub에 있습니다."
                  : "옮기기를 취소하고 이 Hub를 다시 쓸까요?",
              )
            }
          >
            옮기기 취소
          </button>
        </>
      )}
    </section>
  );
}

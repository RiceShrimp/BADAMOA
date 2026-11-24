import { type FormEvent, useEffect, useState } from "react";
import { fileSize } from "./Files";
import { api } from "./api";
import { usePolling } from "./ui";

/** Where the vector DB is and what it holds (``/vector-store``, 관리자). */
interface VectorStoreInfo {
  path: string;
  default: string;
  size: number;
  tables: { name: string; rows: number; indexed: boolean }[];
  moving: { state: string; from: string; to: string; copied: number; total: number; error: string };
}

/** 벡터 DB (이미지 탐색): where the image vectors are kept, how many, and moving them to another folder of the Hub PC. */
export function VectorStorePanel({ isAdmin }: { isAdmin: boolean }) {
  const [moving, setMoving] = useState(false);
  const { data, error, refresh } = usePolling(
    () => (isAdmin ? api<VectorStoreInfo>("/vector-store") : Promise.resolve(null)),
    moving ? 2000 : 30000,
    [isAdmin, moving],
  );
  const [target, setTarget] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const going = data?.moving.state === "moving";
  useEffect(() => {
    if (moving && data && !going) setMoving(false); // moved (or not): back to the slow refresh
  }, [moving, data, going]);
  if (!isAdmin) return null;
  const rows = (data?.tables ?? []).reduce((n, t) => n + t.rows, 0);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!window.confirm(`벡터 DB를 ${target}(으)로 옮길까요? 옮기는 동안 이미지 탐색과 임베딩 저장이 잠시 멈춥니다.`)) return;
    setBusy(true);
    setProblem(null);
    try {
      await api("/vector-store/move", { body: { path: target } });
      setMoving(true);
      setTarget("");
      void refresh();
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="card vector-store">
      <div className="card-head">
        <div>
          <h3>벡터 DB</h3>
          <div className="muted small">
            이미지 탐색이 찾는 이미지 벡터를 두는 곳입니다(LanceDB, 따로 띄울 서버 없이 이 폴더에 저장). 이미지가 많아지면 색인으로 찾습니다.
          </div>
        </div>
        <div className="storage-total">
          <strong>{data ? fileSize(data.size) : "–"}</strong>
          <span className="muted small">벡터 {rows.toLocaleString()}개</span>
        </div>
      </div>
      {error && <div className="error">{error}</div>}
      {data && (
        <>
          <div className="storage-path">
            <span className="muted small">위치</span> <code>{data.path}</code>
            {data.path === data.default && <span className="muted small"> (기본: Hub 폴더 안)</span>}
          </div>
          {data.tables.map((t) => (
            <div className="storage-part" key={t.name}>
              <span>
                {t.name} <span className="muted small">{t.indexed ? "색인으로 찾음" : "색인 없이 모두 비교 (5만 개부터 색인)"}</span>
              </span>
              <span className="num">{t.rows.toLocaleString()}개</span>
            </div>
          ))}
          {going && (
            <div className="notice small">
              {data.moving.to}(으)로 옮기는 중 · {fileSize(data.moving.copied)} / {fileSize(data.moving.total)}
            </div>
          )}
          {data.moving.state === "failed" && <div className="error small">옮기지 못했습니다(지금 위치 그대로): {data.moving.error}</div>}
          <form className="inline-form vector-move" onSubmit={submit}>
            <input
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="옮길 폴더 (Hub PC의 절대 경로, 비어 있거나 없는 폴더) 예: D:\badamoa-vectors"
              disabled={going}
              aria-label="옮길 폴더"
            />
            <button disabled={busy || going || !target.trim()}>옮기기</button>
          </form>
          {problem && <div className="error small">{problem}</div>}
          <div className="muted small">옮기면 지금 있는 벡터를 새 폴더로 복사한 뒤 그곳을 쓰고, 예전 폴더는 지웁니다.</div>
        </>
      )}
    </section>
  );
}

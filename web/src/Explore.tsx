import { type FormEvent, useEffect, useState } from "react";
import { ProjectPicker, useProject } from "./ProjectPicker";
import { api } from "./api";
import type { TreeDataset } from "./catalogTypes";
import { Icon } from "./ui";
import "./explore.css";

/** How much of a project 이미지 탐색 can find (``/projects/{id}/explore``). */
interface ExploreStatus {
  model: string;
  title: string;
  images: number;
  embedded: number;
  text: { offered: boolean; ready: boolean; problem: string };
}

/** An image found: how near (cosine) and a version it is in. */
interface Found {
  id: number;
  file_name: string;
  width: number;
  height: number;
  score: number;
  thumb_url: string | null;
  image_url: string | null;
  release: { id: string; version: string; dataset: string } | null;
}

type Query = { text: string } | { like: Found };

/** A version and how far its images have vectors (``/projects/{id}/explore/versions``). */
interface VersionRow {
  release: { id: string; version: string; dataset: string };
  total: number;
  done: number;
  todo: number;
  going: string | null;
}

/** A server a version's vectors can be made on (``/releases/{id}/embeddings/places``). */
interface Place {
  id: string;
  name: string;
  status: string;
  gpus: number;
  fits: boolean;
  why: string;
  transfer: boolean;
}
interface Places {
  default: string;
  nodes: Place[];
  problem: string;
}

/** 버전별 임베딩: each version's vectors, and 임베딩 만들기 for the ones it lacks, on a server with the 임베딩용 role. */
function VersionEmbeddings({ projectId, onMade }: { projectId: string; onMade: () => void }) {
  const [rows, setRows] = useState<VersionRow[] | null>(null);
  const [open, setOpen] = useState<string | null>(null); // the version whose 임베딩 만들기 is open
  const [places, setPlaces] = useState<Places | null>(null);
  const [node, setNode] = useState("");
  const [cpu, setCpu] = useState(false); // as 오토라벨: a GPU when the server has one, or the CPU
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const load = () => void api<VersionRow[]>(`/projects/${projectId}/explore/versions`).then(setRows).catch(() => setRows([]));
  useEffect(load, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps
  const going = rows?.some((r) => r.going) ?? false;
  useEffect(() => {
    if (!going) return;
    const timer = window.setInterval(() => {
      load();
      onMade();
    }, 10000);
    return () => window.clearInterval(timer);
  }, [going]); // eslint-disable-line react-hooks/exhaustive-deps

  const choose = (releaseId: string) => {
    setOpen(releaseId);
    setPlaces(null);
    setNode("");
    setError(null);
    void api<Places>(`/releases/${releaseId}/embeddings/places`)
      .then((found) => {
        setPlaces(found);
        setNode(found.default);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  };
  const start = async (releaseId: string) => {
    setBusy(true);
    setError(null);
    try {
      await api(`/releases/${releaseId}/embeddings`, { body: { node_id: node, gpus: cpu ? 0 : null } });
      setOpen(null);
      load();
      onMade();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const n = (v: number) => v.toLocaleString();
  if (!rows) return null;
  return (
    <details className="card explore-versions" open={rows.some((r) => r.done === 0) || going}>
      <summary>
        버전별 임베딩 <span className="muted small">— 벡터가 없는 이미지만 임베딩용 서버가 만듭니다</span>
      </summary>
      {rows.length === 0 && <p className="muted small">이 프로젝트에는 등록이 끝난 버전이 없습니다.</p>}
      <ul className="explore-version-list">
        {rows.map((r) => (
          <li key={r.release.id}>
            <div className="explore-version-row">
              <a href={`#/datasets/${r.release.id}`}>
                {r.release.dataset} {r.release.version}
              </a>
              <span className="explore-meter" title={`${n(r.done)}/${n(r.total)}장`}>
                <i style={{ width: `${r.total ? (100 * r.done) / r.total : 0}%` }} />
              </span>
              <span className="muted small nowrap">
                벡터 {n(r.done)}/{n(r.total)}장
              </span>
              {r.going ? (
                <span className="tag">만드는 중</span>
              ) : r.todo === 0 ? (
                <span className="muted small">모두 있음</span>
              ) : (
                <button type="button" className="ghost-btn" onClick={() => (open === r.release.id ? setOpen(null) : choose(r.release.id))}>
                  임베딩 만들기 ({n(r.todo)}장)
                </button>
              )}
            </div>
            {open === r.release.id && (
              <div className="explore-start">
                {!places && !error && <span className="muted small">서버를 찾는 중…</span>}
                {places && (
                  <>
                    <label className="inline-select">
                      만들 서버
                      <select value={node} onChange={(e) => setNode(e.target.value)}>
                        {!places.nodes.some((p) => p.fits) && <option value="">임베딩용 서버가 없습니다</option>}
                        {places.nodes.map((p) => (
                          <option key={p.id} value={p.id} disabled={!p.fits}>
                            {p.name} · {p.gpus ? `GPU ${p.gpus}` : "CPU"} · {p.transfer ? "이미지를 먼저 받음" : "이미지 있음"}
                            {p.status === "online" ? "" : " · 지금 응답 없음"}
                            {p.fits ? "" : ` — ${p.why}`}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="inline-select">
                      계산
                      <select value={cpu ? "cpu" : "gpu"} onChange={(e) => setCpu(e.target.value === "cpu")}>
                        <option value="gpu">GPU (서버에 있으면)</option>
                        <option value="cpu">CPU</option>
                      </select>
                    </label>
                    <button type="button" className="primary" disabled={busy || !node} onClick={() => void start(r.release.id)}>
                      시작
                    </button>
                    {!places.nodes.some((p) => p.fits) && (
                      <span className="warn-text small">
                        <a href="#/nodes">노드</a> 화면에서 벡터를 만들 서버의 설정 → 역할에 &quot;임베딩용&quot;을 켜세요.
                      </span>
                    )}
                    {places.nodes.find((p) => p.id === node)?.transfer && (
                      <span className="muted small">그 서버가 이 버전의 이미지를 먼저 받은 뒤 만듭니다(받은 이미지는 그 서버에 남음).</span>
                    )}
                  </>
                )}
                {error && <span className="bad-text small">{error}</span>}
              </div>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}

/** 이미지 탐색: a project's images found by words or by another image, with the vectors made alongside trainings. */
export function Explore() {
  const { projects, projectId, choose } = useProject();
  const [status, setStatus] = useState<ExploreStatus | null>(null);
  const [tree, setTree] = useState<TreeDataset[]>([]);
  const [release, setRelease] = useState(""); // "": every version of the project
  const [words, setWords] = useState("");
  const [query, setQuery] = useState<Query | null>(null);
  const [found, setFound] = useState<{ searched: number; items: Found[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = () => {
    if (projectId) void api<ExploreStatus>(`/projects/${projectId}/explore`).then(setStatus).catch((e) => setError(String(e)));
  };
  useEffect(() => {
    setStatus(null);
    setFound(null);
    setQuery(null);
    setRelease("");
    if (!projectId) return;
    loadStatus();
    void api<TreeDataset[]>(`/projects/${projectId}/datasets`).then(setTree).catch(() => setTree([]));
  }, [projectId]);

  useEffect(() => {
    if (!query || !projectId) return;
    let live = true;
    setBusy(true);
    setError(null);
    const params = new URLSearchParams({ limit: "120" });
    if ("text" in query) params.set("text", query.text);
    else params.set("like", String(query.like.id));
    if (release) params.set("release_id", release);
    api<{ searched: number; items: Found[] }>(`/projects/${projectId}/explore/search?${params}`)
      .then((got) => live && setFound(got))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, [query, release, projectId]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (words.trim()) setQuery({ text: words.trim() });
  };
  const textReady = Boolean(status?.text.ready);
  const versions = tree.flatMap((d) =>
    d.releases.filter((r) => r.status === "ready" && !r.deleted_at).map((r) => ({ id: r.id, label: `${d.name} ${r.version}` })),
  );
  const n = (v: number) => v.toLocaleString();

  return (
    <>
      <div className="page-head">
        <div>
          <h1>이미지 탐색</h1>
          <p className="muted">
            프로젝트의 모든 데이터셋 이미지를 글이나 비슷한 이미지로 찾습니다. 찾는 것은 벡터가 있는 이미지뿐이며, 벡터는 아래 버전별 임베딩이나 새 학습의
            &quot;학습하면서 이미지 임베딩도 만들기&quot;로 만듭니다.
          </p>
        </div>
        <div className="head-actions">
          <ProjectPicker projects={projects} projectId={projectId} onChange={choose} />
        </div>
      </div>

      {status && (
        <div className="card explore-status">
          <span>
            이미지 <strong>{n(status.images)}</strong>장 중 <strong>{n(status.embedded)}</strong>장을 찾을 수 있습니다
            <span className="muted"> · {status.title}</span>
          </span>
          <span className={`small ${textReady ? "muted" : "warn-text"}`}>
            {textReady
              ? "글 검색 준비됨 (Hub에서 검색어를 벡터로 바꿈)"
              : status.text.problem
                ? `글 검색을 쓸 수 없음: ${status.text.problem}`
                : "글 검색은 첫 임베딩 작업이 끝나면 쓸 수 있습니다"}
          </span>
        </div>
      )}
      {status && status.embedded === 0 && (
        <div className="card empty-state">
          <Icon name="images" size={26} />
          <strong>아직 찾을 수 있는 이미지가 없습니다</strong>
          <p className="muted">
            아래 버전별 임베딩에서 &quot;임베딩 만들기&quot;를 누르거나, 새 학습에서 &quot;학습하면서 이미지 임베딩도 만들기&quot;를 켜고 학습하세요(만드는 서버에
            노드 역할 &quot;임베딩용&quot;이 있어야 함).
          </p>
        </div>
      )}
      {status && projectId && <VersionEmbeddings projectId={projectId} onMade={loadStatus} />}

      {status && status.embedded > 0 && (
        <>
          <form className="explore-bar" onSubmit={submit}>
            <input
              type="search"
              value={words}
              onChange={(e) => setWords(e.target.value)}
              placeholder={textReady ? "찾을 것을 글로: 밤에 찍힌 트럭, a person riding a bike" : "글 검색은 첫 임베딩 작업이 끝나면"}
              disabled={!textReady}
              aria-label="글로 찾기"
            />
            <button className="primary" disabled={!textReady || !words.trim() || busy}>
              <Icon name="eye" size={14} /> 찾기
            </button>
            <select value={release} onChange={(e) => setRelease(e.target.value)} aria-label="찾을 버전">
              <option value="">프로젝트의 모든 버전</option>
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                </option>
              ))}
            </select>
          </form>
          {query && "like" in query && (
            <div className="explore-anchor">
              {query.like.thumb_url && <img src={query.like.thumb_url} alt="" />}
              <span>
                <strong>이 이미지와 비슷한 것</strong>
                <span className="muted small"> · {query.like.file_name}</span>
              </span>
              <button type="button" className="link small" onClick={() => setQuery(words.trim() ? { text: words.trim() } : null)}>
                지우기
              </button>
            </div>
          )}
          {error && <div className="error">{error}</div>}
          {busy && <p className="muted">찾는 중…</p>}
          {!query && !busy && <p className="muted">글로 찾거나, 결과에서 &quot;비슷한 이미지&quot;를 눌러 그 이미지와 비슷한 것을 찾습니다.</p>}
          {found && !busy && (
            <>
              <p className="muted small">
                {n(found.searched)}장에서 가까운 순으로 {n(found.items.length)}장
                {"text" in (query ?? {}) ? " · 글 검색의 점수는 0.1~0.2만 되어도 잘 맞는 편입니다" : ""}
              </p>
              <div className="thumb-grid explore-grid">
                {found.items.map((item) => (
                  <div key={item.id} className="thumb explore-thumb">
                    <a className="thumb-image" href={item.image_url ?? undefined} target="_blank" rel="noreferrer" title="원본 보기">
                      {item.thumb_url ? <img src={item.thumb_url} alt={item.file_name} loading="lazy" /> : <Icon name="images" />}
                    </a>
                    <div className="explore-meta">
                      <span className="tag">{item.score.toFixed(3)}</span>
                      {item.release ? (
                        <a className="small ellipsis" href={`#/datasets/${item.release.id}`} title="그 버전 열기">
                          {item.release.dataset} {item.release.version}
                        </a>
                      ) : (
                        <span className="muted small">버전 없음</span>
                      )}
                    </div>
                    <div className="thumb-name ellipsis" title={item.file_name}>
                      {item.file_name}
                    </div>
                    <button type="button" className="ghost-btn" onClick={() => setQuery({ like: item })}>
                      비슷한 이미지
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </>
  );
}

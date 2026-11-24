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

  useEffect(() => {
    setStatus(null);
    setFound(null);
    setQuery(null);
    setRelease("");
    if (!projectId) return;
    void api<ExploreStatus>(`/projects/${projectId}/explore`).then(setStatus).catch((e) => setError(String(e)));
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
            프로젝트의 모든 데이터셋 이미지를 글이나 비슷한 이미지로 찾습니다. 찾는 것은 벡터가 있는 이미지뿐이며, 벡터는 새 학습에서 &quot;학습하면서
            이미지 임베딩도 만들기&quot;로 만듭니다.
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
            학습 화면의 새 학습에서 &quot;학습하면서 이미지 임베딩도 만들기&quot;를 켜고 학습하면, 학습하는 서버가 그 버전의 이미지 벡터를 만듭니다(그 서버에
            노드 역할 &quot;임베딩용&quot;이 있어야 함).
          </p>
        </div>
      )}

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

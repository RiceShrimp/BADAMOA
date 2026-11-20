import { type FormEvent, useState } from "react";
import { api } from "./api";
import { Badge, CopyBlock, usePolling } from "./ui";

/** The Hub's Docker registry as the Hub sees it (``hub/docker_registry.py``). */
interface RegistryInfo {
  mode: "managed" | "adopted" | "external" | "off" | "none";
  address: string;
  container: string;
  data: string; // where its images are kept on the Hub's PC
  ok: boolean;
  error: string;
  images: number | null; // repositories in it
  checked_at: string | null;
  auto: boolean;
  docker: boolean;
}

/** An image a model kind or an environment names, and whether its registry has it. */
export interface RegistryImage {
  image: string;
  used_by: string[];
  registry: string; // "" for Docker Hub
  present: boolean | null; // null: could not tell (or Docker Hub)
  hub_has: boolean; // the Hub's PC has it: it can be pushed from there
  ours: boolean;
  push: string; // the command to push it where it was built
}

const MODE: Record<RegistryInfo["mode"], string> = {
  managed: "MOA가 만든 레지스트리",
  adopted: "이 PC에 있던 레지스트리를 씀",
  external: "직접 적은 주소",
  off: "꺼 둠",
  none: "없음",
};

/** The images in use, looked up once per screen (admins only: members see nothing of this). */
export function useRegistryImages(isAdmin: boolean) {
  return usePolling(
    () => (isAdmin ? api<RegistryImage[]>("/docker-registry/images") : Promise.resolve([] as RegistryImage[])),
    60000,
    [isAdmin],
  );
}

/** One line under an image: missing from its registry, said plainly. */
export function ImageInRegistry({ image, rows }: { image: string; rows: RegistryImage[] | null | undefined }) {
  const row = rows?.find((r) => r.image === image);
  if (!row || row.present !== false) return null;
  return (
    <div className="small warn-text">
      레지스트리에 이 이미지가 없어 다른 서버가 받지 못합니다 — <a href="#/settings/environments">설정 → 환경에서 올리기</a>
    </div>
  );
}

/** 설정 → 환경 (admins): where the images servers pull are kept, set up by the Hub itself, and images it lacks. */
export function DockerRegistry() {
  const { data, refresh } = usePolling(() => api<RegistryInfo>("/docker-registry"), 30000);
  const [pushed, setPushed] = useState<Record<string, string>>({}); // image → the push job
  const { data: images, refresh: refreshImages } = usePolling(
    () => api<RegistryImage[]>("/docker-registry/images"),
    Object.keys(pushed).length ? 5000 : 60000,
    [Object.keys(pushed).length],
  );
  const [busy, setBusy] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [address, setAddress] = useState("");

  const act = async (what: string, call: () => Promise<unknown>) => {
    setBusy(what);
    setError(null);
    try {
      await call();
      await refresh();
      await refreshImages();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };
  const save = (e: FormEvent) => {
    e.preventDefault();
    void act("save", async () => {
      await api("/docker-registry", { method: "PUT", body: { address: address.trim() || null } });
      setEditing(false);
    });
  };
  const push = (image: string) =>
    void act(image, async () => {
      const job = await api<{ job_id: string }>("/docker-registry/push", { body: { image } });
      setPushed((p) => ({ ...p, [image]: job.job_id }));
    });

  if (!data) return null;
  const live = data.mode === "managed" || data.mode === "adopted" || data.mode === "external";
  return (
    <section className="card registry-card">
      <div className="card-head">
        <div>
          <h3>Docker 레지스트리</h3>
          <div className="muted small">
            모델 종류 · 환경의 Docker 이미지를 두는 곳입니다. 학습 서버들은 여기서 이미지를 받고, Hub는 이 PC에 레지스트리가 없으면 스스로 만듭니다.
          </div>
        </div>
        <span className="env-actions">
          <button className="mini" disabled={!!busy} onClick={() => void act("check", () => api("/docker-registry/check", { method: "POST" }))}>
            {busy === "check" ? "확인하는 중…" : data.mode === "none" || data.mode === "off" ? "만들기 · 다시 확인" : "다시 확인"}
          </button>
          <button className="mini" disabled={!!busy} onClick={() => setEditing(!editing)}>
            주소 바꾸기
          </button>
          {data.mode !== "off" && (
            <button className="mini" disabled={!!busy} onClick={() => void act("off", () => api("/docker-registry", { method: "PUT", body: { off: true } }))}>
              끄기
            </button>
          )}
        </span>
      </div>
      {live ? (
        <div className="registry-line">
          <code className="mono">{data.address}</code>
          <span className="tag">{MODE[data.mode]}</span>
          <Badge kind={data.ok ? "succeeded" : "failed"}>{data.ok ? "연결됨" : "응답 없음"}</Badge>
          {data.images != null && <span className="muted small">저장소 {data.images}개</span>}
        </div>
      ) : (
        <div className="registry-line">
          <span className="tag">{MODE[data.mode]}</span>
          <span className="muted small">
            {data.mode === "off"
              ? "Hub가 레지스트리를 만들거나 쓰지 않습니다. 다시 확인을 누르면 이 PC의 레지스트리를 찾거나 만듭니다."
              : "Docker가 있는 PC면 Hub가 곧 다시 해 봅니다."}
          </span>
        </div>
      )}
      {data.data && (
        <div className="muted small registry-line">
          이미지 보관 폴더 <code className="mono">{data.data}</code>
          {data.container ? ` · 컨테이너 ${data.container}` : ""}
        </div>
      )}
      {data.error && <div className="small warn-text registry-line">{data.error}</div>}
      {live && !data.ok ? null : null}
      {editing && (
        <form className="registry-form" onSubmit={save}>
          <input
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder={data.address || "192.168.0.10:5000"}
            aria-label="레지스트리 주소"
            autoFocus
          />
          <button className="primary" disabled={!!busy}>
            저장
          </button>
          <span className="muted small">비워 두고 저장하면 이 PC의 레지스트리를 다시 찾거나 만듭니다.</span>
        </form>
      )}
      {error && <div className="error">{error}</div>}
      {images && images.length > 0 && (
        <div className="registry-images">
          <div className="eyebrow">쓰는 이미지</div>
          {images.map((row) => (
            <div className="registry-image" key={row.image}>
              <code className="mono" title={row.image}>
                {row.image}
              </code>
              <span className="muted small">{row.used_by.join(", ")}</span>
              {row.present === true ? (
                <Badge kind="succeeded">레지스트리에 있음</Badge>
              ) : row.present === false ? (
                <Badge kind="failed">레지스트리에 없음</Badge>
              ) : (
                <span className="muted small">{row.registry ? "확인 못 함" : "Docker Hub"}</span>
              )}
              {row.present === false && (
                <div className="registry-fix">
                  {pushed[row.image] ? (
                    <a className="small" href={`#/jobs/${pushed[row.image]}`}>
                      올리는 중 · 작업 보기
                    </a>
                  ) : row.hub_has ? (
                    <button className="mini" disabled={!!busy} onClick={() => push(row.image)}>
                      레지스트리에 올리기
                    </button>
                  ) : (
                    <>
                      <span className="muted small">이 PC에는 없습니다. 이미지를 만든 PC에서:</span>
                      <CopyBlock label="" text={row.push} />
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

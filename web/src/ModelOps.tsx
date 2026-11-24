import { FileList } from "./Files";
import { LoggedTable } from "./Media";
import { type FormEvent, Fragment, useEffect, useState } from "react";
import { DataAfterChoice, type GpuChoice, GpuChips, ImagesNote, PlaceGrid, PlaceRow, type PlaceServer, ServerSelect, copiedTo, gpuBody, spaceShort } from "./ServerPlace";
import { api, apiUpload } from "./api";
import { type ReleaseInfo, type TreeDataset, sameTask, taskLabel } from "./catalogTypes";
import { METRIC_LABEL, type ModelDetail, type ModelOp, keyMetrics, metric } from "./trainTypes";
import { Badge, Icon, Modal, STATUS_LABEL, timeAgo, usePolling } from "./ui";

const ACTION_LABEL: Record<string, string> = { eval: "평가", export: "내보내기", predict: "테스트" };

/** " · 1.1.0 코드로" when a newer template version that reads this model's weights did the job. */
function codeNote(op: ModelOp): string {
  return op.code?.trained_with && op.code.version ? ` · ${op.code.name ?? ""} ${op.code.version} 코드로` : "";
}
const SPLIT_LABEL: Record<string, string> = { val: "val", test: "test", train: "train" };
const ACTIVE = ["queued", "running"];

/** What can be done with a trained model — evaluate, export, test on images — and what came of it. */
export function ModelOps({ model }: { model: ModelDetail }) {
  const [busy, setBusy] = useState(false);
  const { data: ops, refresh } = usePolling(() => api<ModelOp[]>(`/models/${model.id}/ops`), busy ? 2000 : 15000, [model.id]);
  const [dialog, setDialog] = useState<"eval" | "export" | "predict" | null>(null);
  useEffect(() => setBusy(!!ops?.some((o) => ACTIVE.includes(o.status))), [ops]);

  const started = () => {
    setDialog(null);
    setBusy(true);
    void refresh();
  };
  const [shown, setShown] = useState<string | null>(null); // the evaluation whose files are open
  const evals = ops?.filter((o) => o.kind === "eval") ?? [];
  const exports = ops?.filter((o) => o.kind === "export") ?? [];
  const tests = ops?.filter((o) => o.kind === "predict") ?? [];
  const metricKeys = keyMetrics(model.task);

  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h3>이 모델로 할 수 있는 것</h3>
            <div className="muted small">
              {model.source === "upload"
                ? `올릴 때 정한 모델 종류(${model.template.title})의 코드와 환경으로 실행합니다.`
                : "학습할 때와 같은 코드와 환경으로 실행합니다."}
            </div>
          </div>
          <div className="head-actions">
            {(["eval", "export", "predict"] as const).map((action) => (
              <button
                key={action}
                onClick={() => setDialog(action)}
                disabled={!model.actions.includes(action)}
                title={model.actions.includes(action) ? undefined : `${model.template.title}에는 ${ACTION_LABEL[action]} 코드가 없습니다`}
              >
                {ACTION_LABEL[action]}
              </button>
            ))}
          </div>
        </div>

        {ops && ops.length === 0 && <p className="muted">아직 평가하거나 내보내거나 테스트한 기록이 없습니다.</p>}

        {evals.length > 0 && (
          <>
            <div className="eyebrow op-head">평가 기록</div>
            <div className="table-wrap">
              <table className="plain">
                <thead>
                  <tr>
                    <th>버전 · split</th>
                    {metricKeys.map((k) => (
                      <th key={k} className="right">
                        {METRIC_LABEL[k] ?? k}
                      </th>
                    ))}
                    <th>상태</th>
                    <th>산출물</th>
                    <th className="right">언제</th>
                  </tr>
                </thead>
                <tbody>
                  {evals.map((op) => (
                    <Fragment key={op.id}>
                      <tr>
                        <td>
                          <a href={`#/models/eval/${op.id}`} title="평가 화면: 로그 · 클래스별 · 다시 실행 · 같은 데이터로 비교">
                            {op.release ? `${op.release.dataset} ${op.release.version}` : "삭제된 버전"} ·{" "}
                            {SPLIT_LABEL[String(op.params.split)] ?? String(op.params.split)}
                          </a>
                          {!ACTIVE.includes(op.status) && (
                            <button
                              className="link small op-delete"
                              title="이 평가 기록 지우기 (실행한 사람이나 관리자)"
                              onClick={() => {
                                if (!window.confirm("이 평가 기록을 지울까요? 지표 · 결과 파일 · 이미지별 예측이 함께 지워지고 되돌릴 수 없습니다.")) return;
                                void api(`/model-ops/${op.id}`, { method: "DELETE" })
                                  .then(() => refresh())
                                  .catch((err) => window.alert(err instanceof Error ? err.message : String(err)));
                              }}
                            >
                              삭제
                            </button>
                          )}
                        </td>
                        {metricKeys.map((k) => (
                          <td key={k} className="num right">
                            {metric(op.result.metrics?.[k])}
                          </td>
                        ))}
                        <td>
                          <OpStatus op={op} />
                        </td>
                        <td>
                          {op.files.length + (op.tables?.length ?? 0) > 0 ? (
                            <button className="link" onClick={() => setShown(shown === op.id ? null : op.id)}>
                              {shown === op.id
                                ? "접기"
                                : op.tables?.length
                                  ? `이미지별 결과 · 파일 ${op.files.length}개`
                                  : `파일 ${op.files.length}개`}
                            </button>
                          ) : (
                            <span className="muted">–</span>
                          )}
                        </td>
                        <td className="muted right">{timeAgo(op.created_at)}
{codeNote(op)}</td>
                      </tr>
                      {shown === op.id && (
                        <tr className="op-files">
                          <td colSpan={metricKeys.length + 4}>
                            {[...(op.tables ?? [])]
                              .sort((a, b) => (a.name.startsWith("failures") ? -1 : 1) - (b.name.startsWith("failures") ? -1 : 1))
                              .map((t) => (
                                <div key={t.name} className="op-table">
                                  <div className="eyebrow">
                                    {t.name.startsWith("failures")
                                      ? "틀린 것이 많은 이미지"
                                      : t.name.startsWith("per_image")
                                        ? "이미지별 결과 (틀린 것 먼저)"
                                        : t.name}
                                  </div>
                                  <LoggedTable
                                    file={t}
                                    images={{
                                      ...Object.fromEntries(op.files.filter((f) => f.kind === "image").map((f) => [f.name, f.url])),
                                      ...(op.table_images ?? {}),
                                    }}
                                  />
                                </div>
                              ))}
                            {op.files.length > 0 && <FileList files={op.files} />}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {exports.length > 0 && (
          <>
            <div className="eyebrow op-head">내보낸 파일</div>
            {exports.map((op) => (
              <div className="op-row" key={op.id}>
                <span className="tag">{String(op.params.format)}</span>
                {op.files.map((f) => (
                  <a key={f.name} className="run-file" href={f.url} download>
                    <Icon name="copy" size={14} />
                    {f.name} <span className="muted small">{(f.bytes / 1024 ** 2).toFixed(1)} MB</span>
                  </a>
                ))}
                {op.files.length === 0 && <OpStatus op={op} />}
                <span className="spacer" />
                <span className="muted small">{timeAgo(op.created_at)}
{codeNote(op)}</span>
              </div>
            ))}
          </>
        )}

        {tests.length > 0 && (
          <>
            <div className="eyebrow op-head">테스트 결과</div>
            {tests.slice(0, 3).map((op) => (
              <TestResult key={op.id} op={op} classes={model.classes} />
            ))}
          </>
        )}
      </section>

      {dialog === "eval" && <EvalDialog model={model} onClose={() => setDialog(null)} onStarted={started} />}
      {dialog === "export" && <ExportDialog model={model} onClose={() => setDialog(null)} onStarted={started} />}
      {dialog === "predict" && <TestDialog model={model} onClose={() => setDialog(null)} onStarted={started} />}
    </>
  );
}

function OpStatus({ op }: { op: ModelOp }) {
  const stop = () => {
    if (op.job_id && window.confirm("중단할까요?")) void api(`/jobs/${op.job_id}/cancel`, { method: "POST" });
  };
  return (
    <>
      <Badge kind={op.status}>{STATUS_LABEL[op.status]}</Badge>
      {ACTIVE.includes(op.status) && op.job_id && (
        <button className="link small" onClick={stop}>
          중단
        </button>
      )}
      {op.status === "queued" && op.wait_reason && <div className="wait-reason">{op.wait_reason}</div>}
      {op.status === "failed" && op.error && (
        <div className="wait-reason">
          {op.error} {op.kind === "eval" ? <a href={`#/models/eval/${op.id}`}>로그 · 다시 실행</a> : <a href="#/jobs">작업 화면에서 로그 보기</a>}
        </div>
      )}
    </>
  );
}

function TestResult({ op, classes }: { op: ModelOp; classes: string[] }) {
  const pictures = op.files.filter((f) => f.kind === "image");
  const predictions = op.result.predictions ?? [];
  return (
    <div className="op-test">
      <div className="op-row">
        <OpStatus op={op} />
        <span className="muted small">
          이미지 {predictions.length || "–"}장 · {timeAgo(op.created_at)}
{codeNote(op)}
        </span>
      </div>
      {pictures.length > 0 && (
        <div className="run-pictures">
          {pictures.map((f) => {
            const found = predictions.find((p) => p.image === f.name);
            return (
              <a key={f.name} href={f.url} target="_blank" rel="noreferrer" title="새 창에서 크게 보기">
                <img src={f.url} alt={f.name} loading="lazy" />
                <span>{found?.boxes ? `${found.boxes.length}개 검출` : f.name}</span>
              </a>
            );
          })}
        </div>
      )}
      {pictures.length === 0 &&
        predictions.map((p) => (
          <div className="op-row" key={p.image}>
            <span className="mono small">{p.image.replace(/^\d+_/, "")}</span>
            {(p.top ?? []).slice(0, 3).map(([index, probability], i) => (
              <span key={index} className={`tag ${i === 0 ? "accent" : ""}`}>
                {classes[index] ?? index} {(probability * 100).toFixed(0)}%
              </span>
            ))}
            {p.boxes && <span className="muted small">{p.boxes.length}개 검출</span>}
          </div>
        ))}
    </div>
  );
}

interface DialogProps {
  model: ModelDetail;
  onClose: () => void;
  onStarted: () => void;
}

/** A server a model's evaluation, export or test can run on, as the Hub sees it. */
interface Place extends PlaceServer {
  trained_here: boolean;
  holds_images: boolean;
}

export interface Places {
  kind: string;
  default: string;
  holder: { id: string; name: string } | null; // an evaluation: the server holding the version's images
  items: number; // an evaluation: what a server without them receives first
  bytes: number;
  nodes: Place[];
}

/**
 * An evaluation's server, as 새 학습 asks it: picked (never chosen for you), what happens to the version's images,
 * its GPU, and what becomes of a copy sent there. ``ready``: a server that fits, with room for the copy.
 */
export function useEvalPlace(places: Places | null | undefined) {
  const [node, setNode] = useState("");
  useEffect(() => setNode(""), [places]);
  const [gpu, setGpu] = useState<GpuChoice>("auto");
  useEffect(() => setGpu("auto"), [node]);
  const [dataAfter, setDataAfter] = useState<"keep" | "delete">("keep");
  const server = places?.nodes.find((n) => n.id === node);
  const copied = copiedTo(server);
  return {
    node,
    setNode,
    gpu,
    setGpu,
    dataAfter,
    setDataAfter,
    server,
    copied,
    ready: Boolean(server?.fits) && !spaceShort(server, places?.bytes ?? 0),
    body: { node_id: node, ...gpuBody(gpu, server), data_after: copied ? dataAfter : "" },
  };
}

export function EvalPlaceFields({
  places,
  error,
  place,
}: {
  places: Places | null | undefined;
  error?: string | null;
  place: ReturnType<typeof useEvalPlace>;
}) {
  if (error) return <div className="error small">{error}</div>;
  if (!places) return <p className="muted small">실행할 서버를 찾는 중…</p>;
  const { server } = place;
  return (
    <PlaceGrid>
      <PlaceRow label="서버">
        <ServerSelect servers={places.nodes} value={place.node} onChange={place.setNode} />
        {!places.nodes.some((n) => n.fits) && (
          <span className="bad-text small">실행할 수 있는 서버가 없습니다. {places.nodes.map((n) => `${n.name}: ${n.why}`).join(" / ")}</span>
        )}
      </PlaceRow>
      {server && (
        <PlaceRow label="이미지">
          <ImagesNote server={server} holder={places.holder?.name} items={places.items} bytes={places.bytes} />
        </PlaceRow>
      )}
      {server?.fits && (server.gpu_list ?? []).length > 0 && (
        <PlaceRow label="GPU">
          <GpuChips gpus={server.gpu_list ?? []} pick={place.gpu} onPick={place.setGpu} doing="평가합니다" />
        </PlaceRow>
      )}
      {server?.fits && place.copied && (
        <PlaceRow label="보낸 데이터">
          <DataAfterChoice value={place.dataAfter} onChange={place.setDataAfter} when="평가가 끝나면" />
        </PlaceRow>
      )}
    </PlaceGrid>
  );
}

export function usePlaces(modelId: string, kind: "eval" | "export" | "predict", releaseId = "") {
  const ready = kind !== "eval" || !!releaseId;
  const query = `kind=${kind}${releaseId ? `&release_id=${releaseId}` : ""}`;
  return usePolling(() => (ready ? api<Places>(`/models/${modelId}/places?${query}`) : Promise.resolve(null)), 0, [modelId, query]);
}

/**
 * Where it runs: the default (evaluation: the server holding the version's images; export and test: the server that
 * trained the model) or another one; the servers that cannot say why. "" stands for the default.
 */
export function PlacePicker({
  places,
  error,
  value,
  onChange,
}: {
  places: Places | null | undefined;
  error?: string | null;
  value: string;
  onChange: (id: string) => void;
}) {
  if (error) return <div className="error small">{error}</div>;
  if (!places) return <p className="muted small">실행할 서버를 찾는 중…</p>;
  const current = value || places.default;
  const chosen = places.nodes.find((n) => n.id === current);
  const label = (n: Place) => {
    const notes = [n.gpus ? `GPU ${n.gpus}` : "CPU"];
    if (n.holds_images) notes.push("이미지 있음");
    if (n.transfer) notes.push("이미지 받아 옴");
    if (n.trained_here) notes.push("학습한 서버");
    if (n.id === places.default) notes.push("기본");
    return `${n.name} · ${notes.join(" · ")}${n.fits ? "" : ` — ${n.why}`}`;
  };
  if (!places.nodes.some((n) => n.fits)) {
    return (
      <div className="notice small">
        실행할 수 있는 서버가 없습니다. {places.nodes.map((n) => `${n.name}: ${n.why}`).join(" / ")}
      </div>
    );
  }
  return (
    <label>
      실행할 서버
      <select value={current} onChange={(e) => onChange(e.target.value === places.default ? "" : e.target.value)}>
        {places.nodes.map((n) => (
          <option key={n.id} value={n.id} disabled={!n.fits}>
            {label(n)}
          </option>
        ))}
      </select>
      {chosen?.transfer && (
        <span className="muted small">
          이 서버에는 이 버전의 이미지가 없어 평가 전에 먼저 받습니다. 받은 이미지는 그 서버에 남고, 노드 상세의 받아 둔 버전에서 지울 수
          있습니다.
        </span>
      )}
      {chosen && !chosen.fits && <span className="bad-text small">{chosen.why}</span>}
    </label>
  );
}

export const placeReady = (places: Places | null | undefined, value: string) =>
  !!places && !!places.nodes.find((n) => n.id === (value || places.default) && n.fits);

function useSubmit(onStarted: () => void) {
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (work: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await work();
      onStarted();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };
  return { error, busy, submit };
}

function EvalDialog({ model, onClose, onStarted }: DialogProps) {
  const [tree, setTree] = useState<TreeDataset[] | null>(null);
  const [projectId, setProjectId] = useState(model.project_id); // another project's version of the same task too
  const { data: projects } = usePolling(() => api<{ id: string; name: string }[]>("/projects"), 0);
  const [release, setRelease] = useState(model.release?.id ?? "");
  const [split, setSplit] = useState("val");
  const [counts, setCounts] = useState<Record<string, number>>({});
  const { data: places, error: placesError } = usePlaces(model.id, "eval", release);
  const place = useEvalPlace(places);
  const { error, busy, submit } = useSubmit(onStarted);

  // Offer only the parts this release has, with how many images each holds.
  useEffect(() => {
    if (!release) return;
    void api<ReleaseInfo>(`/releases/${release}`).then((info) => {
      const found: Record<string, number> = {};
      for (const [name, part] of Object.entries(info.stats.splits ?? {})) {
        const key = name === "" ? "train" : name; // images without a split are trained on
        found[key] = (found[key] ?? 0) + part.items;
      }
      setCounts(found);
      setSplit((current) => (found[current] ? current : (["val", "test", "train"].find((k) => found[k]) ?? "val")));
    });
  }, [release]);

  useEffect(() => {
    void api<TreeDataset[]>(`/projects/${projectId}/datasets`).then((data) => {
      setTree(data);
      const ready = data.filter((d) => sameTask(d.task, model.task)).flatMap((d) => d.releases).filter((r) => r.status === "ready");
      setRelease((current) => (ready.some((r) => r.id === current) ? current : (ready[0]?.id ?? "")));
    });
  }, [projectId, model.task]);
  // Only the datasets of the model's task: a Seg model says nothing useful about a Det dataset.
  const fitting = tree?.filter((d) => sameTask(d.task, model.task)) ?? [];
  const hidden = (tree?.length ?? 0) - fitting.length;

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submit(() => api(`/models/${model.id}/eval`, { body: { release_id: release, split, ...place.body } }));
  };

  return (
    <Modal title="모델 평가" onClose={onClose}>
      <form onSubmit={onSubmit}>
        <p className="muted">
          고른 버전으로 이 모델의 지표를 잽니다. 서로 다른 데이터로 학습한 모델도 같은 버전으로 평가하면 공정하게 비교할 수
          있습니다.
        </p>
        {(projects?.length ?? 0) > 1 && (
          <label>
            프로젝트
            <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>
              {projects?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                  {p.id === model.project_id ? " (이 모델의 프로젝트)" : ""}
                </option>
              ))}
            </select>
            {projectId !== model.project_id && (
              <span className="muted small">다른 프로젝트의 버전으로 평가합니다. 같은 태스크이고 모델의 클래스가 그 버전에 있어야 합니다.</span>
            )}
          </label>
        )}
        <div className="form-row" hidden={!!tree && fitting.length === 0}>
          <label>
            버전
            <select value={release} onChange={(e) => setRelease(e.target.value)} required>
              {fitting.map((d) => (
                <optgroup key={d.id} label={d.name}>
                  {d.releases
                    .filter((r) => r.status === "ready" && !r.deleted_at)
                    .map((r) => (
                      <option key={r.id} value={r.id}>
                        {d.name} {r.version}
                        {r.id === model.release?.id ? " (학습에 쓴 버전)" : ""}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label>
            평가할 부분
            <select value={split} onChange={(e) => setSplit(e.target.value)}>
              {["val", "test", "train"]
                .filter((k) => counts[k])
                .map((k) => (
                  <option key={k} value={k}>
                    {k} · {counts[k].toLocaleString()}장
                  </option>
                ))}
            </select>
          </label>
        </div>
        {hidden > 0 && (
          <p className="muted small">
            태스크가 다른 데이터셋 {hidden}개는 뺐습니다({taskLabel(model.task)} 모델은 {taskLabel(model.task)} 데이터셋으로만 평가합니다).
          </p>
        )}
        {fitting.length === 0 && tree && <div className="notice">이 프로젝트에는 이 모델로 평가할 {taskLabel(model.task)} 데이터셋이 없습니다.</div>}
        {release && <EvalPlaceFields places={places} error={placesError} place={place} />}
        {split === "train" && release === model.release?.id && (
          <div className="notice">이 모델이 학습에 쓴 이미지가 들어 있습니다. 지표가 실제보다 높게 나옵니다.</div>
        )}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={busy || !release || !place.ready}>
            평가 시작
          </button>
        </div>
      </form>
    </Modal>
  );
}

function ExportDialog({ model, onClose, onStarted }: DialogProps) {
  const [format, setFormat] = useState("onnx");
  const [node, setNode] = useState("");
  const { data: places, error: placesError } = usePlaces(model.id, "export");
  const { error, busy, submit } = useSubmit(onStarted);
  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submit(() => api(`/models/${model.id}/export`, { body: { format, node_id: node } }));
  };
  return (
    <Modal title="모델 내보내기" onClose={onClose}>
      <form onSubmit={onSubmit}>
        <p className="muted">다른 프로그램이나 기기에서 쓸 수 있는 형식으로 바꿉니다. 끝나면 이 화면에서 내려받을 수 있습니다.</p>
        <label>
          형식
          <select value={format} onChange={(e) => setFormat(e.target.value)}>
            <option value="onnx">ONNX — 대부분의 추론 엔진에서 사용</option>
            <option value="torchscript">TorchScript — PyTorch(C++ 포함)에서 사용</option>
          </select>
        </label>
        <PlacePicker places={places} error={placesError} value={node} onChange={setNode} />
        {model.template.name === "yolo" && (
          <p className="muted small">Ultralytics YOLO는 AGPL-3.0 라이선스입니다. 제품에 탑재하기 전에 라이선스를 확인하세요.</p>
        )}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={busy || !placeReady(places, node)}>
            내보내기
          </button>
        </div>
      </form>
    </Modal>
  );
}

function TestDialog({ model, onClose, onStarted }: DialogProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [conf, setConf] = useState("0.25");
  const [node, setNode] = useState("");
  const { data: places, error: placesError } = usePlaces(model.id, "predict");
  const { error, busy, submit } = useSubmit(onStarted);
  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submit(async () => {
      const images = [];
      for (const file of files) {
        images.push(await apiUpload<{ name: string; blob: string }>(`/uploads?name=${encodeURIComponent(file.name)}`, file, file.type));
      }
      await api(`/models/${model.id}/predict`, { body: { images, conf: Number(conf), node_id: node } });
    });
  };
  return (
    <Modal title="이미지로 테스트" onClose={onClose}>
      <form onSubmit={onSubmit}>
        <p className="muted">이미지를 올리면 이 모델이 찾은 것을 보여 줍니다. 한 번에 20장까지 올릴 수 있습니다.</p>
        <label>
          이미지
          <input
            type="file"
            accept="image/*"
            multiple
            required
            onChange={(e) => setFiles(Array.from(e.target.files ?? []).slice(0, 20))}
          />
        </label>
        {model.task !== "classify" && (
          <label>
            이 값보다 확신이 낮은 것은 버림 (0~1)
            <input type="number" min={0} max={1} step={0.05} value={conf} onChange={(e) => setConf(e.target.value)} />
          </label>
        )}
        <PlacePicker places={places} error={placesError} value={node} onChange={setNode} />
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={busy || files.length === 0 || !placeReady(places, node)}>
            {busy ? "올리는 중…" : `테스트 (${files.length}장)`}
          </button>
        </div>
      </form>
    </Modal>
  );
}

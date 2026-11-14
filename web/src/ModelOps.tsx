import { FileList } from "./Files";
import { type FormEvent, Fragment, useEffect, useState } from "react";
import { api, apiUpload } from "./api";
import type { ReleaseInfo, TreeDataset } from "./catalogTypes";
import { METRIC_LABEL, type ModelDetail, type ModelOp, keyMetrics, metric } from "./trainTypes";
import { Badge, Icon, Modal, STATUS_LABEL, timeAgo, usePolling } from "./ui";

const ACTION_LABEL: Record<string, string> = { eval: "평가", export: "내보내기", predict: "테스트" };
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
            <div className="muted small">학습할 때와 같은 코드와 환경으로 실행합니다.</div>
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
                    <th>릴리스 · split</th>
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
                          {op.release ? `${op.release.dataset} ${op.release.version}` : "삭제된 릴리스"} ·{" "}
                          {SPLIT_LABEL[String(op.params.split)] ?? String(op.params.split)}
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
                          {op.files.length > 0 ? (
                            <button className="link" onClick={() => setShown(shown === op.id ? null : op.id)}>
                              {shown === op.id ? "접기" : `파일 ${op.files.length}개`}
                            </button>
                          ) : (
                            <span className="muted">–</span>
                          )}
                        </td>
                        <td className="muted right">{timeAgo(op.created_at)}</td>
                      </tr>
                      {shown === op.id && (
                        <tr className="op-files">
                          <td colSpan={metricKeys.length + 4}>
                            <FileList files={op.files} />
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
                <span className="muted small">{timeAgo(op.created_at)}</span>
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
  return (
    <>
      <Badge kind={op.status}>{STATUS_LABEL[op.status]}</Badge>
      {op.status === "queued" && op.wait_reason && <div className="wait-reason">{op.wait_reason}</div>}
      {op.status === "failed" && op.error && (
        <div className="wait-reason">
          {op.error} <a href="#/jobs">작업 화면에서 로그 보기</a>
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
  const [release, setRelease] = useState(model.release?.id ?? "");
  const [split, setSplit] = useState("val");
  const [counts, setCounts] = useState<Record<string, number>>({});
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
    void api<TreeDataset[]>(`/projects/${model.project_id}/datasets`).then((data) => {
      setTree(data);
      const ready = data.flatMap((d) => d.releases).filter((r) => r.status === "ready");
      setRelease((current) => (ready.some((r) => r.id === current) ? current : (ready[0]?.id ?? "")));
    });
  }, [model.project_id]);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submit(() => api(`/models/${model.id}/eval`, { body: { release_id: release, split } }));
  };

  return (
    <Modal title="모델 평가" onClose={onClose}>
      <form onSubmit={onSubmit}>
        <p className="muted">
          고른 릴리스로 이 모델의 지표를 잽니다. 서로 다른 데이터로 학습한 모델도 같은 릴리스로 평가하면 공정하게 비교할 수
          있습니다.
        </p>
        <div className="form-row">
          <label>
            릴리스
            <select value={release} onChange={(e) => setRelease(e.target.value)} required>
              {tree?.map((d) => (
                <optgroup key={d.id} label={d.name}>
                  {d.releases
                    .filter((r) => r.status === "ready" && !r.deleted_at)
                    .map((r) => (
                      <option key={r.id} value={r.id}>
                        {d.name} {r.version}
                        {r.id === model.release?.id ? " (학습에 쓴 릴리스)" : ""}
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
        {split === "train" && release === model.release?.id && (
          <div className="notice">이 모델이 학습에 쓴 이미지가 들어 있습니다. 지표가 실제보다 높게 나옵니다.</div>
        )}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={busy || !release}>
            평가 시작
          </button>
        </div>
      </form>
    </Modal>
  );
}

function ExportDialog({ model, onClose, onStarted }: DialogProps) {
  const [format, setFormat] = useState("onnx");
  const { error, busy, submit } = useSubmit(onStarted);
  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submit(() => api(`/models/${model.id}/export`, { body: { format } }));
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
        {model.template.name === "yolo" && (
          <p className="muted small">Ultralytics YOLO는 AGPL-3.0 라이선스입니다. 제품에 탑재하기 전에 라이선스를 확인하세요.</p>
        )}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={busy}>
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
  const { error, busy, submit } = useSubmit(onStarted);
  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submit(async () => {
      const images = [];
      for (const file of files) {
        images.push(await apiUpload<{ name: string; blob: string }>(`/uploads?name=${encodeURIComponent(file.name)}`, file, file.type));
      }
      await api(`/models/${model.id}/predict`, { body: { images, conf: Number(conf) } });
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
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={busy || files.length === 0}>
            {busy ? "올리는 중…" : `테스트 (${files.length}장)`}
          </button>
        </div>
      </form>
    </Modal>
  );
}

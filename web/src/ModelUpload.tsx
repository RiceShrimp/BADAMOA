import { type FormEvent, useEffect, useState } from "react";
import { api, apiForm } from "./api";
import { type Project, bytesLabel, taskLabel } from "./catalogTypes";
import type { ModelDetail } from "./trainTypes";
import { Modal, usePolling } from "./ui";

/** A model kind that can run uploaded Ultralytics weights (GET /api/model-upload/kinds). */
interface UploadKind {
  id: string;
  name: string;
  title: string;
  version: string;
  builtin: boolean;
  tasks: string[];
  requirement: string;
}

/** The sample class yamls the Hub hands out (GET /api/model-upload/sample?kind=…). */
const SAMPLES: { kind: string; label: string; title: string }[] = [
  { kind: "example", label: "예시 (클래스 3개)", title: "내 모델의 클래스로 고쳐 쓰는 예시" },
  { kind: "coco", label: "COCO 80 클래스", title: "yolo11n.pt · yolo11n-seg.pt 같은 Ultralytics 사전학습 가중치용" },
  { kind: "coco-pose", label: "COCO 사람 키포인트", title: "yolo11n-pose.pt 같은 Ultralytics 사전학습 pose 가중치용" },
];

const TASKS: { value: string; label: string }[] = [
  { value: "", label: "가중치에서 읽기" },
  { value: "detect", label: "Det · 객체 검출" },
  { value: "segment", label: "Seg · 인스턴스 분할" },
  { value: "pose", label: "Kpt · 키포인트" },
  { value: "classify", label: "Cls · 분류" },
];

/**
 * The class names of a YOLO data yaml, as the Hub reads them (``names:`` as a list, ``[a, b]`` or ``0: a`` lines),
 * to show the count before anything is sent. The Hub reads the file again and decides.
 */
export function yamlNames(text: string): string[] | null {
  let names: string[] | null = null;
  const numbered: [number, string][] = [];
  let inNames = false;
  const unquote = (v: string) => v.trim().replace(/^['"]|['"]$/g, "");
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.split("#", 1)[0].replace(/\s+$/, "");
    if (!line.trim()) continue;
    if (inNames && /^[\s-]/.test(line)) {
      const item = line.trim();
      if (item.startsWith("-")) (names ??= []).push(unquote(item.slice(1)));
      else if (item.includes(":")) {
        const at = item.indexOf(":");
        numbered.push([Number(item.slice(0, at).trim()), unquote(item.slice(at + 1))]);
      }
      continue;
    }
    inNames = false;
    const at = line.indexOf(":");
    if (at < 0 || line.slice(0, at).trim() !== "names") continue;
    const value = line.slice(at + 1).trim();
    if (value.startsWith("[")) names = value.replace(/^\[|\]$/g, "").split(",").map(unquote).filter(Boolean);
    else if (value.startsWith("{"))
      for (const pair of value.replace(/^\{|\}$/g, "").split(",")) {
        const [k, v] = pair.split(":");
        if (v !== undefined) numbered.push([Number(k.trim()), unquote(v)]);
      }
    else inNames = true;
  }
  if (numbered.length) names = numbered.sort((a, b) => a[0] - b[0]).map(([, v]) => v);
  return names && names.length ? names : null;
}

function stem(fileName: string): string {
  return fileName.replace(/\.[^.]+$/, "");
}

/**
 * 모델 올리기: weights trained elsewhere (an Ultralytics YOLO ``.pt``) with the yaml naming their classes. The Hub reads
 * the class count from the weights and takes the upload only when the yaml has as many; it then evaluates, labels,
 * exports and compares like a model trained here.
 */
export function UploadModel({
  projects,
  projectId,
  onClose,
}: {
  projects: Project[] | null;
  projectId?: string;
  onClose: () => void;
}) {
  const [project, setProject] = useState(projectId ?? "");
  const [weights, setWeights] = useState<File | null>(null);
  const [data, setData] = useState<File | null>(null);
  const [names, setNames] = useState<string[] | null>(null);
  const [name, setName] = useState("");
  const [named, setNamed] = useState(false); // the person typed a name: picking another file keeps it
  const [task, setTask] = useState("");
  const [kind, setKind] = useState("");
  const { data: kinds } = usePolling(() => api<UploadKind[]>("/model-upload/kinds"), 600000);
  const [imgsz, setImgsz] = useState("");
  const [note, setNote] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [sent, setSent] = useState<{ sent: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!project && projects?.length) setProject(projects[0].id);
  }, [project, projects]);
  useEffect(() => {
    let alive = true;
    setNames(null);
    if (data) void data.text().then((text) => alive && setNames(yamlNames(text) ?? []));
    return () => {
      alive = false;
    };
  }, [data]);

  const busy = sent !== null;
  const yamlBad = data !== null && names !== null && names.length === 0;
  const ready = Boolean(project && weights && data && names && names.length && name.trim()) && !busy;

  const pickWeights = (file: File | null) => {
    setWeights(file);
    if (file && !named) setName(stem(file.name));
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!weights || !data) return;
    const form = new FormData();
    form.append("weights", weights);
    form.append("data", data);
    form.append("name", name.trim());
    if (task) form.append("task", task);
    if (kind) form.append("kind", kind);
    if (imgsz.trim()) form.append("imgsz", imgsz.trim());
    if (note.trim()) form.append("note", note.trim());
    setError(null);
    setSent({ sent: 0, total: weights.size + data.size });
    try {
      const model = await apiForm<ModelDetail>(`/projects/${project}/models/upload`, form, (done, total) =>
        setSent({ sent: done, total: total || weights.size + data.size }),
      );
      window.location.hash = `#/models/${model.id}`;
      onClose();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      if (message.includes("태스크") || message.includes("모델 종류") || message.includes("돌리지")) setAdvanced(true);
      setSent(null);
    }
  };

  const percent = sent && sent.total ? Math.min(100, Math.round((sent.sent / sent.total) * 100)) : 0;

  return (
    <Modal title="모델 올리기" onClose={busy ? () => undefined : onClose}>
      <form onSubmit={submit} className="model-upload">
        <p className="muted">
          다른 곳에서 학습했거나 공개된 Ultralytics YOLO 가중치(.pt)를 올려, 여기서 학습한 모델처럼 평가 · 검수 · 오토라벨 · 내보내기 ·
          비교 · 파이프라인에 씁니다. 지금은 YOLO 모델 종류(기본 제공{kinds && kinds.length > 1 ? ` · ${kinds.slice(1).map((k) => k.title.split("(")[0].trim()).join(" · ")}` : ""})로 돌리는 가중치만 올립니다.
        </p>
        {projectId === undefined && (
          <label>
            프로젝트
            <select value={project} onChange={(e) => setProject(e.target.value)} required disabled={busy}>
              {projects?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.folder ? `${p.folder} / ` : ""}
                  {p.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <label>
          가중치 (.pt)
          <input
            type="file"
            accept=".pt,.pth"
            required
            disabled={busy}
            onChange={(e) => pickWeights(e.target.files?.[0] ?? null)}
          />
          {weights && <span className="muted small">{bytesLabel(weights.size)}</span>}
        </label>
        <label>
          클래스 정보 yaml <span className="muted small">(필수)</span>
          <input
            type="file"
            accept=".yaml,.yml"
            required
            disabled={busy}
            onChange={(e) => setData(e.target.files?.[0] ?? null)}
          />
          {names && names.length > 0 && (
            <span className="muted small">
              클래스 {names.length}개: {names.slice(0, 6).join(", ")}
              {names.length > 6 ? ` 외 ${names.length - 6}개` : ""}
            </span>
          )}
        </label>
        {yamlBad && <div className="error">이 yaml에서 클래스 이름(names:)을 찾지 못했습니다.</div>}
        <div className="notice small upload-yaml-guide">
          <strong>yaml도 같이 올려야 합니다.</strong> YOLO 데이터셋을 올릴 때의 data.yaml과 같은 형식으로, names: 아래에 모델의 클래스를
          번호 순서대로 적습니다. 학습에 쓴 data.yaml을 그대로 올려도 됩니다. 가중치의 클래스 수와 yaml의 클래스 수가 같아야 올라갑니다.
          <div className="sample-links">
            샘플 yaml 받기:{" "}
            {SAMPLES.map((s, i) => (
              <span key={s.kind}>
                {i > 0 && " · "}
                <a href={`/api/model-upload/sample?kind=${s.kind}`} download title={s.title}>
                  {s.label}
                </a>
              </span>
            ))}
          </div>
        </div>
        <label>
          모델 이름
          <input
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setNamed(true);
            }}
            maxLength={128}
            required
            disabled={busy}
            placeholder="가중치 파일 이름"
          />
          <span className="muted small">이 프로젝트에 같은 이름의 모델이 있으면 그 모델의 다음 버전이 됩니다.</span>
        </label>
        <details className="advanced" open={advanced} onToggle={(e) => setAdvanced((e.target as HTMLDetailsElement).open)}>
          <summary>세부 설정</summary>
          <label>
            태스크
            <select value={task} onChange={(e) => setTask(e.target.value)} disabled={busy}>
              {TASKS.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
            <span className="muted small">보통 가중치에 적혀 있습니다. 적혀 있지 않을 때만 고르세요.</span>
          </label>
          <label>
            모델 종류 <span className="muted small">(이 가중치를 돌릴 코드)</span>
            <select value={kind} onChange={(e) => setKind(e.target.value)} disabled={busy}>
              <option value="">자동 · 가중치를 저장한 ultralytics 버전으로 고름</option>
              {kinds?.map((k) => (
                <option key={k.id} value={k.name} disabled={task !== "" && !k.tasks.includes(task)}>
                  {k.title} {k.version}
                  {k.builtin ? " · 기본 제공" : ""} · {k.tasks.map(taskLabel).join(" · ")}
                </option>
              ))}
            </select>
            <span className="muted small">
              자동은 포크로 저장한 가중치(예: DCL의 ultralytics 8.4.138.1)를 그 종류로, 나머지는 기본 제공 Ultralytics YOLO로 돌립니다.
            </span>
          </label>
          <label>
            이미지 크기 (px)
            <input
              type="number"
              min={32}
              max={8192}
              step={32}
              value={imgsz}
              onChange={(e) => setImgsz(e.target.value)}
              placeholder="가중치에서 읽기"
              disabled={busy}
            />
          </label>
          <label>
            설명
            <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="어디서 학습했는지, 무엇에 쓰는지" disabled={busy} />
          </label>
        </details>
        {busy && (
          <div className="upload-progress">
            <progress max={100} value={percent} />
            <span className="muted small">
              {percent < 100
                ? `올리는 중 ${percent}% (${bytesLabel(sent.sent)} / ${bytesLabel(sent.total)})`
                : "가중치의 클래스 수를 yaml과 맞춰 보는 중…"}
            </span>
          </div>
        )}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose} disabled={busy}>
            취소
          </button>
          <button className="primary" disabled={!ready}>
            {busy ? "올리는 중…" : "올리기"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
